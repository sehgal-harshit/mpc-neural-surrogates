import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "MSA_NARX_MPC", "mpc"))   # for mpc_msa_utils

import numpy as np
import torch

from common.mpc_common import NARXWindowManager, load_scaler
from common.plant_interface import COBRPlant
from common.closed_loop import run_closed_loop, prime_window_from_plant
from MSA_NARX_MPC.mpc.mpc_msa_utils import solve_msa_step, warmstart_shift
import yaml

# narx_model_full.pt was pickled when the network classes lived in the `helpers`
# package under MSA_NARX_MPC/training/. Post-reorg they are in common/shared_helpers,
# so alias the old module path before torch.load can resolve the pickled class.
import common.shared_helpers as _shared_helpers
import common.shared_helpers.helper_classes_MSA as _hc_msa
sys.modules.setdefault("helpers", _shared_helpers)
sys.modules.setdefault("helpers.helper_classes_MSA", _hc_msa)

CONFIG = os.path.join(REPO, "Data_Sampling/sampling/configs/thermal_cobr_config.yaml")
MODEL_DIR = os.path.join(REPO, "MSA_NARX_MPC/training/Models_MSA/version_3")
METADATA = os.path.join(MODEL_DIR, "model_metadata.yml")

N_PAST = 146
M = 15
N_STEPS = 20
T_SP = 330.0
FLOW_PHYS = 4.1667e-6
SETPOINT_LADDER = [363.15, 353.15, 343.15, 333.15, 323.15, 313.15, 303.15, 293.15]


def make_msa_solver(msa_model, feat_scaler, label_scaler, inp_feat_scaler,
                    T_sp=330.0, M=15, flow_phys=FLOW_PHYS, max_iter=100, **w):
    flow_sc = (flow_phys - inp_feat_scaler['mean'][0]) / inp_feat_scaler['std'][0]

    def solver(window, u_warm, uq_band):                       # uq_band ignored (nominal MSA)
        flow_future_sc = np.full(M - 1, flow_sc)
        u_next_sc, U_opt_sc, obj = solve_msa_step(
            msa_model, window, flow_future_sc, feat_scaler, label_scaler, inp_feat_scaler,
            T_sp=T_sp, U_warm=u_warm, M=M, max_iter=max_iter, **w)
        setpt_phys = u_next_sc * inp_feat_scaler['std'][1:9] + inp_feat_scaler['mean'][1:9]
        u_apply = {'flow_inlet': flow_phys, 'T_setpoint': setpt_phys}
        U_full = np.stack([np.concatenate([[flow_sc], U_opt_sc[i*8:(i+1)*8]])
                           for i in range(M - 1)]).flatten()
        msa_in = np.concatenate([window, U_full])
        with torch.no_grad():
            Y = msa_model(torch.tensor(msa_in.reshape(1, -1), dtype=torch.float32)).numpy().reshape(M, 26)
        pred_phys = Y * label_scaler['std'] + label_scaler['mean']
        return u_apply, U_opt_sc, {'objective': obj, 'pred_traj': pred_phys}

    return solver


def test_closed_loop_msa_smoke():
    msa_model = torch.load(os.path.join(MODEL_DIR, "narx_model_full.pt"),
                           map_location='cpu', weights_only=False)
    msa_model.eval()
    feat_scaler = load_scaler(os.path.join(MODEL_DIR, "msa_feature_scaler.yml"))
    label_scaler = load_scaler(os.path.join(MODEL_DIR, "label_scaler.yml"))
    inp_feat_scaler = load_scaler(os.path.join(MODEL_DIR, "input_feature_scaler.yml"))
    with open(METADATA) as f:
        meta = yaml.safe_load(f)

    plant = COBRPlant(CONFIG, METADATA, dt=15.0)
    plant.reset(warmup_s=3600.0)

    wm = NARXWindowManager(meta['dataset_metadata'])
    u_nominal = {'flow_inlet': FLOW_PHYS, 'T_setpoint': np.array(SETPOINT_LADDER)}
    prime_window_from_plant(plant, wm, N_PAST, u_nominal, feat_scaler)

    solver = make_msa_solver(msa_model, feat_scaler, label_scaler, inp_feat_scaler, T_sp=T_SP, M=M)

    res = run_closed_loop(
        plant, solver, wm, n_steps=N_STEPS, setpoint_target=T_SP,
        constraint=[{'name': 'T_max', 'output_index': 25, 'limit': 380.0, 'sense': 'upper'}],
        warmstart_fn=lambda U: warmstart_shift(U, 14, 8), log=True,
    )

    assert res['true_meas'].shape == (N_STEPS, 26)
    assert res['u_applied'].shape == (N_STEPS, 9)
    assert res['pred_traj'].shape == (N_STEPS, M, 26)
    assert res['solve_time'].shape == (N_STEPS,)
    assert res['objective'].shape == (N_STEPS,)
    assert res['violation'].shape == (N_STEPS, 1)
    assert res['constraint_margin'].shape == (N_STEPS, 1)
    assert np.all(res['solve_time'] > 0)
    m = res['metrics']
    assert np.isfinite(m['tracking_rmse'])
    assert 0.0 <= m['violation_rate_overall'] <= 1.0
    print("OK", m)


if __name__ == "__main__":
    test_closed_loop_msa_smoke()
