import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import do_mpc

from common.plant.base_cobr_model import get_base_COBR_model
from common.plant_interface import COBRPlant

CONFIG = "Data_Sampling/sampling/configs/thermal_cobr_config.yaml"
METADATA = "MSA_NARX_MPC/training/Models_MSA/version_3/model_metadata.yml"

DT = 15.0
WARMUP_S = 3600.0
N_STEPS = 200
H_REACTOR = 2200.0
H_LOSS = 12.0


def _build_reference_sim(cfg, h_reactor, h_loss, dt):
    """Direct do-mpc simulator built straight from base_cobr_model — the ground truth."""
    th = cfg['model_parameters']['thermal']
    pid = th['pid_control']
    sp = cfg['system_properties']
    sm = cfg['simulation_settings']['smoothing']
    mv = [t['max_vol_flow'] for t in cfg['reactor_properties']['thermostats']]
    tvp_cfg = cfg['operating_conditions']['tvp']

    model = get_base_COBR_model(cfg)
    model.setup()
    sim = do_mpc.simulator.Simulator(model)
    sim.settings.t_step = dt

    p = sim.get_p_template()
    p['h_reactor'] = h_reactor
    p['h_jacket'] = th['heat_transfer']['jacket_heat_transfer_coefficient']
    p['h_loss'] = h_loss
    p['K_p'] = np.array(pid['proportional_gain']).reshape(-1, 1)
    p['K_i'] = np.array(pid['integral_gain']).reshape(-1, 1)
    p['max_inflow_multipliers'] = np.array(pid['max_inflow_multipliers']).reshape(-1, 1)
    p['heating_power_avg_time_constant'] = np.array(
        pid['heating_power_avg_time_constant']).reshape(-1, 1)
    p['max_vol_flow_thermostats'] = np.array(mv).reshape(-1, 1)
    p['rho_reaction_mixture'] = sp['densities']['reaction_mixture']
    p['cp_reaction_mixture'] = sp['heat_capacities']['reaction_mixture']
    p['rho_oil'] = sp['densities']['oil']
    p['cp_oil'] = sp['heat_capacities']['oil']
    p['rho_steel'] = sp['densities']['steel']
    p['cp_steel'] = sp['heat_capacities']['steel']
    p['smoothing_factor_heating'] = sm['smoothing_factor_heating']
    p['anti_windup_factor'] = sm['anti_windup_factor']
    p['smoothing_factor_integral'] = sm['smoothing_factor_integral']
    sim.set_p_fun(lambda t: p)

    tvp = sim.get_tvp_template()

    def tvp_fun(t):
        tvp['T_flow_inlet'] = tvp_cfg['T_flow_inlet']
        tvp['T_environment'] = tvp_cfg['T_environment']
        return tvp

    sim.set_tvp_fun(tvp_fun)
    sim.setup()
    return sim, model


def _ref_x0(cfg, model):
    ic = cfg['initial_conditions']
    n_elem = cfg['reactor_properties']['reactor']['number_tanks_in_series']
    length = cfg['reactor_properties']['reactor']['length']
    z = np.linspace(0, length, n_elem)

    def prof(key):
        (z0, t0), (z1, t1) = ic[key]
        return np.interp(z, [z0, z1], [t0, t1])

    x0 = model._x(0)
    x0['T_reactor'] = prof('temperature_reactor').reshape(-1, 1)
    x0['T_jacket'] = prof('temperature_jacket').reshape(-1, 1)
    x0['T_inner_wall'] = prof('temperature_inner_wall').reshape(-1, 1)
    x0['T_outer_wall'] = prof('temperature_outer_wall').reshape(-1, 1)
    x0['T_thermostat'] = np.array(ic['temperature_thermostat']).reshape(-1, 1)
    x0['integral_term'] = np.array(ic['integral_term']).reshape(-1, 1)
    x0['heating_power_avg'] = np.array(ic['heating_power_avg']).reshape(-1, 1)
    return x0


def _u_vec(model, flow, setpoints, n_th):
    ut = model._u(0)
    ut['flow_inlet'] = flow
    for j in range(n_th):
        ut[f'T_setpoint_thermostat_{j}'] = setpoints[j]
    return np.array(ut.cat)


def test_plant_matches_direct_run():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    n_th = len(cfg['reactor_properties']['thermostats'])
    nom = cfg['operating_conditions']['inputs']
    flow = nom['flow_inlet']
    setpoints = list(nom['T_setpoint_thermostats'])

    pos = cfg['measurements']['temperature_reactor']['positions']
    n_elem = cfg['reactor_properties']['reactor']['number_tanks_in_series']
    length = cfg['reactor_properties']['reactor']['length']
    r_idx = [int(np.clip(p * n_elem / length, 0, n_elem - 1)) for p in pos]

    # --- wrapper ---
    plant = COBRPlant(CONFIG, METADATA, dt=DT, h_reactor=H_REACTOR, h_loss=H_LOSS)
    plant.reset(warmup_s=WARMUP_S)

    # --- reference: identical x0 + identical warmup, then driven the same ---
    sim, model = _build_reference_sim(cfg, H_REACTOR, H_LOSS, DT)
    sim.reset_history()
    sim.t0 = 0.0
    sim.x0 = _ref_x0(cfg, model)
    u_hold = _u_vec(model, flow, setpoints, n_th)
    for _ in range(int(round(WARMUP_S / DT))):
        sim.make_step(u_hold)
    x_warm = sim.x0
    sim.reset_history()
    sim.t0 = 0.0
    sim.x0 = x_warm

    u = {'flow_inlet': flow, 'T_setpoint': np.array(setpoints)}
    plant_traj = np.zeros((N_STEPS, len(r_idx)))
    ref_traj = np.zeros((N_STEPS, len(r_idx)))
    for k in range(N_STEPS):
        plant_traj[k] = plant.step(u)['T_reactor']
        sim.make_step(u_hold)
        ref_traj[k] = np.array(sim.x0['T_reactor']).reshape(-1)[r_idx]

    assert np.max(np.abs(plant_traj - ref_traj)) < 1e-6

    # layout / extraction sanity
    assert len(plant.measurement_layout) in (25, 26)
    m = plant.step(u)
    assert plant.measurement_vector(m).shape[0] == len(plant.measurement_layout)
    assert m['T_reactor_max'] >= m['T_reactor'].max()


if __name__ == "__main__":
    test_plant_matches_direct_run()
    print("OK")
