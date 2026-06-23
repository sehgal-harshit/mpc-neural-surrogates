import time, inspect, numpy as np

def _band_value(uq_band, h, idx):
    if uq_band is None:
        return 0.0
    n = len(inspect.signature(uq_band).parameters)
    return float(uq_band(h, idx)) if n >= 2 else float(uq_band(h))

def run_closed_loop(plant, surrogate_solver, window_manager, n_steps, setpoint_target,
                    uq_band=None, constraint=None, warmstart_fn=None, log=False):
    """Receding-horizon loop: solve surrogate -> apply to plant -> TRUE-measurement feedback.

    surrogate_solver(window, u_warm, uq_band) -> (u_apply, U_opt, info)
        window:  scaled (3796,) from window_manager.get_window()
        u_apply: physical dict {'flow_inlet':float,'T_setpoint':(8,)} for plant.step()
        U_opt:   opaque, passed to warmstart_fn next step
        info:    {'objective':float, 'pred_traj':(M,26) physical | None, ...}
    constraint: list of {'name','output_index','limit','sense'('upper'|'lower')}.
    warmstart_fn(U_opt)->U_warm' (None -> pass U_opt through).
    log: False | True (one line/step) | callable(k, record).
    """
    constraints = list(constraint or [])
    L = {k: [] for k in ('true_meas', 'u_applied', 'pred_traj', 'solve_time',
                         'objective', 'constraint_margin', 'violation')}
    u_warm = None
    for k in range(n_steps):
        window = window_manager.get_window()
        t0 = time.perf_counter()
        u_apply, U_opt, info = surrogate_solver(window, u_warm, uq_band)
        solve_time = time.perf_counter() - t0

        meas = plant.step(u_apply)                                  # Advance TRUE plant
        window_manager.push_true_measurement(meas, u_apply)         # TRUE feedback
        u_warm = warmstart_fn(U_opt) if warmstart_fn else U_opt

        meas_vec = plant.measurement_vector(meas)                   # (26,) physical
        pred = info.get('pred_traj')                                # (M,26) physical | None

        margins, viols = [], []
        for c in constraints:
            idx, lim, sense = c['output_index'], c['limit'], c.get('sense', 'upper')
            if pred is not None:
                p = np.asarray(pred)[:, idx]
                b = np.array([_band_value(uq_band, h, idx) for h in range(len(p))])
                m = float(np.min(lim - (p + b))) if sense == 'upper' else float(np.min((p - b) - lim))
            else:
                m = np.nan
            margins.append(m)
            viols.append(bool(meas_vec[idx] > lim) if sense == 'upper' else bool(meas_vec[idx] < lim))

        u_vec = np.concatenate([[float(u_apply['flow_inlet'])], np.asarray(u_apply['T_setpoint'], float).reshape(-1)])
        
        rec = dict(
            true_meas = meas_vec,
            u_applied = u_vec,
            pred_traj = (np.asarray(pred) if pred is not None else None),
            solve_time = solve_time,
            objective = float(info.get('objective', np.nan)),
            constraint_margin = np.array(margins),
            violation = np.array(viols, bool)
        )

        for key in L:
            L[key].append(rec[key])
        if callable(log):
            log(k, rec)
        elif log:
            vtxt = '' if not constraints else f"  viol={int(rec['violation'].any())}"
            print(f"[step {k:3d}] obj={rec['objective']:.3g}  t={solve_time:5.2f}s{vtxt}", flush=True)
        
    results = {
        'true_meas': np.array(L['true_meas']),
        'u_applied': np.array(L['u_applied']),
        'pred_traj': (np.array(L['pred_traj']) if (not constraints or L['pred_traj'][0] is not None) else None),
        'solve_time': np.array(L['solve_time']),
        'objective': np.array(L['objective']),
        'constraint_margin': np.array(L['constraint_margin']) if constraints else np.empty((n_steps, 0)),
        'violation': np.array(L['violation']) if constraints else np.empty((n_steps, 0), bool),
        'constraints': constraints,
        'setpoint_target': setpoint_target,
    }
    results['metrics'] = closed_loop_metrics(results, setpoint_target)
    return results


def closed_loop_metrics(results, setpoint_target=None, reactor_slice=slice(0,9)):
    sp = results['setpoint_target'] if setpoint_target is None else setpoint_target
    Tr = results['true_meas'][:, reactor_slice]                # (n,9) physical
    err = Tr - np.asarray(sp)
    st = results['solve_time']
    viol = results['violation']
    return {
        'tracking_rmse': float(np.sqrt(np.mean(err ** 2))),
        'tracking_rmse_per_zone': np.sqrt(np.mean(err ** 2, axis=0)),
        'tracking_mae': float(np.mean(np.abs(err))),
        'violation_rate': (viol.mean(axis=0) if viol.size else np.array([])),
        'violation_rate_overall': float(viol.any(axis=1).mean()) if viol.size else 0.0,
        'solve_time_mean': float(st.mean()), 'solve_time_max': float(st.max()),
    }


def prime_window_from_plant(plant, window_manager, n_past, u_nominal, feat_scaler):
    """Roll the plant n_past steps under a fixed nominal control, collect per-group physical
    history, init the lag window. Dataset-free, plant-consistent warm start."""
    Tr, Tt = [], []
    for _ in range(n_past):
        m = plant.step(u_nominal)
        Tr.append(np.asarray(m['T_reactor']).reshape(-1))
        Tt.append(np.asarray(m['T_thermostat']).reshape(-1))
    flow = float(u_nominal['flow_inlet']); sp = np.asarray(u_nominal['T_setpoint'], float).reshape(-1)
    obs = {
        'T_reactor_meas': np.array(Tr),                         # (n_past,9)
        'T_thermostat_meas': np.array(Tt),                      # (n_past,8)
        'flow_inlet': np.full((n_past, 1), flow),
        'T_setpoint_thermostats': np.tile(sp, (n_past, 1)),     # (n_past,8)
    }
    window_manager.init_from_obs(obs, feat_scaler)
    window_manager.feat_scaler = feat_scaler                    # used by push_true_measurement