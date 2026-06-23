import re

import numpy as np
import yaml
import do_mpc

from common.plant.base_cobr_model import get_base_COBR_model

class COBRPlant:
    """Closed-loop wrapper around COBR model,
    mimics offline sampling pipeline"""

    def __init__(self, config_path, metadata_path, dt=15.0, h_reactor=2200.0, h_loss=12.0, seed=None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(metadata_path) as f:
            meta = yaml.safe_load(f)
        
        self.dt=float(dt)
        self.h_reactor=float(h_reactor)
        self.h_loss=float(h_loss)
        self.rng = np.random.default_rng(seed)
        self._layout=list(meta['model_config']['label_names'])

        rp = self.cfg['reactor_properties']
        self.n_elem = rp['reactor']['number_tanks_in_series']
        self.length = rp['reactor']['length']
        self.n_th = len(rp['thermostats'])
        pos = self.cfg['measurements']['temperature_reactor']['positions']
        self._r_idx = [int(np.clip(p * self.n_elem / self.length, 0, self.n_elem - 1))
                       for p in pos]
        
        self.model = get_base_COBR_model(self.cfg)
        self.model.setup()

        self.sim = do_mpc.simulator.Simulator(self.model)
        self.sim.settings.t_step = self.dt
        self._set_params()
        self._set_tvp()
        self.sim.setup()

# --- _p template from model config ----

    def _set_params(self):
        th = self.cfg['model_parameters']['thermal']
        pid = th['pid_control']
        sp = self.cfg['system_properties']
        sm = self.cfg['simulation_settings']['smoothing']
        mv = [t['max_vol_flow'] for t in self.cfg['reactor_properties']['thermostats']]

        p = self.sim.get_p_template()
        p['h_reactor'] = self.h_reactor
        p['h_jacket'] = th['heat_transfer']['jacket_heat_transfer_coefficient']
        p['h_loss'] = self.h_loss
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
        self.sim.set_p_fun(lambda t: p)

    def _set_tvp(self):
        tvp_cfg = self.cfg['operating_conditions']['tvp']
        tvp = self.sim.get_tvp_template()

        def fun(t):
            tvp['T_flow_inlet'] = tvp_cfg['T_flow_inlet']
            tvp['T_environment'] = tvp_cfg['T_environment']
            return tvp

        self.sim.set_tvp_fun(fun)
    
    # --- deterministic initial state from config initial_condition

    def _initial_x0(self):
        ic = self.cfg['initial_conditions']
        z = np.linspace(0, self.length, self.n_elem)

        def prof(key): # [[0, T_hi], [L, T_lo]] -> linear over elements
            (z0, t0), (z1, t1) = ic[key]
            return np.interp(z, [z0, z1], [t0, t1])
        
        x0 = self.sim.model._x(0)
        x0['T_reactor'] = prof('temperature_reactor').reshape(-1, 1)
        x0['T_jacket'] = prof('temperature_jacket').reshape(-1, 1)
        x0['T_inner_wall'] = prof('temperature_inner_wall').reshape(-1, 1)
        x0['T_outer_wall'] = prof('temperature_outer_wall').reshape(-1, 1)
        x0['T_thermostat'] = np.array(ic['temperature_thermostat']).reshape(-1, 1)
        x0['integral_term'] = np.array(ic['integral_term']).reshape(-1, 1)
        x0['heating_power_avg'] = np.array(ic['heating_power_avg']).reshape(-1, 1)
        return x0

    def _u_vec(self, flow, setpoints):
        ut = self.sim.model._u(0)
        ut['flow_inlet'] = flow
        for j in range(self.n_th):
            ut[f'T_setpoint_thermostat_{j}'] = setpoints[j]
        return np.array(ut.cat)
    
    def reset(self, warmup_s=3600.0):
        self.sim.reset_history()
        self.sim.t0 = 0.0
        self.sim.x0 = self._initial_x0()
        nom = self.cfg['operating_conditions']['inputs']
        u_hold = self._u_vec(nom['flow_inlet'], nom['T_setpoint_thermostats'])
        for _ in range(int(round(warmup_s / self.dt))):
            self.sim.make_step(u_hold)
        x_warm = self.sim.x0                # post-warmup steady state
        self.sim.reset_history()
        self.sim.t0 = 0.0
        self.sim.x0 = x_warm
        return self._measure()
    
    def step(self, u):
        flow = float(u['flow_inlet'])
        sp = np.asarray(u['T_setpoint']).reshape(-1)
        assert sp.size == self.n_th
        self.sim.make_step(self._u_vec(flow, sp))
        return self._measure()
    
    def _measure(self):
        x = self.sim.x0                     # current (post-step) state structure
        Tr = np.array(x['T_reactor']).reshape(-1)
        return {
            'T_reactor':         Tr[self._r_idx].copy(),
            'T_thermostat':      np.array(x['T_thermostat']).reshape(-1).copy(),
            'heating_power_avg': np.array(x['heating_power_avg']).reshape(-1).copy(),
            'T_reactor_max':     float(Tr.max()),
        }

    @property
    def measurement_layout(self):
        """Ordered label names (matches the surrogate's output layout)."""
        return list(self._layout)

    # maps each label's base name to the corresponding key in a _measure() dict.
    # 'T_reactor' is the max-reduced reactor label (single value) -> T_reactor_max.
    _LABEL_TO_KEY = {
        'T_reactor_meas':    'T_reactor',
        'T_thermostat_meas': 'T_thermostat',
        'heating_power_avg': 'heating_power_avg',
        'T_reactor':         'T_reactor_max',
    }

    def measurement_vector(self, m):
        """Flatten a measurement dict into a vector ordered by measurement_layout."""
        vals = []
        for name in self._layout:
            base, idx = re.match(r'([A-Za-z_]+)\[(\d+)\]', name).groups()
            key = self._LABEL_TO_KEY[base]
            val = m[key]
            vals.append(float(val) if np.ndim(val) == 0 else float(val[int(idx)]))
        return np.asarray(vals, dtype=float)
    
    
