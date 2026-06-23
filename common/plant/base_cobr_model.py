import do_mpc
import numpy as np
import casadi as ca


def _get_sym_backend(model):
    return ca.SX if getattr(model, 'symvar_type', 'MX') == 'SX' else ca.MX


def get_base_COBR_model(config, no_avg_heat=False, symvar_type='MX'):
    """Create base COBR model with thermal dynamics."""
    model = do_mpc.model.Model('continuous', symvar_type=symvar_type)
    sym = ca.SX if symvar_type == 'SX' else ca.MX
    
    # Extract configuration
    reactor_props = config['reactor_properties']
    n_elements = reactor_props['reactor']['number_tanks_in_series']
    n_jackets = len(reactor_props['jackets'])
    n_thermostats = len(reactor_props['thermostats'])
    
    # States
    model.set_variable('_x', 'T_reactor', shape=(n_elements, 1))
    model.set_variable('_x', 'T_jacket', shape=(n_elements, 1))
    model.set_variable('_x', 'T_inner_wall', shape=(n_elements, 1))
    model.set_variable('_x', 'T_outer_wall', shape=(n_elements, 1))
    model.set_variable('_x', 'T_thermostat', shape=(n_thermostats, 1))
    model.set_variable('_x', 'integral_term', shape=(n_thermostats, 1))
    if not no_avg_heat:
        model.set_variable('_x', 'heating_power_avg', shape=(n_thermostats, 1))

    # Inputs - all scalar for parameter estimation compatibility
    model.set_variable('_u', 'flow_inlet')
    
    # Individual thermostat setpoint inputs
    for j in range(n_thermostats):
        model.set_variable('_u', f'T_setpoint_thermostat_{j}')
    
    # Conditional inputs
    if reactor_props.get('side_inlets'):
        n_side_inlets = len(reactor_props['side_inlets'])
        for j in range(n_side_inlets):
            model.set_variable('_u', f'flow_side_inlet_{j}')
            model.set_variable('_tvp', f'T_side_inlet_{j}')
        flow_side_inlets = sym.zeros(n_side_inlets, 1)
        T_side_inlets = sym.zeros(n_side_inlets, 1)
        for j in range(n_side_inlets):
            flow_side_inlets[j] = model.u[f'flow_side_inlet_{j}']
            T_side_inlets[j] = model.tvp[f'T_side_inlet_{j}']
        model.set_expression('T_side_inlets', T_side_inlets)
        model.set_expression('flow_side_inlets', flow_side_inlets)
    
    if reactor_props.get('microwaves'):
        n_microwaves = len(reactor_props['microwaves'])
        for j in range(n_microwaves):
            model.set_variable('_u', f'power_microwave_{j}')
    
    # Create auxiliary expressions that stack scalar inputs into vectors
    # This maintains compatibility with existing RHS equations
    T_setpoint_thermostats = sym.zeros(n_thermostats, 1)
    for j in range(n_thermostats):
        T_setpoint_thermostats[j] = model.u[f'T_setpoint_thermostat_{j}']
    model.set_expression('T_setpoint_thermostats', T_setpoint_thermostats)
    
    if reactor_props.get('microwaves'):
        power_microwaves = sym.zeros(n_microwaves, 1)
        for j in range(n_microwaves):
            power_microwaves[j] = model.u[f'power_microwave_{j}']
        model.set_expression('power_microwaves', power_microwaves)

    # Time-varying parameters
    model.set_variable('_tvp', 'T_flow_inlet')
    model.set_variable('_tvp', 'T_environment')
    
    # Parameters
    thermal_params = config['model_parameters']['thermal']
    model.set_variable('_p', 'h_reactor')
    model.set_variable('_p', 'h_jacket')
    model.set_variable('_p', 'h_loss')
    model.set_variable('_p', 'K_p', shape=(n_thermostats, 1))
    model.set_variable('_p', 'K_i', shape=(n_thermostats, 1))
    model.set_variable('_p', 'max_inflow_multipliers', shape=(n_thermostats, 1))
    model.set_variable('_p', 'max_vol_flow_thermostats', shape=(n_thermostats, 1))
    
    # Physical properties parameters
    system_props = config['system_properties']
    model.set_variable('_p', 'rho_reaction_mixture')
    model.set_variable('_p', 'cp_reaction_mixture')
    model.set_variable('_p', 'rho_oil')
    model.set_variable('_p', 'cp_oil')
    model.set_variable('_p', 'rho_steel')
    model.set_variable('_p', 'cp_steel')
    
    # Simulation settings parameters
    sim_settings = config['simulation_settings']['smoothing']
    model.set_variable('_p', 'smoothing_factor_heating')
    model.set_variable('_p', 'anti_windup_factor')
    model.set_variable('_p', 'smoothing_factor_integral')
    if not no_avg_heat:
        model.set_variable('_p', 'heating_power_avg_time_constant', shape=(n_thermostats, 1))
    
    # Set thermal RHS equations
    _set_thermal_rhs(model, config, no_avg_heat=no_avg_heat)

    # Temperature measurements setup (simplified)
    _setup_temperature_reactor_measurements(model, config)

    _setup_temperature_thermostat_measurements(model, config)

    _setup_flow_measurements(model, config)
    
    return model

def _setup_temperature_reactor_measurements(model, config):
    """Setup temperature measurements at specified positions."""
    sym = _get_sym_backend(model)

    # Check if measurements are specified
    if 'measurements' not in config or 'temperature_reactor' not in config['measurements']:
        return
    
    meas_config = config['measurements']['temperature_reactor']
    reactor_props = config['reactor_properties']['reactor']
    reactor_length = reactor_props['length']
    n_elements = reactor_props['number_tanks_in_series']
    
    # Get positions
    if 'positions' in meas_config:
        positions = meas_config['positions']
    elif 'positions_fractional' in meas_config:
        positions = [frac * reactor_length for frac in meas_config['positions_fractional']]
    else:
        return
    
    # Get names (auto-generate if not provided)
    names = meas_config.get('names', [f"T_meas_{i}" for i in range(len(positions))])
    measurement_noise = meas_config.get('measurement_noise', True)
    
    # Create measurements
    for i, (name, pos) in enumerate(zip(names, positions)):
        # Convert position to element index
        element_idx = int(np.clip(pos * n_elements / reactor_length, 0, n_elements - 1))
        
        # Add measurement
        model.set_meas(name, model.x['T_reactor'][element_idx], meas_noise=measurement_noise)

    # Only create auxiliary expression if no measurement noise
    if not measurement_noise:
        T_reactor_meas = sym.zeros(len(positions), 1)
        for i, (name, pos) in enumerate(zip(names, positions)):
            element_idx = int(np.clip(pos * n_elements / reactor_length, 0, n_elements - 1))
            T_reactor_meas[i] = model.x['T_reactor'][element_idx]
        model.set_expression('T_reactor_meas', T_reactor_meas)

def _setup_temperature_thermostat_measurements(model, config):
    """Setup temperature measurements for thermostats."""
    sym = _get_sym_backend(model)

    # Check if thermostat measurements are specified
    if 'measurements' not in config or 'temperature_thermostat' not in config['measurements'] or 'temperature_setpoint' not in config['measurements']:
        return
    
    n_thermostats = len(config['reactor_properties']['thermostats'])
    
    if 'temperature_thermostat' in config['measurements']:
        meas_config = config['measurements']['temperature_thermostat']
        
        # Get names (auto-generate if not provided)
        names = meas_config.get('names', [f"T_meas_thermostat_{i}" for i in range(n_thermostats)])
        measurement_noise = meas_config.get('measurement_noise', True)

        # Create measurements
        for i, name in enumerate(names):
            model.set_meas(name, model.x['T_thermostat'][i], meas_noise=measurement_noise)

        # Only create auxiliary expression if no measurement noise
        if not measurement_noise:
            T_thermostat = sym.zeros(n_thermostats, 1)
            for i, name in enumerate(names):
                T_thermostat[i] = model.x['T_thermostat'][i]
            model.set_expression('T_thermostat_meas', T_thermostat)

    # Set temperature setpoint measurement
    if 'temperature_setpoint' in config['measurements']:
        setpoint_config = config['measurements']['temperature_setpoint']
        setpoint_names = setpoint_config.get('names', [f"T_setpoint_thermostat_{i}" for i in range(n_thermostats)])
        measurement_noise = setpoint_config.get('measurement_noise', True)
        
        for i, name in enumerate(setpoint_names):
            model.set_meas(name, model.aux['T_setpoint_thermostats'][i], meas_noise=measurement_noise)

def _setup_flow_measurements(model, config):
    """Setup flow measurements at specified positions."""
    # Check if measurements are specified
    if 'measurements' not in config or 'flow_inlet' not in config['measurements']:
        return
    
    meas_config = config['measurements']['flow_inlet']
    
    # Get names (auto-generate if not provided)
    names = meas_config.get('names', ['flow_inlet'])
    measurement_noise = meas_config.get('measurement_noise', True)
    
    # Create flow measurement
    model.set_meas(names, model.u['flow_inlet'], meas_noise=measurement_noise)
        
        
def _set_thermal_rhs(model, config, no_avg_heat=False):
    """Set RHS equations with correct finite volume discretization."""
    reactor_props = config['reactor_properties']
    n_elements = reactor_props['reactor']['number_tanks_in_series']
    reactor_length = reactor_props['reactor']['length']
    cross_area = reactor_props['reactor']['cross_sectional_area']
    element_length = reactor_length / n_elements
    element_volume = cross_area * element_length
    
    # Helper functions for correct flow calculations
    def get_element_center_position(i):
        return element_length * (i + 0.5)

    def get_inlet_flow_at_element(i):
        """Calculate volumetric flow entering element i from upstream."""
        flow = model.u['flow_inlet']
        if reactor_props.get('side_inlets'):
            for j, inlet in enumerate(reactor_props['side_inlets']):
                inlet_element = int(inlet['position'] / reactor_length * n_elements)
                if inlet_element < i:
                    flow += model.u[f'flow_side_inlet_{j}']
        return flow
    
    def get_outflow_from_element(i):
        """Calculate volumetric flow leaving element i."""
        flow = model.u['flow_inlet']
        if reactor_props.get('side_inlets'):
            for j, inlet in enumerate(reactor_props['side_inlets']):
                inlet_element = int(inlet['position'] / reactor_length * n_elements)
                if inlet_element <= i:
                    flow += model.u[f'flow_side_inlet_{j}']
        return flow
    
    def get_ht_factor(material_type):
        """Get heat transfer coefficient factor with proper scaling."""
        if material_type == 'reaction_mixture':
            return 1 / (model.p['rho_reaction_mixture'] * model.p['cp_reaction_mixture'])
        elif material_type == 'oil':
            return 1 / (model.p['rho_oil'] * model.p['cp_oil'])
        elif material_type == 'steel':
            return 1 / (model.p['rho_steel'] * model.p['cp_steel'])
    
    def smooth_bound(x, lower, upper, k):
        """Smooth bounding function."""
        upper_bounded = -_smooth_max(-x, -upper, k)
        lower_bounded = _smooth_max(upper_bounded, lower, k)
        return lower_bounded
    
    def _smooth_max(x, y, k):
        return (x + y + ca.sqrt((x - y)**2 + 2*k**2)) / 2
    
    def smooth_saturation_indicator(x, lower, upper, epsilon):
        """Smooth saturation indicator function."""
        scaled_x = (x - lower) / (upper - lower)
        return 1/(1 + ca.exp(-1/epsilon * scaled_x)) * 1/(1 + ca.exp(1/epsilon * (scaled_x - 1)))
    
    # Create jacket-element mapping
    jacket_element_map = {}
    jacket_flow_directions = {}
    jacket_inlet_outlets = {}
    
    for j, jacket in enumerate(reactor_props['jackets']):
        elements_in_jacket = []
        for i in range(n_elements):
            pos = get_element_center_position(i)
            if ((pos >= jacket['inlet'] and pos < jacket['outlet']) or 
                (pos < jacket['inlet'] and pos >= jacket['outlet'])):
                elements_in_jacket.append(i)
        
        jacket_element_map[j] = elements_in_jacket
        
        # Determine flow direction
        if jacket['inlet'] < jacket['outlet']:
            jacket_flow_directions[j] = 1  # co-current
            jacket_inlet_outlets[j] = {'inlet_elem': elements_in_jacket[0], 'outlet_elem': elements_in_jacket[-1]}
        else:
            jacket_flow_directions[j] = -1  # counter-current
            jacket_inlet_outlets[j] = {'inlet_elem': elements_in_jacket[-1], 'outlet_elem': elements_in_jacket[0]}
    
    # CORRECTED Reactor temperature RHS with proper finite volume discretization
    sym = _get_sym_backend(model)

    dT_reactor_dt = sym.zeros(n_elements, 1)
    
    for i in range(n_elements):
        rate = 0.0
        conv_coeff = 1 / element_volume
        
        # CORRECTED Convective terms - proper finite volume balance
        # Inflow term
        if i == 0:
            rate += conv_coeff * get_inlet_flow_at_element(i) * model.tvp['T_flow_inlet']
        else:
            rate += conv_coeff * get_inlet_flow_at_element(i) * model.x['T_reactor'][i-1]
        
        # Outflow term (consistent for all elements)
        rate -= conv_coeff * get_outflow_from_element(i) * model.x['T_reactor'][i]
        
        # Side inlet contributions (added separately - no double counting)
        if reactor_props.get('side_inlets'):
            for j, inlet in enumerate(reactor_props['side_inlets']):
                inlet_element = int(inlet['position'] / reactor_length * n_elements)
                if inlet_element == i:
                    rate += conv_coeff * model.aux['flow_side_inlets'][j] * model.aux['T_side_inlets'][j]
        
        # Heat transfer to wall
        rate += (4 * model.p['h_reactor'] / reactor_props['reactor']['inner_diameter'] * 
                get_ht_factor('reaction_mixture') * 
                (model.x['T_inner_wall'][i] - model.x['T_reactor'][i]))
        
        # Microwave heating
        if reactor_props.get('microwaves'):
            for j, mw in enumerate(reactor_props['microwaves']):
                pos = get_element_center_position(i)
                if mw['start'] <= pos < mw['end']:
                    # Count elements in microwave zone
                    n_mw_elements = sum(1 for k in range(n_elements) 
                                       if mw['start'] <= get_element_center_position(k) < mw['end'])
                    # power_density = (model.u['power_microwaves'][j] / n_mw_elements / 
                    #                (model.p['rho_reaction_mixture'] * model.p['cp_reaction_mixture'] * element_volume))
                    power_density = (model.aux['power_microwaves'][j] / n_mw_elements /
                                   (model.p['rho_reaction_mixture'] * model.p['cp_reaction_mixture'] * element_volume))
                    rate += power_density
        
        dT_reactor_dt[i] = rate
    
    model.set_rhs('T_reactor', dT_reactor_dt)
    
    # Jacket temperature RHS (same logic as before)
    dT_jacket_dt = sym.zeros(n_elements, 1)
    
    for i in range(n_elements):
        rate = 0.0
        
        # Find which jacket this element belongs to
        for j in range(len(reactor_props['jackets'])):
            if i in jacket_element_map[j]:
                jacket = reactor_props['jackets'][j]
                conv_coeff = 1 / (jacket['cross_sectional_area'] * element_length)
                
                # Use proper jacket flow rate
                jacket_flow_rate = (model.p['max_vol_flow_thermostats'][j] * 
                                   model.p['max_inflow_multipliers'][j])
                
                # Axial convective heat transfer
                if i == jacket_inlet_outlets[j]['inlet_elem']:
                    # Inlet boundary condition
                    rate += conv_coeff * jacket_flow_rate * (model.x['T_thermostat'][j] - model.x['T_jacket'][i])
                else:
                    # Interior elements - flow direction matters
                    upstream_elem = i - jacket_flow_directions[j]
                    rate += conv_coeff * jacket_flow_rate * (model.x['T_jacket'][upstream_elem] - model.x['T_jacket'][i])
                
                # Heat transfer from inner wall to jacket
                rate += (4 * model.p['h_jacket'] / jacket['inner_diameter'] * 
                        get_ht_factor('oil') * 
                        (model.x['T_inner_wall'][i] - model.x['T_jacket'][i]))
                
                # Heat transfer from jacket to outer wall
                rate += (4 * model.p['h_jacket'] / jacket['outer_diameter'] * 
                        get_ht_factor('oil') * 
                        (model.x['T_outer_wall'][i] - model.x['T_jacket'][i]))
                
                break
        
        dT_jacket_dt[i] = rate
    
    model.set_rhs('T_jacket', dT_jacket_dt)
    
    # Inner wall temperature RHS
    dT_inner_wall_dt = sym.zeros(n_elements, 1)
    
    for i in range(n_elements):
        rate = 0.0
        
        # Heat transfer from reactor to inner wall
        rate += (4 * model.p['h_reactor'] / reactor_props['reactor']['inner_diameter'] * 
                get_ht_factor('steel') * 
                (model.x['T_reactor'][i] - model.x['T_inner_wall'][i]))
        
        # Heat transfer from inner wall to jacket
        rate += (4 * model.p['h_jacket'] / reactor_props['jackets'][0]['inner_diameter'] * 
                get_ht_factor('steel') * 
                (model.x['T_jacket'][i] - model.x['T_inner_wall'][i]))
        
        dT_inner_wall_dt[i] = rate
    
    model.set_rhs('T_inner_wall', dT_inner_wall_dt)
    
    # Outer wall temperature RHS
    dT_outer_wall_dt = sym.zeros(n_elements, 1)
    
    for i in range(n_elements):
        rate = 0.0
        
        # Heat transfer from jacket to outer wall
        rate += (4 * model.p['h_jacket'] / reactor_props['jackets'][0]['outer_diameter'] * 
                get_ht_factor('steel') * 
                (model.x['T_jacket'][i] - model.x['T_outer_wall'][i]))
        
        # Heat loss to environment
        rate += (4 * model.p['h_loss'] / reactor_props['jackets'][0]['outer_diameter'] * 
                get_ht_factor('steel') * 
                (model.tvp['T_environment'] - model.x['T_outer_wall'][i]))
        
        dT_outer_wall_dt[i] = rate
    
    model.set_rhs('T_outer_wall', dT_outer_wall_dt)
    
    # Thermostat dynamics
    heating_powers = sym.zeros(len(reactor_props['thermostats']), 1)
    
    for j in range(len(reactor_props['thermostats'])):
        thermostat = reactor_props['thermostats'][j]
        
        # PID heating power calculation
        # error = model.u['T_setpoint_thermostats'][j] - model.x['T_thermostat'][j]
        error = model.aux['T_setpoint_thermostats'][j] - model.x['T_thermostat'][j]
        heating = (model.p['K_p'][j] * error + 
                  model.p['K_i'][j] * model.x['integral_term'][j])
        
        # Smooth bounds on heating power
        heating_bounded = smooth_bound(heating, 
                                     -thermostat['cooling_power'], 
                                     thermostat['heating_power'],
                                     model.p['smoothing_factor_heating'])
        
        heating_powers[j] = heating_bounded
    
    # Set heating power as auxiliary variable
    model.set_expression('heating_power_thermostats', heating_powers)
    
    # Thermostat temperature and integral term RHS
    dT_thermostat_dt = sym.zeros(len(reactor_props['thermostats']), 1)
    dintegral_dt = sym.zeros(len(reactor_props['thermostats']), 1)
    
    for j in range(len(reactor_props['thermostats'])):
        thermostat = reactor_props['thermostats'][j]
        
        # Thermostat temperature dynamics
        outlet_elem = jacket_inlet_outlets[j]['outlet_elem']
        jacket_flow_rate = (model.p['max_vol_flow_thermostats'][j] * 
                           model.p['max_inflow_multipliers'][j])
        
        rate = (jacket_flow_rate / thermostat['volume'] *
                (model.x['T_jacket'][outlet_elem] - model.x['T_thermostat'][j]))
        
        rate += (heating_powers[j] / (thermostat['volume'] * model.p['rho_oil'] * model.p['cp_oil']))
        
        dT_thermostat_dt[j] = rate
        
        # Integral term with anti-windup
        # error = model.u['T_setpoint_thermostats'][j] - model.x['T_thermostat'][j]
        error = model.aux['T_setpoint_thermostats'][j] - model.x['T_thermostat'][j]
        sat_indicator = smooth_saturation_indicator(
            heating_powers[j],
            -thermostat['cooling_power'] * model.p['anti_windup_factor'],
            thermostat['heating_power'] * model.p['anti_windup_factor'],
            model.p['smoothing_factor_integral']
        )
        
        dintegral_dt[j] = error * sat_indicator
    
    model.set_rhs('T_thermostat', dT_thermostat_dt)
    model.set_rhs('integral_term', dintegral_dt)

    if not no_avg_heat:
        # Heating power moving average RHS
        dheating_power_avg_dt = sym.zeros(len(reactor_props['thermostats']), 1)
        
        for j in range(len(reactor_props['thermostats'])):
            # Exponential moving average: d(avg)/dt = (current_value - avg) / time_constant
            dheating_power_avg_dt[j] = (heating_powers[j] - model.x['heating_power_avg'][j]) / model.p['heating_power_avg_time_constant'][j]
        
        model.set_rhs('heating_power_avg', dheating_power_avg_dt)
    sym = _get_sym_backend(model)

    sym = _get_sym_backend(model)
