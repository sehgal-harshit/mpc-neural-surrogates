import casadi as ca


def add_chemistry_states(model, config):
    """Add simple crystallization states to the COBR model."""
    reactor = config["reactor_properties"]["reactor"]
    n_elements = reactor["number_tanks_in_series"]

    model.set_variable("_x", "zeroth_moment", shape=(n_elements, 1))
    model.set_variable("_x", "first_moment", shape=(n_elements, 1))
    model.set_variable("_x", "second_moment", shape=(n_elements, 1))
    model.set_variable("_x", "concentration_solute_dissolved", shape=(n_elements, 1))
    model.set_variable("_x", "concentration_solute_solid", shape=(n_elements, 1))


def add_chemistry_parameters(model, config):
    """Add model parameters for the lightweight literature crystallization model."""
    model.set_variable("_p", "k_birth")
    model.set_variable("_p", "exp_birth")
    model.set_variable("_p", "activation_energy_birth")

    model.set_variable("_p", "k_growth_1")
    model.set_variable("_p", "exp_growth_1")
    model.set_variable("_p", "activation_energy_growth_1")

    model.set_variable("_p", "k_growth_2")
    model.set_variable("_p", "exp_growth_2")
    model.set_variable("_p", "activation_energy_growth_2")

    model.set_variable("_p", "reference_temperature_kinetics")
    model.set_variable("_p", "solubility_A")
    model.set_variable("_p", "solubility_B")
    model.set_variable("_p", "rho_crystal")
    model.set_variable("_p", "mw_solute")
    model.set_variable("_p", "k_v")
    model.set_variable("_p", "smoothing_parameter")

    model.set_variable("_p", "inflow_zeroth_moment")
    model.set_variable("_p", "inflow_first_moment")
    model.set_variable("_p", "inflow_second_moment")
    model.set_variable("_p", "inflow_concentration_solute_dissolved")
    model.set_variable("_p", "inflow_concentration_solute_solid")

    reactor_props = config["reactor_properties"]
    n_side_inlets = len(reactor_props.get("side_inlets", []))
    if n_side_inlets > 0:
        model.set_variable("_p", "side_inlet_zeroth_moment", shape=(n_side_inlets, 1))
        model.set_variable("_p", "side_inlet_first_moment", shape=(n_side_inlets, 1))
        model.set_variable("_p", "side_inlet_second_moment", shape=(n_side_inlets, 1))
        model.set_variable(
            "_p", "side_inlet_concentration_solute_dissolved", shape=(n_side_inlets, 1)
        )
        model.set_variable(
            "_p", "side_inlet_concentration_solute_solid", shape=(n_side_inlets, 1)
        )


def add_auxiliary_expressions(model, config):
    """Add solubility, supersaturation, and rate expressions."""
    reactor = config["reactor_properties"]["reactor"]
    n_elements = reactor["number_tanks_in_series"]

    def smooth_max(x, y, eps):
        return (x + y + ca.sqrt((x - y) ** 2 + 2 * eps**2)) / 2

    def arrhenius_factor(activation_energy, temperature):
        return ca.exp(
            -activation_energy
            / 8.314
            * (1.0 / temperature - 1.0 / model.p["reference_temperature_kinetics"])
        )

    solubility = ca.MX.zeros(n_elements, 1)
    supersaturation = ca.MX.zeros(n_elements, 1)
    growth_rate_1 = ca.MX.zeros(n_elements, 1)
    growth_rate_2 = ca.MX.zeros(n_elements, 1)
    growth_rate_effective = ca.MX.zeros(n_elements, 1)
    birth_rate = ca.MX.zeros(n_elements, 1)
    crystallization_rate = ca.MX.zeros(n_elements, 1)
    mean_crystal_size = ca.MX.zeros(n_elements, 1)

    eps = model.p["smoothing_parameter"]

    for i in range(n_elements):
        temperature_c = model.x["T_reactor"][i] - 273.15
        solubility[i] = model.p["solubility_A"] * ca.exp(
            model.p["solubility_B"] * temperature_c
        )

        raw_ss = (
            model.x["concentration_solute_dissolved"][i] - solubility[i]
        ) / (solubility[i] + 1.0e-12)
        supersaturation[i] = smooth_max(0.0, raw_ss, eps)

        growth_rate_1[i] = (
            model.p["k_growth_1"]
            * arrhenius_factor(model.p["activation_energy_growth_1"], model.x["T_reactor"][i])
            * supersaturation[i] ** model.p["exp_growth_1"]
        )
        growth_rate_2[i] = (
            model.p["k_growth_2"]
            * arrhenius_factor(model.p["activation_energy_growth_2"], model.x["T_reactor"][i])
            * supersaturation[i] ** model.p["exp_growth_2"]
        )
        growth_rate_effective[i] = 0.5 * (growth_rate_1[i] + growth_rate_2[i])
        birth_rate[i] = (
            model.p["k_birth"]
            * arrhenius_factor(model.p["activation_energy_birth"], model.x["T_reactor"][i])
            * supersaturation[i] ** model.p["exp_birth"]
        )
        crystallization_rate[i] = (
            3.0
            * growth_rate_effective[i]
            * model.x["second_moment"][i]
            * model.p["rho_crystal"]
            / model.p["mw_solute"]
            * model.p["k_v"]
        )
        mean_crystal_size[i] = model.x["first_moment"][i] / (
            model.x["zeroth_moment"][i] + 1.0e-12
        )

    model.set_expression("concentration_solute_solubility", solubility)
    model.set_expression("supersaturation", supersaturation)
    model.set_expression("growth_rate_1", growth_rate_1)
    model.set_expression("growth_rate_2", growth_rate_2)
    model.set_expression("growth_rate_effective", growth_rate_effective)
    model.set_expression("birth_rate", birth_rate)
    model.set_expression("crystallization_rate", crystallization_rate)
    model.set_expression("mean_crystal_size", mean_crystal_size)


def set_chemistry_rhs(model, config):
    """Set the RHS equations for the crystallization states."""
    reactor = config["reactor_properties"]["reactor"]
    reactor_props = config["reactor_properties"]
    n_elements = reactor["number_tanks_in_series"]
    reactor_length = reactor["length"]
    element_volume = reactor["cross_sectional_area"] * reactor_length / n_elements
    conv_coeff = 1.0 / element_volume

    def get_inlet_flow_at_element(i):
        flow = model.u["flow_inlet"]
        if reactor_props.get("side_inlets"):
            for j, inlet in enumerate(reactor_props["side_inlets"]):
                inlet_element = int(inlet["position"] / reactor_length * n_elements)
                if inlet_element < i:
                    flow += model.aux["flow_side_inlets"][j]
        return flow

    def get_outflow_from_element(i):
        flow = model.u["flow_inlet"]
        if reactor_props.get("side_inlets"):
            for j, inlet in enumerate(reactor_props["side_inlets"]):
                inlet_element = int(inlet["position"] / reactor_length * n_elements)
                if inlet_element <= i:
                    flow += model.aux["flow_side_inlets"][j]
        return flow

    def convective_transfer(state_name, i, inflow_param_name, side_inlet_param_name=None):
        state = model.x[state_name]
        rate = -conv_coeff * get_outflow_from_element(i) * state[i]
        if i == 0:
            rate += conv_coeff * get_inlet_flow_at_element(i) * model.p[inflow_param_name]
        else:
            rate += conv_coeff * get_inlet_flow_at_element(i) * state[i - 1]

        if reactor_props.get("side_inlets"):
            for j, inlet in enumerate(reactor_props["side_inlets"]):
                inlet_element = int(inlet["position"] / reactor_length * n_elements)
                if inlet_element == i:
                    side_concentration = (
                        model.p[side_inlet_param_name][j]
                        if side_inlet_param_name
                        else model.p[inflow_param_name]
                    )
                    rate += (
                        conv_coeff
                        * model.aux["flow_side_inlets"][j]
                        * side_concentration
                    )
        return rate

    d_m0 = ca.MX.zeros(n_elements, 1)
    d_m1 = ca.MX.zeros(n_elements, 1)
    d_m2 = ca.MX.zeros(n_elements, 1)
    d_c_dissolved = ca.MX.zeros(n_elements, 1)
    d_c_solid = ca.MX.zeros(n_elements, 1)

    for i in range(n_elements):
        d_m0[i] = model.aux["birth_rate"][i] + convective_transfer(
            "zeroth_moment", i, "inflow_zeroth_moment", "side_inlet_zeroth_moment"
        )
        d_m1[i] = (
            model.aux["growth_rate_effective"][i] * model.x["zeroth_moment"][i]
            + convective_transfer("first_moment", i, "inflow_first_moment", "side_inlet_first_moment")
        )
        d_m2[i] = (
            2.0 * model.aux["growth_rate_effective"][i] * model.x["first_moment"][i]
            + convective_transfer("second_moment", i, "inflow_second_moment", "side_inlet_second_moment")
        )
        d_c_dissolved[i] = (
            -model.aux["crystallization_rate"][i]
            + convective_transfer(
                "concentration_solute_dissolved",
                i,
                "inflow_concentration_solute_dissolved",
                "side_inlet_concentration_solute_dissolved",
            )
        )
        d_c_solid[i] = (
            model.aux["crystallization_rate"][i]
            + convective_transfer(
                "concentration_solute_solid",
                i,
                "inflow_concentration_solute_solid",
                "side_inlet_concentration_solute_solid",
            )
        )

    model.set_rhs("zeroth_moment", d_m0)
    model.set_rhs("first_moment", d_m1)
    model.set_rhs("second_moment", d_m2)
    model.set_rhs("concentration_solute_dissolved", d_c_dissolved)
    model.set_rhs("concentration_solute_solid", d_c_solid)


def add_literature_crystallization(model, config):
    """Attach the lightweight literature crystallization model to the COBR model."""
    if "kinetic" not in config["model_parameters"]:
        raise ValueError(
            "Chemistry parameters not found in config. Please add a 'kinetic' section."
        )

    add_chemistry_states(model, config)
    add_chemistry_parameters(model, config)
    add_auxiliary_expressions(model, config)
    set_chemistry_rhs(model, config)
    return model
