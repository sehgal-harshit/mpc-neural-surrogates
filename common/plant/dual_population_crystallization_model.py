import casadi as ca


def add_chemistry_states(model, config):
    """Add dual-population crystallization states to the COBR model."""
    reactor = config["reactor_properties"]["reactor"]
    n_elements = reactor["number_tanks_in_series"]

    for prefix in ["fine", "coarse"]:
        model.set_variable("_x", f"{prefix}_zeroth_moment", shape=(n_elements, 1))
        model.set_variable("_x", f"{prefix}_first_moment", shape=(n_elements, 1))
        model.set_variable("_x", f"{prefix}_second_moment", shape=(n_elements, 1))

    model.set_variable("_x", "concentration_solute_dissolved", shape=(n_elements, 1))
    model.set_variable("_x", "concentration_solute_solid_fines", shape=(n_elements, 1))
    model.set_variable("_x", "concentration_solute_solid_coarse", shape=(n_elements, 1))


def add_chemistry_parameters(model, config):
    """Add model parameters for the dual-population crystallization model."""
    model.set_variable("_p", "k_primary_birth")
    model.set_variable("_p", "exp_primary_birth")
    model.set_variable("_p", "k_secondary_birth")
    model.set_variable("_p", "exp_secondary_birth")
    model.set_variable("_p", "k_attrition_birth")
    model.set_variable("_p", "exp_attrition_birth")

    model.set_variable("_p", "k_growth_fines")
    model.set_variable("_p", "exp_growth_fines")
    model.set_variable("_p", "k_growth_coarse")
    model.set_variable("_p", "exp_growth_coarse")

    model.set_variable("_p", "k_attrition_mass")
    model.set_variable("_p", "exp_attrition_mass")
    model.set_variable("_p", "k_attrition_moments")

    model.set_variable("_p", "nucleation_size_fines")
    model.set_variable("_p", "solubility_A")
    model.set_variable("_p", "solubility_B")
    model.set_variable("_p", "rho_crystal")
    model.set_variable("_p", "mw_solute")
    model.set_variable("_p", "k_v")
    model.set_variable("_p", "smoothing_parameter")

    for state_name in [
        "fine_zeroth_moment",
        "fine_first_moment",
        "fine_second_moment",
        "coarse_zeroth_moment",
        "coarse_first_moment",
        "coarse_second_moment",
        "concentration_solute_dissolved",
        "concentration_solute_solid_fines",
        "concentration_solute_solid_coarse",
    ]:
        model.set_variable("_p", f"inflow_{state_name}")

    reactor_props = config["reactor_properties"]
    n_side_inlets = len(reactor_props.get("side_inlets", []))
    if n_side_inlets > 0:
        for state_name in [
            "fine_zeroth_moment",
            "fine_first_moment",
            "fine_second_moment",
            "coarse_zeroth_moment",
            "coarse_first_moment",
            "coarse_second_moment",
            "concentration_solute_dissolved",
            "concentration_solute_solid_fines",
            "concentration_solute_solid_coarse",
        ]:
            model.set_variable("_p", f"side_inlet_{state_name}", shape=(n_side_inlets, 1))


def add_auxiliary_expressions(model, config):
    """Add solubility, supersaturation, and population-specific kinetic expressions."""
    reactor = config["reactor_properties"]["reactor"]
    n_elements = reactor["number_tanks_in_series"]

    def smooth_max(x, y, eps):
        return (x + y + ca.sqrt((x - y) ** 2 + 2 * eps**2)) / 2

    eps = model.p["smoothing_parameter"]

    solubility = ca.MX.zeros(n_elements, 1)
    supersaturation = ca.MX.zeros(n_elements, 1)
    growth_rate_fines = ca.MX.zeros(n_elements, 1)
    growth_rate_coarse = ca.MX.zeros(n_elements, 1)
    primary_birth_rate = ca.MX.zeros(n_elements, 1)
    secondary_birth_rate = ca.MX.zeros(n_elements, 1)
    attrition_birth_rate = ca.MX.zeros(n_elements, 1)
    total_fine_birth_rate = ca.MX.zeros(n_elements, 1)
    attrition_mass_rate = ca.MX.zeros(n_elements, 1)
    attrition_moment_rate = ca.MX.zeros(n_elements, 1)
    crystallization_rate_fines = ca.MX.zeros(n_elements, 1)
    crystallization_rate_coarse = ca.MX.zeros(n_elements, 1)
    fine_mean_size = ca.MX.zeros(n_elements, 1)
    coarse_mean_size = ca.MX.zeros(n_elements, 1)
    total_solid = ca.MX.zeros(n_elements, 1)
    fine_mass_fraction = ca.MX.zeros(n_elements, 1)

    for i in range(n_elements):
        temperature_c = model.x["T_reactor"][i] - 273.15
        solubility[i] = model.p["solubility_A"] * ca.exp(
            model.p["solubility_B"] * temperature_c
        )

        raw_ss = (
            model.x["concentration_solute_dissolved"][i] - solubility[i]
        ) / (solubility[i] + 1.0e-12)
        supersaturation[i] = smooth_max(0.0, raw_ss, eps)

        growth_rate_fines[i] = (
            model.p["k_growth_fines"]
            * supersaturation[i] ** model.p["exp_growth_fines"]
        )
        growth_rate_coarse[i] = (
            model.p["k_growth_coarse"]
            * supersaturation[i] ** model.p["exp_growth_coarse"]
        )

        primary_birth_rate[i] = (
            model.p["k_primary_birth"]
            * supersaturation[i] ** model.p["exp_primary_birth"]
        )
        secondary_birth_rate[i] = (
            model.p["k_secondary_birth"]
            * supersaturation[i] ** model.p["exp_secondary_birth"]
            * model.x["coarse_second_moment"][i]
        )
        attrition_birth_rate[i] = (
            model.p["k_attrition_birth"]
            * supersaturation[i] ** model.p["exp_attrition_birth"]
            * model.x["concentration_solute_solid_coarse"][i]
        )
        total_fine_birth_rate[i] = (
            primary_birth_rate[i] + secondary_birth_rate[i] + attrition_birth_rate[i]
        )

        attrition_mass_rate[i] = (
            model.p["k_attrition_mass"]
            * supersaturation[i] ** model.p["exp_attrition_mass"]
            * model.x["concentration_solute_solid_coarse"][i]
        )
        attrition_moment_rate[i] = (
            model.p["k_attrition_moments"]
            * supersaturation[i] ** model.p["exp_attrition_mass"]
        )

        crystallization_rate_fines[i] = (
            3.0
            * growth_rate_fines[i]
            * model.x["fine_second_moment"][i]
            * model.p["rho_crystal"]
            / model.p["mw_solute"]
            * model.p["k_v"]
        )
        crystallization_rate_coarse[i] = (
            3.0
            * growth_rate_coarse[i]
            * model.x["coarse_second_moment"][i]
            * model.p["rho_crystal"]
            / model.p["mw_solute"]
            * model.p["k_v"]
        )

        fine_mean_size[i] = model.x["fine_first_moment"][i] / (
            model.x["fine_zeroth_moment"][i] + 1.0e-12
        )
        coarse_mean_size[i] = model.x["coarse_first_moment"][i] / (
            model.x["coarse_zeroth_moment"][i] + 1.0e-12
        )
        total_solid[i] = (
            model.x["concentration_solute_solid_fines"][i]
            + model.x["concentration_solute_solid_coarse"][i]
        )
        fine_mass_fraction[i] = model.x["concentration_solute_solid_fines"][i] / (
            total_solid[i] + 1.0e-12
        )

    model.set_expression("concentration_solute_solubility", solubility)
    model.set_expression("supersaturation", supersaturation)
    model.set_expression("growth_rate_fines", growth_rate_fines)
    model.set_expression("growth_rate_coarse", growth_rate_coarse)
    model.set_expression("primary_birth_rate", primary_birth_rate)
    model.set_expression("secondary_birth_rate", secondary_birth_rate)
    model.set_expression("attrition_birth_rate", attrition_birth_rate)
    model.set_expression("total_fine_birth_rate", total_fine_birth_rate)
    model.set_expression("attrition_mass_rate", attrition_mass_rate)
    model.set_expression("attrition_moment_rate", attrition_moment_rate)
    model.set_expression("crystallization_rate_fines", crystallization_rate_fines)
    model.set_expression("crystallization_rate_coarse", crystallization_rate_coarse)
    model.set_expression("fine_mean_size", fine_mean_size)
    model.set_expression("coarse_mean_size", coarse_mean_size)
    model.set_expression("concentration_solute_solid_total", total_solid)
    model.set_expression("fine_mass_fraction", fine_mass_fraction)


def set_chemistry_rhs(model, config):
    """Set the RHS equations for the dual-population crystallization states."""
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

    d_states = {
        "fine_zeroth_moment": ca.MX.zeros(n_elements, 1),
        "fine_first_moment": ca.MX.zeros(n_elements, 1),
        "fine_second_moment": ca.MX.zeros(n_elements, 1),
        "coarse_zeroth_moment": ca.MX.zeros(n_elements, 1),
        "coarse_first_moment": ca.MX.zeros(n_elements, 1),
        "coarse_second_moment": ca.MX.zeros(n_elements, 1),
        "concentration_solute_dissolved": ca.MX.zeros(n_elements, 1),
        "concentration_solute_solid_fines": ca.MX.zeros(n_elements, 1),
        "concentration_solute_solid_coarse": ca.MX.zeros(n_elements, 1),
    }

    l_birth = model.p["nucleation_size_fines"]

    for i in range(n_elements):
        fine_birth = model.aux["total_fine_birth_rate"][i]
        coarse_attrition = model.aux["attrition_moment_rate"][i]

        d_states["fine_zeroth_moment"][i] = fine_birth + convective_transfer(
            "fine_zeroth_moment",
            i,
            "inflow_fine_zeroth_moment",
            "side_inlet_fine_zeroth_moment",
        )
        d_states["fine_first_moment"][i] = (
            model.aux["growth_rate_fines"][i] * model.x["fine_zeroth_moment"][i]
            + fine_birth * l_birth
            + convective_transfer(
                "fine_first_moment",
                i,
                "inflow_fine_first_moment",
                "side_inlet_fine_first_moment",
            )
        )
        d_states["fine_second_moment"][i] = (
            2.0
            * model.aux["growth_rate_fines"][i]
            * model.x["fine_first_moment"][i]
            + fine_birth * l_birth**2
            + convective_transfer(
                "fine_second_moment",
                i,
                "inflow_fine_second_moment",
                "side_inlet_fine_second_moment",
            )
        )

        d_states["coarse_zeroth_moment"][i] = (
            -coarse_attrition * model.x["coarse_zeroth_moment"][i]
            + convective_transfer(
                "coarse_zeroth_moment",
                i,
                "inflow_coarse_zeroth_moment",
                "side_inlet_coarse_zeroth_moment",
            )
        )
        d_states["coarse_first_moment"][i] = (
            model.aux["growth_rate_coarse"][i] * model.x["coarse_zeroth_moment"][i]
            - coarse_attrition * model.x["coarse_first_moment"][i]
            + convective_transfer(
                "coarse_first_moment",
                i,
                "inflow_coarse_first_moment",
                "side_inlet_coarse_first_moment",
            )
        )
        d_states["coarse_second_moment"][i] = (
            2.0
            * model.aux["growth_rate_coarse"][i]
            * model.x["coarse_first_moment"][i]
            - coarse_attrition * model.x["coarse_second_moment"][i]
            + convective_transfer(
                "coarse_second_moment",
                i,
                "inflow_coarse_second_moment",
                "side_inlet_coarse_second_moment",
            )
        )

        d_states["concentration_solute_dissolved"][i] = (
            -model.aux["crystallization_rate_fines"][i]
            -model.aux["crystallization_rate_coarse"][i]
            + convective_transfer(
                "concentration_solute_dissolved",
                i,
                "inflow_concentration_solute_dissolved",
                "side_inlet_concentration_solute_dissolved",
            )
        )
        d_states["concentration_solute_solid_fines"][i] = (
            model.aux["crystallization_rate_fines"][i]
            + model.aux["attrition_mass_rate"][i]
            + convective_transfer(
                "concentration_solute_solid_fines",
                i,
                "inflow_concentration_solute_solid_fines",
                "side_inlet_concentration_solute_solid_fines",
            )
        )
        d_states["concentration_solute_solid_coarse"][i] = (
            model.aux["crystallization_rate_coarse"][i]
            - model.aux["attrition_mass_rate"][i]
            + convective_transfer(
                "concentration_solute_solid_coarse",
                i,
                "inflow_concentration_solute_solid_coarse",
                "side_inlet_concentration_solute_solid_coarse",
            )
        )

    for state_name, rhs in d_states.items():
        model.set_rhs(state_name, rhs)


def add_dual_population_crystallization(model, config):
    """Attach the dual-population crystallization model to the COBR model."""
    if "kinetic" not in config["model_parameters"]:
        raise ValueError(
            "Chemistry parameters not found in config. Please add a 'kinetic' section."
        )

    add_chemistry_states(model, config)
    add_chemistry_parameters(model, config)
    add_auxiliary_expressions(model, config)
    set_chemistry_rhs(model, config)
    return model
