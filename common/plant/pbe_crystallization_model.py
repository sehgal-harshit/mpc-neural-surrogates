from pathlib import Path
import sys

import casadi as ca
import numpy as np


DO_CRYSTAL_DIR = Path(__file__).resolve().parent / "do-crystal"
if str(DO_CRYSTAL_DIR) not in sys.path:
    sys.path.append(str(DO_CRYSTAL_DIR))

from PBE_sol.PBE import PBE


def build_pbe(config):
    """Build and configure the do-crystal PBE object."""
    pbe_cfg = config["model_parameters"]["pbe"]
    pbe = PBE(
        pbe_cfg.get("method", "DPBE"),
        coordinate=pbe_cfg.get("coordinate", "L"),
    )
    pbe.setup(
        scheme=pbe_cfg["scheme"],
        spacing=pbe_cfg["spacing"],
        no_class=int(pbe_cfg["no_class"]),
        q=int(pbe_cfg["q"]),
        L_0=float(pbe_cfg["L_0"]),
        domain=[float(pbe_cfg["domain"][0]), float(pbe_cfg["domain"][1])],
    )
    return pbe


def add_chemistry_states(model, config, pbe):
    """Add dissolved-solute and PSD states to the COBR model."""
    n_elements = config["reactor_properties"]["reactor"]["number_tanks_in_series"]
    n_classes = int(pbe.state_shape()[0])

    model.set_variable("_x", "PBE_state", shape=(n_classes * n_elements, 1))
    model.set_variable("_x", "concentration_solute_dissolved", shape=(n_elements, 1))


def add_chemistry_parameters(model, config, pbe):
    """Add kinetic and inflow parameters for the PBE crystallization model."""
    model.set_variable("_p", "k_primary_birth")
    model.set_variable("_p", "exp_primary_birth")
    model.set_variable("_p", "k_secondary_birth")
    model.set_variable("_p", "exp_secondary_birth")
    model.set_variable("_p", "k_growth")
    model.set_variable("_p", "exp_growth")
    model.set_variable("_p", "k_dissolution")
    model.set_variable("_p", "exp_dissolution")
    model.set_variable("_p", "solubility_A")
    model.set_variable("_p", "solubility_B")
    model.set_variable("_p", "rho_crystal")
    model.set_variable("_p", "mw_solute")
    model.set_variable("_p", "k_v")
    model.set_variable("_p", "smoothing_parameter")

    n_classes = int(pbe.state_shape()[0])
    model.set_variable("_p", "inflow_PBE_state", shape=(n_classes, 1))
    model.set_variable("_p", "inflow_concentration_solute_dissolved")

    n_side_inlets = len(config["reactor_properties"].get("side_inlets", []))
    if n_side_inlets > 0:
        model.set_variable("_p", "side_inlet_PBE_state", shape=(n_classes, n_side_inlets))
        model.set_variable("_p", "side_inlet_concentration_solute_dissolved", shape=(n_side_inlets, 1))


def add_auxiliary_expressions(model, config, pbe):
    """Add solubility, supersaturation, kinetic, and PSD-derived expressions."""
    n_elements = config["reactor_properties"]["reactor"]["number_tanks_in_series"]
    pbe_cfg = config["model_parameters"]["pbe"]
    fines_cutoff = float(pbe_cfg["fines_cutoff"])

    l_grid = np.asarray(pbe.L_i, dtype=float).reshape(-1, 1)
    d_l = np.asarray(pbe.del_L_i, dtype=float).reshape(-1, 1)
    grid_dm = ca.DM(l_grid)
    d_l_dm = ca.DM(d_l)
    fine_mask = ca.DM((l_grid <= fines_cutoff).astype(float))

    def get_state_slice(i):
        start = i * int(pbe.state_shape()[0])
        stop = (i + 1) * int(pbe.state_shape()[0])
        return model.x["PBE_state"][start:stop]

    def smooth_max(x, y, eps):
        return (x + y + ca.sqrt((x - y) ** 2 + 2 * eps**2)) / 2

    solubility = ca.SX.zeros(n_elements, 1)
    supersaturation = ca.SX.zeros(n_elements, 1)
    undersaturation = ca.SX.zeros(n_elements, 1)
    growth_rate = ca.SX.zeros(n_elements, 1)
    birth_rate_primary = ca.SX.zeros(n_elements, 1)
    birth_rate_secondary = ca.SX.zeros(n_elements, 1)
    birth_rate_total = ca.SX.zeros(n_elements, 1)
    total_solid_concentration = ca.SX.zeros(n_elements, 1)
    fines_solid_concentration = ca.SX.zeros(n_elements, 1)
    coarse_solid_concentration = ca.SX.zeros(n_elements, 1)
    crystallization_rate = ca.SX.zeros(n_elements, 1)
    mean_crystal_size = ca.SX.zeros(n_elements, 1)
    third_moment = ca.SX.zeros(n_elements, 1)

    eps = model.p["smoothing_parameter"]

    for i in range(n_elements):
        state_i = ca.reshape(get_state_slice(i), -1, 1)
        mu0 = ca.sum1(state_i * d_l_dm)
        mu1 = ca.sum1(state_i * d_l_dm * grid_dm)
        mu2 = ca.sum1(state_i * d_l_dm * (grid_dm**2))
        mu3 = ca.sum1(state_i * d_l_dm * (grid_dm**3))
        mu3_fines = ca.sum1(state_i * d_l_dm * (grid_dm**3) * fine_mask)

        temperature_c = model.x["T_reactor"][i] - 273.15
        solubility[i] = model.p["solubility_A"] * ca.exp(
            model.p["solubility_B"] * temperature_c
        )
        raw_ss = (
            model.x["concentration_solute_dissolved"][i] - solubility[i]
        ) / (solubility[i] + 1.0e-12)
        supersaturation[i] = smooth_max(0.0, raw_ss, eps)
        undersaturation[i] = smooth_max(0.0, -raw_ss, eps)

        growth_rate[i] = (
            model.p["k_growth"] * supersaturation[i] ** model.p["exp_growth"]
            - model.p["k_dissolution"] * undersaturation[i] ** model.p["exp_dissolution"]
        )
        birth_rate_primary[i] = (
            model.p["k_primary_birth"]
            * supersaturation[i] ** model.p["exp_primary_birth"]
        )
        birth_rate_secondary[i] = (
            model.p["k_secondary_birth"]
            * supersaturation[i] ** model.p["exp_secondary_birth"]
            * mu2
        )
        birth_rate_total[i] = birth_rate_primary[i] + birth_rate_secondary[i]

        third_moment[i] = mu3
        total_solid_concentration[i] = (
            model.p["k_v"] * model.p["rho_crystal"] * mu3 / model.p["mw_solute"]
        )
        fines_solid_concentration[i] = (
            model.p["k_v"] * model.p["rho_crystal"] * mu3_fines / model.p["mw_solute"]
        )
        coarse_solid_concentration[i] = (
            total_solid_concentration[i] - fines_solid_concentration[i]
        )
        crystallization_rate[i] = (
            3.0
            * growth_rate[i]
            * mu2
            * model.p["rho_crystal"]
            / model.p["mw_solute"]
            * model.p["k_v"]
        )
        mean_crystal_size[i] = mu1 / (mu0 + 1.0e-12)

    model.set_expression("concentration_solute_solubility", solubility)
    model.set_expression("supersaturation", supersaturation)
    model.set_expression("undersaturation", undersaturation)
    model.set_expression("growth_rate", growth_rate)
    model.set_expression("birth_rate_primary", birth_rate_primary)
    model.set_expression("birth_rate_secondary", birth_rate_secondary)
    model.set_expression("birth_rate_total", birth_rate_total)
    model.set_expression("concentration_solute_solid_total", total_solid_concentration)
    model.set_expression("concentration_solute_solid_fines", fines_solid_concentration)
    model.set_expression("concentration_solute_solid_coarse", coarse_solid_concentration)
    model.set_expression("crystallization_rate", crystallization_rate)
    model.set_expression("mean_crystal_size", mean_crystal_size)
    model.set_expression("third_moment", third_moment)


def set_chemistry_rhs(model, config, pbe):
    """Set the RHS equations for the PBE crystallization states."""
    reactor = config["reactor_properties"]["reactor"]
    reactor_props = config["reactor_properties"]
    n_elements = reactor["number_tanks_in_series"]
    reactor_length = reactor["length"]
    element_volume = reactor["cross_sectional_area"] * reactor_length / n_elements
    conv_coeff = 1.0 / element_volume
    n_classes = int(pbe.state_shape()[0])

    def get_state_slice(i):
        start = i * n_classes
        stop = (i + 1) * n_classes
        return model.x["PBE_state"][start:stop]

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

    def convective_transfer_scalar(state_name, i, inflow_param_name, side_inlet_param_name=None):
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
                    side_value = (
                        model.p[side_inlet_param_name][j]
                        if side_inlet_param_name
                        else model.p[inflow_param_name]
                    )
                    rate += conv_coeff * model.aux["flow_side_inlets"][j] * side_value
        return rate

    d_pbe = ca.SX.zeros(n_classes * n_elements, 1)
    d_c_dissolved = ca.SX.zeros(n_elements, 1)

    for i in range(n_elements):
        state_i = get_state_slice(i)
        state_diff = -conv_coeff * get_outflow_from_element(i) * state_i
        if i == 0:
            state_diff += conv_coeff * get_inlet_flow_at_element(i) * model.p["inflow_PBE_state"]
        else:
            state_diff += conv_coeff * get_inlet_flow_at_element(i) * get_state_slice(i - 1)

        if reactor_props.get("side_inlets"):
            for j, inlet in enumerate(reactor_props["side_inlets"]):
                inlet_element = int(inlet["position"] / reactor_length * n_elements)
                if inlet_element == i:
                    state_diff += (
                        conv_coeff
                        * model.aux["flow_side_inlets"][j]
                        * model.p["side_inlet_PBE_state"][:, j]
                    )

        d_pbe[i * n_classes : (i + 1) * n_classes] = pbe.rhs(
            state_i,
            0,
            model.aux["growth_rate"][i],
            model.aux["birth_rate_total"][i],
            None,
            0.0,
            state_diff,
            1.0,
        )

        d_c_dissolved[i] = (
            -model.aux["crystallization_rate"][i]
            + convective_transfer_scalar(
                "concentration_solute_dissolved",
                i,
                "inflow_concentration_solute_dissolved",
                "side_inlet_concentration_solute_dissolved",
            )
        )

    model.set_rhs("PBE_state", d_pbe)
    model.set_rhs("concentration_solute_dissolved", d_c_dissolved)


def add_pbe_crystallization(model, config, pbe):
    """Attach the PBE crystallization model to the COBR model."""
    add_chemistry_states(model, config, pbe)
    add_chemistry_parameters(model, config, pbe)
    add_auxiliary_expressions(model, config, pbe)
    set_chemistry_rhs(model, config, pbe)
    return model
