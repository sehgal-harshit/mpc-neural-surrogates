"""
Plotting script for thermal COBR sampled dataset.

Usage:
    python plotting.py                          # auto-detects latest HDF5 in data_sets/
    python plotting.py path/to/file.h5          # explicit file
    python plotting.py path/to/file.h5 --sim 5  # highlight simulation index 5

Produces 4 figures saved to a figures/ subfolder next to the HDF5 file.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_h5() -> Path:
    base = Path(__file__).parent / "data_sets"
    candidates = sorted(base.rglob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No .h5 files found under {base}")
    return candidates[-1]


def load_sim(f: h5py.File, idx: int) -> dict:
    g = f[f"simulations/{idx}"]
    return {
        "time":   g["time"][:],
        "x":      {k: g["x"][k][:] for k in g["x"]},
        "u":      {k: g["u"][k][:] for k in g["u"]},
        "y":      {k: g["y"][k][:] for k in g["y"]},
        "aux":    {k: g["aux"][k][:] for k in g["aux"]},
    }


def kelvin_to_celsius(arr):
    return arr - 273.15


# ── figure 1: dataset overview — spaghetti at 4 reactor positions ─────────────

def fig_dataset_overview(f: h5py.File, highlight_idx: int, out_dir: Path):
    n_sims = len(f["simulations"])
    positions = ["T_reactor_2m", "T_reactor_6m", "T_reactor_12m", "T_reactor_18m"]
    labels    = ["2 m", "6 m", "12 m", "18 m"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, key, label in zip(axes, positions, labels):
        for i in range(n_sims):
            t  = f[f"simulations/{i}/time"][:]
            T  = kelvin_to_celsius(f[f"simulations/{i}/y/{key}"][:])
            ax.plot(t / 60, T, color="steelblue", alpha=0.15, linewidth=0.6)

        # highlight one simulation
        t_hi = f[f"simulations/{highlight_idx}/time"][:]
        T_hi = kelvin_to_celsius(f[f"simulations/{highlight_idx}/y/{key}"][:])
        ax.plot(t_hi / 60, T_hi, color="crimson", linewidth=1.6, label=f"sim {highlight_idx}")

        ax.set_title(f"Reactor temperature @ {label}")
        ax.set_ylabel("Temperature (°C)")
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Dataset overview — {n_sims} simulations", fontsize=13)
    fig.tight_layout()
    path = out_dir / "fig1_dataset_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 2: spatial temperature heatmap for one simulation ─────────────────

def fig_reactor_heatmap(sim: dict, sim_idx: int, reactor_length: float, out_dir: Path):
    T = kelvin_to_celsius(sim["x"]["T_reactor"])   # (150, 120)
    t = sim["time"] / 60                            # minutes
    z = np.linspace(0, reactor_length, T.shape[1]) # metres

    fig, ax = plt.subplots(figsize=(11, 5))
    pcm = ax.pcolormesh(z, t, T, cmap="RdYlBu_r", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Temperature (°C)")
    ax.set_xlabel("Reactor position (m)")
    ax.set_ylabel("Time (min)")
    ax.set_title(f"Reactor temperature profile — simulation {sim_idx}")
    fig.tight_layout()
    path = out_dir / "fig2_reactor_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 3: thermostat temperatures vs setpoints ───────────────────────────

_ZONE_POSITIONS = [2.25, 4.5, 6.75, 9.0, 11.25, 13.5, 15.75, 18.0]  # jacket outlet positions (m)

def fig_thermostats(sim: dict, sim_idx: int, out_dir: Path):
    t = sim["time"] / 60
    n_zones = sim["x"]["T_thermostat"].shape[1]
    colors = [cm.tab10(i / n_zones) for i in range(n_zones)]

    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n_zones]):
        T_therm = kelvin_to_celsius(sim["x"]["T_thermostat"][:, i])
        T_set   = kelvin_to_celsius(sim["u"][f"T_setpoint_thermostat_{i}"][:])
        pos_label = f"{_ZONE_POSITIONS[i]} m"

        ax.plot(t, T_therm, color=colors[i], linewidth=1.5, label=f"T_thermostat")
        ax.plot(t, T_set,   color=colors[i], linewidth=1.0, linestyle="--", label=f"T_setpoint")
        ax.set_title(f"Zone {i+1} ({pos_label})", fontsize=9)
        ax.set_ylabel("Temp (°C)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_zones:]:
        ax.set_visible(False)

    for ax in axes[n_zones - 2:n_zones]:
        ax.set_xlabel("Time (min)")

    fig.suptitle(f"Thermostat temperatures vs. setpoints — simulation {sim_idx}", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig3_thermostats.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── figure 4: control inputs ──────────────────────────────────────────────────

def fig_inputs(sim: dict, sim_idx: int, out_dir: Path):
    t = sim["time"] / 60

    fig, (ax_flow, ax_set) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # flow inlet in mL/min for readability
    flow = sim["u"]["flow_inlet"][:] * 1e6 * 60  # m³/s → mL/min
    ax_flow.plot(t, flow, color="darkgreen", linewidth=1.5)
    ax_flow.set_ylabel("Flow inlet (mL/min)")
    ax_flow.grid(True, alpha=0.3)

    n_zones = sim["x"]["T_thermostat"].shape[1]
    colors = [cm.tab10(i / n_zones) for i in range(n_zones)]
    for i, color in enumerate(colors):
        T_set = kelvin_to_celsius(sim["u"][f"T_setpoint_thermostat_{i}"][:])
        ax_set.plot(t, T_set, color=color, linewidth=1.2, label=f"Zone {i+1} ({_ZONE_POSITIONS[i]} m)")
    ax_set.set_ylabel("Setpoint temperature (°C)")
    ax_set.set_xlabel("Time (min)")
    ax_set.legend(loc="upper right", fontsize=8)
    ax_set.grid(True, alpha=0.3)

    fig.suptitle(f"Control inputs — simulation {sim_idx}", fontsize=12)
    fig.tight_layout()
    path = out_dir / "fig4_inputs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", nargs="?", default=None, help="Path to HDF5 dataset file")
    parser.add_argument("--sim", type=int, default=0, help="Simulation index to highlight/detail (default: 0)")
    args = parser.parse_args()

    h5_path = Path(args.h5_file) if args.h5_file else find_latest_h5()
    print(f"Loading: {h5_path}")

    out_dir = h5_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    reactor_length = 18.0  # metres

    with h5py.File(h5_path, "r") as f:
        n_sims = len(f["simulations"])
        highlight = min(args.sim, n_sims - 1)
        print(f"  {n_sims} simulations, highlighting index {highlight}")

        fig_dataset_overview(f, highlight, out_dir)
        sim = load_sim(f, highlight)

    fig_reactor_heatmap(sim, highlight, reactor_length, out_dir)
    fig_thermostats(sim, highlight, out_dir)
    fig_inputs(sim, highlight, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
