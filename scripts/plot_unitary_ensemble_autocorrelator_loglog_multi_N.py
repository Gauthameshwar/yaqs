# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Multi-system-size log–log autocorrelator plot using default YAQS unitary ensemble.

Uses :func:`mqt.yaqs.simulator.run` with ``initial_state`` as ``list[MPS]`` (no noise),
the same onsite autocorrelator as the benchmark scripts::

    C_i(t) = \\langle Z_i(t) Z_i(0) \\rangle

(ensemble-averaged and stored in ``AnalogSimParams.autocorrelator_results``).

Hamiltonian (Pauli convention, matching :func:`MPO.hamiltonian`)::

    H = -J_{xy} \\sum_i (X_i X_{i+1} + Y_i Y_{i+1})
        - \\Delta \\sum_i Z_i Z_{i+1}
        - h \\sum_i X_i .

Default sweep: ``N \\in \\{10, 14, 18, 22\\}``, ``T = 300``, ``200`` random initial MPS,
parallel ensemble evolution.

Environment overrides::

    YAQS_LENGTHS       comma-separated (default ``10,14,18,22``)
    YAQS_NUM_STATES    default ``200``
    YAQS_T_FINAL       default ``300``
    YAQS_DT            default ``1.0``  (301 samples from t=0 .. T inclusive)
    YAQS_J, YAQS_DELTA, YAQS_FIELD   transverse XXZ parameters (defaults 1.0, 1.0, 0.5)
    YAQS_PAD           random-MPS bond cap (default ``16``)
    YAQS_MAX_BOND_DIM  TDVP truncation cap (default ``128``)
    YAQS_PARALLEL      ``1``/``0`` (default ``1``)
    YAQS_SHOW_PROGRESS ``1``/``0`` (default ``1``; set ``0`` for quiet/smoke runs)

Output figure (default path; override with ``YAQS_LOGLOG_OUT``)::

    scripts/figures/unitary_ensemble_autocorrelator_loglog_multi_N.png

Run::

    uv run python scripts/plot_unitary_ensemble_autocorrelator_loglog_multi_N.py

Full defaults are expensive for large ``N`` and ``T``; use env overrides for dry runs.
"""

from __future__ import annotations

import os
import time

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z


def _parse_lengths() -> list[int]:
    raw = os.environ.get("YAQS_LENGTHS", "10,14,18,22")
    lengths = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return sorted(set(lengths))


def main() -> None:
    lengths = _parse_lengths()
    num_states = int(os.environ.get("YAQS_NUM_STATES", "200"))
    t_final = float(os.environ.get("YAQS_T_FINAL", "300"))
    dt = float(os.environ.get("YAQS_DT", "1.0"))
    j_xy = float(os.environ.get("YAQS_J", "1.0"))
    delta = float(os.environ.get("YAQS_DELTA", "1.0"))
    field = float(os.environ.get("YAQS_FIELD", "0.5"))
    pad = int(os.environ.get("YAQS_PAD", "16"))
    max_bond_dim = int(os.environ.get("YAQS_MAX_BOND_DIM", "128"))
    threshold = float(os.environ.get("YAQS_THRESHOLD", "1e-9"))
    parallel = os.environ.get("YAQS_PARALLEL", "1") != "0"
    show_progress = os.environ.get("YAQS_SHOW_PROGRESS", "1") != "0"

    print(
        f"Unitary ensemble log-log sweep: N in {lengths}, states={num_states}, "
        f"T={t_final}, dt={dt}, parallel={parallel}"
    )
    print(f"Hamiltonian: J_xy={j_xy}, Delta={delta}, h={field}; pad={pad}, max_bond_dim={max_bond_dim}")

    _grid_len = len(np.arange(0.0, t_final + dt, dt, dtype=np.float64))
    _inner_steps = max(_grid_len - 1, 0)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is required for plotting. Install with: uv pip install matplotlib")
        raise

    series: list[tuple[int, np.ndarray, np.ndarray]] = []

    for length in lengths:
        hamiltonian = MPO.hamiltonian(
            length=length,
            two_body=[(-j_xy, "X", "X"), (-j_xy, "Y", "Y"), (-delta, "Z", "Z")],
            one_body=[(-field, "X")],
            bc="open",
        )
        center = length // 2
        correlator_op = Observable(Z(), center)

        initial_states = [MPS(length, state="random", pad=pad) for _ in range(num_states)]

        sim_params = AnalogSimParams(
            observables=[],
            elapsed_time=t_final,
            dt=dt,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            order=1,
            sample_timesteps=True,
            show_progress=show_progress,
            compute_autocorrelator=True,
            autocorrelator_observable=correlator_op,
        )

        t0 = time.perf_counter()
        simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=parallel)
        elapsed = time.perf_counter() - t0

        assert sim_params.autocorrelator_results is not None
        assert sim_params.autocorrelator_times is not None
        times = np.asarray(sim_params.autocorrelator_times, dtype=np.float64)
        corr = np.asarray(sim_params.autocorrelator_results, dtype=np.complex128)
        series.append((length, times, corr))
        print(f"  N={length} done in {elapsed:.1f} s, time points={len(times)}")

    plt.figure(figsize=(8, 6))
    eps_y = float(os.environ.get("YAQS_LOGLOG_EPS", "1e-15"))
    for length, times, corr in series:
        # Positive magnitude for log scale (|Re C| with tiny floor for numerical zeros)
        y = np.maximum(np.abs(np.real(corr)), eps_y)
        mask = times > 0
        plt.loglog(times[mask], y[mask], linewidth=2, label=f"N={length}")

    plt.xlabel("time")
    plt.ylabel(r"$|\mathrm{Re}\,\langle Z_i(t)Z_i(0)\rangle|$ (ensemble mean)")
    plt.title(
        r"Unitary ensemble autocorrelator (transverse XXZ, Pauli) — log–log"
        f"\n$J_{{xy}}$={j_xy}, $\\Delta$={delta}, $h$={field}, "
        f"{num_states} states / curve"
    )
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(title="system size")
    plt.tight_layout()

    out = os.environ.get(
        "YAQS_LOGLOG_OUT",
        "scripts/figures/unitary_ensemble_autocorrelator_loglog_multi_N.png",
    )
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
