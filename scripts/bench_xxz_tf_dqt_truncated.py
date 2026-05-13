# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""XXZ + transverse-field DQT: time-step sweep (full-bond) vs ED.

Same ensemble and observables as :mod:`bench_xxz_tf_dqt_exact`, but repeats the
YAQS evolution for several time steps ``dt`` (fixed full bond cap) to show
how integrator error decreases with smaller ``dt``.

**Reference curve:** The dense curve labeled **ED (k-basis mean)** is the exact
average of ``\langle n|U^\dagger\sigma U\sigma|n\rangle`` over the **same**
first ``k`` computational kets as the YAQS ensemble (``k =`` number of initial
states). When ``k = 2^L``, this equals ``\mathrm{Tr}[\cdot]/2^L``. If
``k < 2^L``, a second curve **ED (Tr/2^L)** shows the full infinite-temperature
trace for comparison only — differences there are **not** TDVP or bond error.

Environment variables::

    YAQS_DTS       comma-separated dt list (default ``0.1,0.01,0.001``)
    YAQS_T_FINAL   default ``10.0`` for this script
    YAQS_NUM_STATES  optional; unset = all ``2^L`` basis states; if ``k`` = first ``k`` kets (see exact script)
    YAQS_DQT_DIAG  optional; if set, print and log (NDJSON) full vs shared-time-grid error metrics
    YAQS_DQT_COMMON_TIME_TOL  optional float; tolerance matching ``times`` to the coarse grid (default ``1e-7`` times ``max(1,T)``)
    (other knobs match the exact script)

Run::

    uv run python scripts/bench_xxz_tf_dqt_truncated.py

**Note:** ED uses :math:`\exp(-iHt)` directly at each sampled time while YAQS is
discretized in ``dt`` (first-order TDVP path), so the ``dt`` sweep isolates
time-integration error once the reference matches the ensemble definition.
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

from scripts.xxz_tf_dqt_common import (
    build_basis_ensemble,
    build_xxz_tf_mpo,
    exact_dqt_pauli_autocorr_xyz,
    exact_dqt_pauli_autocorr_xyz_first_k_basis_mean,
    indices_near_uniform_time_grid,
    max_center_bond_dim,
    metrics_vs_ed,
    metrics_vs_ed_on_indices,
    middle_site,
    parse_dqt_env,
    run_yaqs_dqt_basis_sum,
)


def _debug_ndjson(payload: dict) -> None:
    # #region agent log
    path = "/Users/gauthameshwar/Documents/VSCode/yaqs/.cursor/debug-f332a9.log"
    line = json.dumps({**payload, "sessionId": "f332a9", "timestamp": int(time.time() * 1000)})
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    # #endregion


def _parse_dt_list(raw: str) -> list[float]:
    vals = [float(p.strip()) for p in raw.split(",") if p.strip()]
    if not vals:
        msg = "YAQS_DTS must contain at least one positive dt value."
        raise ValueError(msg)
    if any(v <= 0.0 for v in vals):
        msg = "All YAQS_DTS values must be positive."
        raise ValueError(msg)
    return vals


def main() -> None:
    cfg = parse_dqt_env(
        default_fig_out="scripts/figures/xxz_tf_dqt_dtsweep.png",
        default_chis="unused",
    )
    t_final = float(os.environ.get("YAQS_T_FINAL", "10.0"))
    dt_values = _parse_dt_list(os.environ.get("YAQS_DTS", "0.1,0.01,0.001"))
    chi = max_center_bond_dim(cfg.n)
    states_template = build_basis_ensemble(cfg.n, cfg.num_states)
    n_states = len(states_template)
    n_dim = 2**cfg.n

    print(
        f"XXZ+tf DQT dt-sweep benchmark: L={cfg.n}, J={cfg.j_xy}, Δ={cfg.delta}, h_x={cfg.h_x}, "
        f"T={t_final}, dt runs={dt_values}, max_bond={chi}, threshold={cfg.threshold}, "
        f"ensemble={n_states}/{n_dim} basis states"
    )
    if n_states < n_dim:
        print(
            "Note: metrics and error panels use ED (k-basis mean) matching this ensemble; "
            "ED (Tr/2^L) is shown as an extra curve for reference.",
            flush=True,
        )

    h_mpo = build_xxz_tf_mpo(cfg.n, cfg.j_xy, cfg.delta, cfg.h_x)
    mid = middle_site(cfg.n)

    results: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for dt in dt_values:
        states = [copy.deepcopy(s) for s in states_template]
        times, yaqs = run_yaqs_dqt_basis_sum(
            states=states,
            hamiltonian=h_mpo,
            elapsed_time=t_final,
            dt=dt,
            max_bond_dim=chi,
            threshold=cfg.threshold,
            mid_site=mid,
            parallel=cfg.parallel,
            show_progress=cfg.show_progress,
            track_max_bond=False,
        )
        ed_fair = exact_dqt_pauli_autocorr_xyz_first_k_basis_mean(
            length=cfg.n,
            j_xy=cfg.j_xy,
            delta=cfg.delta,
            h_x=cfg.h_x,
            times=times,
            k=n_states,
            mid_site=mid,
        )
        ed_full: np.ndarray | None
        if n_states < n_dim:
            ed_full = exact_dqt_pauli_autocorr_xyz(
                length=cfg.n,
                j_xy=cfg.j_xy,
                delta=cfg.delta,
                h_x=cfg.h_x,
                times=times,
            )
        else:
            ed_full = None
        results.append((dt, times, yaqs, ed_fair, ed_full))

    coarse_step = max(dt_values)
    tol = float(os.environ.get("YAQS_DQT_COMMON_TIME_TOL", "1e-7")) * max(1.0, t_final)

    diag = bool(os.environ.get("YAQS_DQT_DIAG", ""))
    if diag:
        for dt, times, yaqs, ed_fair, _ed_full in results:
            n_steps = int(len(times) - 1)
            idx_c = indices_near_uniform_time_grid(times, step=coarse_step, t_final=t_final, tol=tol)
            mx_n, rms_n = metrics_vs_ed(yaqs, ed_fair)
            mx_c, rms_c = metrics_vs_ed_on_indices(yaqs, ed_fair, idx_c)
            _debug_ndjson(
                {
                    "runId": "dt-sweep",
                    "hypothesisId": "H1_common_times",
                    "location": "bench_xxz_tf_dqt_truncated.py:common_time_subset",
                    "message": "full native grid vs subset on coarsest-dt time samples",
                    "data": {
                        "dt": dt,
                        "n_steps": n_steps,
                        "parallel": cfg.parallel,
                        "coarse_step": coarse_step,
                        "n_times_native": int(len(times)),
                        "n_common_indices": int(idx_c.size),
                        "max_abs_native": mx_n,
                        "rms_native": rms_n,
                        "max_abs_common_subset": mx_c,
                        "rms_common_subset": rms_c,
                    },
                }
            )

        print(
            "\nDiagnostics (YAQS_DQT_DIAG=1): full native time grid vs same physical times only "
            f"(t = 0, {coarse_step:g}, … ≤ {t_final:g})."
        )
        print("dt        n_steps  n_common  max|full|  max|shared|  RMS|full|  RMS|shared|")
        print("--------  -------  --------  ----------  ------------  ----------  ------------")
        for dt, times, yaqs, ed_fair, _ed_full in results:
            n_steps = len(times) - 1
            idx_c = indices_near_uniform_time_grid(times, step=coarse_step, t_final=t_final, tol=tol)
            mx_n, rms_n = metrics_vs_ed(yaqs, ed_fair)
            mx_c, rms_c = metrics_vs_ed_on_indices(yaqs, ed_fair, idx_c)
            leg = f"dt={dt:g}"
            print(
                f"{leg:<10}  {n_steps:<7}  {idx_c.size:<8}  {mx_n:10.4e}  {mx_c:12.4e}  {rms_n:10.4e}  {rms_c:12.4e}"
            )

    print("\ndt        Pauli     max|YAQS-ED_fair|  RMS|YAQS-ED_fair|")
    print("--------  --------  ------------------  ------------------")
    labels_p = (r"$\sigma^x$", r"$\sigma^y$", r"$\sigma^z$")
    labels_txt = ("sigma_x", "sigma_y", "sigma_z")
    for dt, _t, yaqs, ed_fair, _ed_full in results:
        legend = f"dt={dt:g}"
        for i, pname in enumerate(labels_txt):
            mx, rms = metrics_vs_ed(yaqs[i : i + 1], ed_fair[i : i + 1])
            print(f"{legend:<9}  {pname:<8}  {mx:12.4e}  {rms:13.4e}")
        mx_all, rms_all = metrics_vs_ed(yaqs, ed_fair)
        print(f"{legend:<9}  {'ALL':<8}  {mx_all:12.4e}  {rms_all:13.4e}\n")

    out_dir = os.path.dirname(cfg.fig_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True, constrained_layout=True)
    err_floor = 1e-16

    for col in range(3):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.set_yscale("symlog", linthresh=1e-3)
        _, first_times, _, first_fair, first_full = results[0]
        ax_top.plot(first_times, first_fair[col], "k-", lw=2.4, label=rf"ED ($k={n_states}$ mean)")
        if first_full is not None:
            ax_top.plot(first_times, first_full[col], ":", color="0.35", lw=1.8, label=r"ED ($\mathrm{Tr}/2^L$)")
        for run_idx, (dt, ts, yaqs, ed_fair, _ed_full) in enumerate(results):
            c = colors[run_idx % len(colors)]
            leg = rf"YAQS $dt={dt:g}$"
            ax_top.plot(ts, yaqs[col], color=c, ls="--", lw=1.7, label=leg)
        ax_top.set_title(labels_p[col])
        ax_top.grid(True, which="both", alpha=0.35)
        ax_top.legend(fontsize=6, loc="upper right")
        if col == 0:
            ax_top.set_ylabel(r"$\mathrm{Re}\,\langle\sigma^a(t)\sigma^a\rangle$")

        for run_idx, (dt, ts, yaqs, ed_fair, _ed_full) in enumerate(results):
            c = colors[run_idx % len(colors)]
            leg = rf"$dt={dt:g}$"
            err = np.maximum(np.abs(yaqs[col] - ed_fair[col]), err_floor)
            ax_bot.semilogy(ts, err, color=c, lw=1.5, label=leg)
        ax_bot.set_xlabel(r"$t$")
        ax_bot.grid(True, which="both", alpha=0.35)
        if col == 0:
            ax_bot.set_ylabel(r"$|\mathrm{YAQS} - \mathrm{ED}_{\mathrm{fair}}|$")

    fig.suptitle(
        rf"XXZ+tf DQT (dt sweep)  $L={cfg.n}$, $J={cfg.j_xy}$, $\Delta={cfg.delta}$, $h_x={cfg.h_x}$, "
        rf"$\chi={chi}$, $T={t_final}$, "
        rf"{n_states}/{n_dim} basis states",
        fontsize=10,
    )
    plt.savefig(cfg.fig_out, dpi=150)
    plt.close()
    print(f"Saved figure: {cfg.fig_out}")


if __name__ == "__main__":
    main()
