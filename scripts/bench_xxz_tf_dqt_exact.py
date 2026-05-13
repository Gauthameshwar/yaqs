# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""XXZ + transverse-field DQT: exact (full-bond) YAQS full-summation vs ED.

Builds computational-basis MPS (all ``2^L`` by default, or the first ``k`` if
``YAQS_NUM_STATES=k``), evolves in parallel with no effective truncation (bond cap
``2 ** ceil(L/2)``), and compares middle-site Pauli autocorrelators to dense ED on the same time grid.
The primary reference **ED** matches the ensemble: **k-basis mean** when using
``k`` states, which equals ``Tr[\cdot]/2^L`` when ``k = 2^L``. If ``k < 2^L``,
a dotted **ED (Tr/2^L)** curve is drawn for comparison only.

Environment variables::

    YAQS_N, YAQS_J, YAQS_DELTA, YAQS_HX
    YAQS_T_FINAL, YAQS_DT
    YAQS_THRESHOLD (default 1e-14 for this script)
    YAQS_NUM_STATES  optional; if unset, use all ``2^L`` basis states. If set to ``k``,
        use the first ``k`` computational basis kets (lexicographic order). Then the
        YAQS curve is a **partial** ensemble average and will not match the full
        infinite-temperature ED trace unless ``k = 2^L``.
    YAQS_PARALLEL, YAQS_SHOW_PROGRESS
    YAQS_FIG_OUT (default scripts/figures/xxz_tf_dqt_exact.png)

Run::

    uv run python scripts/bench_xxz_tf_dqt_exact.py

**Note:** ED uses the exact continuous-time unitary :math:`\exp(-iHt)` on the time
grid; YAQS uses first-order TDVP with step ``YAQS_DT``. Use a smaller ``YAQS_DT``
if the top-row curves should match ED more closely than the default integration error.

**Symmetries (``h_z = 0``):** integrability is absent for generic :math:`J \neq \Delta`;
:math:`S^z_{\mathrm{tot}}` is broken by ``h_x``; global spin-flip :math:`P_x = \prod_i X_i`
commutes with the XXZ + :math:`h_x\sum X` Hamiltonian (``Z_2``), but the full basis
ensemble still reproduces :math:`\mathrm{Tr}[\cdot]/2^L` by construction.
"""

from __future__ import annotations

import os

import numpy as np

from scripts.xxz_tf_dqt_common import (
    build_basis_ensemble,
    build_xxz_tf_mpo,
    exact_dqt_pauli_autocorr_xyz,
    exact_dqt_pauli_autocorr_xyz_first_k_basis_mean,
    max_center_bond_dim,
    metrics_vs_ed,
    middle_site,
    parse_dqt_env,
    run_yaqs_dqt_basis_sum,
)


def main() -> None:
    cfg = parse_dqt_env(default_fig_out="scripts/figures/xxz_tf_dqt_exact.png")
    threshold = float(os.environ.get("YAQS_THRESHOLD", "1e-14"))
    chi = max_center_bond_dim(cfg.n)
    states = build_basis_ensemble(cfg.n, cfg.num_states)
    n_states = len(states)
    n_dim = 2**cfg.n

    print(
        f"XXZ+tf DQT exact benchmark: L={cfg.n}, J={cfg.j_xy}, Δ={cfg.delta}, h_x={cfg.h_x}, "
        f"T={cfg.t_final}, dt={cfg.dt}, max_bond={chi}, threshold={threshold}, "
        f"ensemble={n_states}/{n_dim} basis states"
    )
    if n_states < n_dim:
        print(
            "Note: primary ED curve is the k-basis mean matching this ensemble; "
            "Tr[·]/2^L is shown as an extra reference on the plot.",
            flush=True,
        )

    h_mpo = build_xxz_tf_mpo(cfg.n, cfg.j_xy, cfg.delta, cfg.h_x)
    mid = middle_site(cfg.n)

    times, yaqs = run_yaqs_dqt_basis_sum(
        states=states,
        hamiltonian=h_mpo,
        elapsed_time=cfg.t_final,
        dt=cfg.dt,
        max_bond_dim=chi,
        threshold=threshold,
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
            length=cfg.n, j_xy=cfg.j_xy, delta=cfg.delta, h_x=cfg.h_x, times=times
        )
    else:
        ed_full = None

    labels_plot = (r"$\sigma^x$", r"$\sigma^y$", r"$\sigma^z$")
    labels_txt = ("sigma_x", "sigma_y", "sigma_z")
    print("\nPauli    max|YAQS-ED_fair|  RMS|YAQS-ED_fair|")
    print("--------  ------------------  ------------------")
    for i, name in enumerate(labels_txt):
        mx, rms = metrics_vs_ed(yaqs[i : i + 1], ed_fair[i : i + 1])
        print(f"{name:<8}  {mx:12.4e}  {rms:13.4e}")
    mx_all, rms_all = metrics_vs_ed(yaqs, ed_fair)
    print(f"{'ALL':<8}  {mx_all:12.4e}  {rms_all:13.4e}")

    out_dir = os.path.dirname(cfg.fig_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True, constrained_layout=True)
    err_floor = 1e-16
    for col in range(3):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.set_yscale("symlog", linthresh=1e-3)
        ax_top.plot(times, ed_fair[col], "k-", lw=2.0, label=rf"ED ($k={n_states}$ mean)")
        if ed_full is not None:
            ax_top.plot(times, ed_full[col], ":", color="0.35", lw=1.6, label=r"ED ($\mathrm{Tr}/2^L$)")
        ax_top.plot(times, yaqs[col], "C0--", lw=1.8, label="YAQS (full χ)")
        ax_top.set_title(labels_plot[col])
        ax_top.grid(True, which="both", alpha=0.35)
        ax_top.legend(fontsize=8)
        if col == 0:
            ax_top.set_ylabel(r"$\mathrm{Re}\,\langle\sigma^a(t)\sigma^a\rangle_{\beta=0}$")

        err = np.maximum(np.abs(yaqs[col] - ed_fair[col]), err_floor)
        ax_bot.semilogy(times, err, "C1-", lw=1.5)
        ax_bot.set_xlabel(r"$t$")
        ax_bot.grid(True, which="both", alpha=0.35)
        if col == 0:
            ax_bot.set_ylabel(r"$|\mathrm{YAQS} - \mathrm{ED}_{\mathrm{fair}}|$")

    fig.suptitle(
        rf"XXZ+tf DQT (exact MPS)  $L={cfg.n}$, $J={cfg.j_xy}$, $\Delta={cfg.delta}$, $h_x={cfg.h_x}$, "
        rf"$\chi={chi}$, {n_states}/{n_dim} basis states",
        fontsize=10,
    )
    plt.savefig(cfg.fig_out, dpi=150)
    plt.close()
    print(f"\nSaved figure: {cfg.fig_out}")


if __name__ == "__main__":
    main()
