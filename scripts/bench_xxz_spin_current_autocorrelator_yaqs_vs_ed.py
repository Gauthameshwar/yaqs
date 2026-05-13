# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""XXZ spin-current autocorrelator: YAQS ensemble vs exact diagonalization.

Computes the infinite-temperature (ensemble-averaged) autocorrelation of the
spin current in the periodic XXZ chain, using the spin-1/2 convention
``S^alpha = sigma^alpha / 2`` as in the paper,

    C_J(t) = Re ⟨ J(t) J(0) ⟩ / L,

    H = J \\sum_r (S^x_r S^x_{r+1} + S^y_r S^y_{r+1} + Δ S^z_r S^z_{r+1})

with periodic boundary conditions.

Local current on bond ``(r, r+1)``::

    j_r = J ( S^x_r S^y_{r+1} - S^y_r S^x_{r+1} ),

and total current ``J = \\sum_r j_r``.

**Exact reference:** full Hilbert-space diagonalization at small ``L_ED`` (default
12; dense ``eigh`` — use ``L_ED <= 14`` on typical workstations).

**YAQS:** full periodic spin current correlator ``(1/L)∑_{r,s}⟨j_r(t)j_s(0)⟩`` (including
cross terms and the wrap bond), using ``L`` unitary-ensemble evolutions per ensemble member
(one per initial bond ``j_s``) and the extended wrap-aware :func:`~mqt.yaqs.analog.autocorrelator.apply_observable_inplace`.

For each anisotropy ``Δ``, the script produces:

- ED mean curve at ``L_ED`` (full ``J``),
- YAQS ensemble mean at ``L_ED`` (overlay for error metrics).

For moderate ``L_ED`` (for example ``≥ 8``) and long times, use a **large** ``YAQS_MAX_BOND_DIM`` and set
``YAQS_PAD`` to at least that value for ``haar-random`` states; otherwise truncation can make the MPS
curve decay toward zero while ED keeps a finite tail.

Environment variables::

    YAQS_L_ED          small chain for ED (default 12)
    YAQS_L_YAQS        reserved for future large-L panel (default 20)
    YAQS_J             in-plane coupling J (default 1.0)
    YAQS_DELTAS        comma-separated Δ list (default 0.5,1.0,1.5)
    YAQS_NUM_STATES    ensemble size (default 40)
    YAQS_T_FINAL, YAQS_DT   time grid
    YAQS_MAX_BOND_DIM, YAQS_THRESHOLD
    YAQS_PAD           bond cap for random MPS init (default 4)
    YAQS_INIT          ``random`` | ``haar-random`` (default haar-random)
    YAQS_PARALLEL      1/0 (default 1)
    YAQS_SHOW_PROGRESS 1/0
    YAQS_FIG_OUT       output PNG path

Run::

    uv run python scripts/bench_xxz_spin_current_autocorrelator_yaqs_vs_ed.py
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import sys
from functools import lru_cache
from typing import TypedDict

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.analog.unitary_ensemble import unitary_ensemble_member_worker
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate


@lru_cache(maxsize=1)
def _pauli_xyz() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Pauli matrices X, Y, Z as complex128."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return x, y, z


def spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    """4×4 Pauli-space matrix for ``J (Sx⊗Sy - Sy⊗Sx)`` on a bond.

    Args:
        j_coupling: Exchange strength ``J`` matching the Hamiltonian prefactor.

    Returns:
        Complex Hermitian matrix of shape ``(4, 4)``.
    """
    x, y, _ = _pauli_xyz()
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def _bit_index(site: int, length: int) -> int:
    """Return the bit position for a site in the dense basis ordering."""
    return length - 1 - site


def embed_two_site_operator_periodic(length: int, site_left: int, site_right: int, op4: np.ndarray) -> np.ndarray:
    """Embed a two-qubit operator onto arbitrary sites in the dense basis.

    This supports the periodic bond ``(L-1, 0)`` using a bitwise embedding.
    """
    if site_left == site_right:
        msg = "Two-site operator requires distinct sites."
        raise ValueError(msg)
    if site_left < 0 or site_left >= length or site_right < 0 or site_right >= length:
        msg = f"Invalid sites {site_left},{site_right} for length {length}."
        raise ValueError(msg)

    if site_right == site_left + 1:
        return embed_two_site_operator(length, site_left, op4)

    dim = 2**length
    out = np.zeros((dim, dim), dtype=np.complex128)
    bit_left = _bit_index(site_left, length)
    bit_right = _bit_index(site_right, length)
    clear_mask = ~(1 << bit_left) & ~(1 << bit_right)
    for col in range(dim):
        in_left = (col >> bit_left) & 1
        in_right = (col >> bit_right) & 1
        in_idx = (in_left << 1) | in_right
        base = col & clear_mask
        for out_idx in range(4):
            amp = op4[out_idx, in_idx]
            if amp == 0:
                continue
            out_left = (out_idx >> 1) & 1
            out_right = out_idx & 1
            row = base | (out_left << bit_left) | (out_right << bit_right)
            out[row, col] += amp
    return out


def embed_two_site_operator(length: int, site_left: int, op4: np.ndarray) -> np.ndarray:
    """Embed a two-qubit operator onto sites ``(site_left, site_left+1)`` on an open chain.

    Args:
        length: Number of qubits ``L``.
        site_left: Left site index of the bond.
        op4: Shape ``(4, 4)`` operator in lexicographic order ``|ab⟩`` with ``a`` left qubit.

    Returns:
        Full Hilbert-space matrix of shape ``(2**L, 2**L)``.
    """
    if site_left < 0 or site_left + 1 >= length:
        msg = f"Invalid bond {site_left},{site_left + 1} for length {length}."
        raise ValueError(msg)
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    mid = op4.astype(np.complex128)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), mid), np.eye(right_dim, dtype=np.complex128))


def build_xxz_periodic_chain_dense(length: int, j_xy: float, delta: float) -> np.ndarray:
    """Dense Hamiltonian: ``J ∑(SxSx+SySy+Δ SzSz)`` (periodic chain).

    Args:
        length: ``L``.
        j_xy: In-plane coupling ``J``.
        delta: Anisotropy ``Δ`` on ``ZZ``.

    Returns:
        Hermitian matrix of shape ``(2**L, 2**L)``.
    """
    if length < 2:
        msg = "length must be at least 2."
        raise ValueError(msg)
    x, y, z = _pauli_xyz()
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    dim = 2**length
    h = np.zeros((dim, dim), dtype=np.complex128)
    scale = 0.25
    for i in range(length - 1):
        h += scale * j_xy * embed_two_site_operator(length, i, xx)
        h += scale * j_xy * embed_two_site_operator(length, i, yy)
        h += scale * delta * embed_two_site_operator(length, i, zz)
    h += scale * j_xy * embed_two_site_operator_periodic(length, length - 1, 0, xx)
    h += scale * j_xy * embed_two_site_operator_periodic(length, length - 1, 0, yy)
    h += scale * delta * embed_two_site_operator_periodic(length, length - 1, 0, zz)
    return h


def build_xxz_open_chain_dense(length: int, j_xy: float, delta: float) -> np.ndarray:
    """Dense XXZ Hamiltonian on an **open** chain (no periodic wrap term)."""
    if length < 2:
        msg = "length must be at least 2."
        raise ValueError(msg)
    x, y, z = _pauli_xyz()
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    dim = 2**length
    h = np.zeros((dim, dim), dtype=np.complex128)
    scale = 0.25
    for i in range(length - 1):
        h += scale * j_xy * embed_two_site_operator(length, i, xx)
        h += scale * j_xy * embed_two_site_operator(length, i, yy)
        h += scale * delta * embed_two_site_operator(length, i, zz)
    return h


def center_bond_left(length: int) -> int:
    """Left index of a central nearest-neighbor bond (``L`` even: ``(L//2-1, L//2)``)."""
    return max(0, length // 2 - 1)


def parse_anisotropies(raw: str) -> list[float]:
    """Parse comma-separated positive anisotropy values."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        msg = "At least one anisotropy is required."
        raise ValueError(msg)
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return out


def exact_autocorr_spin_current(
    initial_state_vectors: list[np.ndarray],
    length: int,
    bond_left: int,
    times: np.ndarray,
    j_xy: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-averaged ``Re ⟨J(t)J⟩/L`` via full diagonalization.

    Uses the same mixed-inner-product scheme as the Z autocorrelator benchmarks:
    ``C(t) = Re vdot(U ψ, j U j ψ)`` with ``U = exp(-i H t)``.
    """
    h_dense = build_xxz_periodic_chain_dense(length, j_xy, delta)
    j_bond = spin_current_bond_matrix(j_xy)
    j_full = np.zeros((2**length, 2**length), dtype=np.complex128)
    for i in range(length - 1):
        j_full += embed_two_site_operator(length, i, j_bond)
    j_full += embed_two_site_operator_periodic(length, length - 1, 0, j_bond)

    evals, evecs = np.linalg.eigh(h_dense)
    evecs_dag = evecs.conj().T
    phases = np.exp(-1j * np.outer(evals, times))

    correlations = np.zeros((len(initial_state_vectors), len(times)), dtype=np.complex128)
    for idx, psi0 in enumerate(initial_state_vectors):
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        nrm = np.linalg.norm(psi)
        if nrm == 0:
            msg = "Initial state vector has zero norm."
            raise ValueError(msg)
        psi /= nrm
        phi0 = j_full @ psi

        psi_e = evecs_dag @ psi
        phi_e = evecs_dag @ phi0

        for t_idx in range(len(times)):
            psi_t = evecs @ (phases[:, t_idx] * psi_e)
            phi_t = evecs @ (phases[:, t_idx] * phi_e)
            correlations[idx, t_idx] = np.vdot(psi_t, j_full @ phi_t)

    mean_corr = np.real(np.mean(correlations, axis=0)) / length
    n_states = len(initial_state_vectors)
    if n_states > 1:
        stderr = np.std(np.real(correlations), axis=0, ddof=1) / np.sqrt(n_states)
    else:
        stderr = np.zeros(len(times), dtype=np.float64)
    return mean_corr, stderr


def periodic_bond_endpoints(length: int) -> list[tuple[int, int]]:
    """Left/right site indices for each spin-current bond (periodic chain).

    For ``L>2`` the last bond wraps ``(L-1, 0)``. For ``L==2`` there is only one physical bond.
    """
    if length < 2:
        msg = "length must be at least 2."
        raise ValueError(msg)
    out: list[tuple[int, int]] = [(i, i + 1) for i in range(length - 1)]
    if length > 2:
        out.append((length - 1, 0))
    return out


def spin_current_observable_for_periodic_bond(
    length: int, bond_left: int, bond_right: int, j_xy: float
) -> Observable:
    """Two-site spin-current observable for one periodic-chain bond."""
    if bond_right == bond_left + 1:
        gate = BaseGate(spin_current_bond_matrix(j_xy))
        return Observable(gate, sites=[bond_left, bond_right])
    if bond_left == length - 1 and bond_right == 0:
        gate = BaseGate(spin_current_bond_matrix(j_xy))
        return Observable(gate, sites=[length - 1, 0])
    msg = f"Invalid periodic bond ({bond_left}, {bond_right}) for length {length}."
    raise ValueError(msg)


def run_yaqs_full_periodic_current_autocorr(
    *,
    initial_states: list[MPS],
    hamiltonian: MPO,
    j_xy: float,
    elapsed_time: float,
    dt: float,
    max_bond_dim: int,
    threshold: float,
    show_progress: bool,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-averaged ``Re⟨J(t)J(0)⟩/L`` with ``J=∑_r j_r`` on the periodic chain (MPS/TDVP).

    Uses ``L`` dual evolutions per ensemble member (one per bond ``j_s``), summing
    ``∑_r ⟨j_r(t)j_s(0)⟩`` for each, then dividing by ``L`` and averaging over the ensemble.
    """
    length = initial_states[0].length
    bonds = periodic_bond_endpoints(length)
    row_obs = tuple(
        spin_current_observable_for_periodic_bond(length, a, b, j_xy) for a, b in bonds
    )

    n_states = len(initial_states)
    max_workers = max(1, min(n_states, simulator.available_cpus() - 1))
    times = np.asarray(
        AnalogSimParams(
            observables=[],
            elapsed_time=elapsed_time,
            dt=dt,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            order=1,
            sample_timesteps=True,
            show_progress=False,
        ).times,
        dtype=np.float64,
    )
    acc = np.zeros(len(times), dtype=np.float64)

    for obs_s in row_obs:
        pairs = [(obs_r, obs_s) for obs_r in row_obs]
        sim_sweep = AnalogSimParams(
            observables=[],
            elapsed_time=elapsed_time,
            dt=dt,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            order=1,
            sample_timesteps=True,
            show_progress=False,
            compute_autocorrelator=False,
            two_time_correlators=pairs,
        )
        args_list = [
            (i, copy.deepcopy(initial_states[i]), sim_sweep, hamiltonian) for i in range(n_states)
        ]
        if parallel and n_states > 1:
            ctx = multiprocessing.get_context("fork" if sys.platform == "linux" else "spawn")
            with ctx.Pool(processes=max_workers) as pool:
                raw_results = pool.map(unitary_ensemble_member_worker, args_list)
        else:
            raw_results = [
                simulator._call_backend(
                    unitary_ensemble_member_worker,
                    arg,
                    n_threads=simulator.available_cpus(),
                )
                for arg in args_list
            ]

        two_time_rows: list[np.ndarray] = [np.asarray(r[2], dtype=np.complex128) for r in raw_results]
        stack = np.mean(np.stack(two_time_rows, axis=0), axis=0)
        acc += np.sum(np.real(stack), axis=0)

    if show_progress:
        print(f"YAQS full-J autocorr: finished {len(bonds)} sweeps over initial bonds × {n_states} states.")

    return times, acc / float(length)


def _make_initial_ensemble(
    *,
    length: int,
    num_states: int,
    init_mode: str,
    pad: int | None,
) -> list[MPS]:
    if init_mode == "random":
        states = [MPS(length, state="random", pad=pad) for _ in range(num_states)]
    elif init_mode == "haar-random":
        chi = 1 if pad is None else pad
        states = [MPS(length, state="haar-random", pad=chi) for _ in range(num_states)]
    else:
        msg = f"Unknown YAQS_INIT={init_mode!r}; use 'random' or 'haar-random'."
        raise ValueError(msg)
    for s in states:
        s.normalize("B")
    return states


def run_yaqs_autocorr(
    *,
    initial_states: list[MPS],
    hamiltonian: MPO,
    bond_left: int,
    j_xy: float,
    elapsed_time: float,
    dt: float,
    max_bond_dim: int,
    threshold: float,
    show_progress: bool,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, mean_autocorr_real)`` from ``simulator.run``."""
    j_mat = spin_current_bond_matrix(j_xy)
    gate = BaseGate(j_mat)
    obs = Observable(gate, sites=[bond_left, bond_left + 1])

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=elapsed_time,
        dt=dt,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        order=1,
        sample_timesteps=True,
        show_progress=show_progress,
        compute_autocorrelator=True,
        autocorrelator_observable=obs,
    )
    states_copy = [copy.deepcopy(s) for s in initial_states]
    simulator.run(states_copy, hamiltonian, sim_params, noise_model=None, parallel=parallel)
    assert sim_params.autocorrelator_times is not None
    assert sim_params.autocorrelator_results is not None
    times = np.asarray(sim_params.autocorrelator_times, dtype=np.float64)
    corr = np.asarray(sim_params.autocorrelator_results, dtype=np.complex128)
    length = initial_states[0].length
    return times, np.real(corr)


def _metrics_vs_ed(
    yaqs_mean: np.ndarray, ed_mean: np.ndarray, ed_times: np.ndarray, yaqs_times: np.ndarray
) -> tuple[float, float]:
    ed_interp = np.interp(yaqs_times, ed_times, ed_mean)
    diff = yaqs_mean - ed_interp
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


class _SeriesEntry(TypedDict):
    """One anisotropy slice of benchmark outputs for plotting."""

    delta: float
    times_ed: np.ndarray
    ed_mean_full_j: np.ndarray
    yaqs_times_s: np.ndarray
    yaqs_mean_s: np.ndarray
    yaqs_times_b: np.ndarray
    yaqs_mean_b: np.ndarray


def main() -> None:
    l_ed = int(os.environ.get("YAQS_L_ED", "12"))
    l_yaqs = int(os.environ.get("YAQS_L_YAQS", "20"))
    j_xy = float(os.environ.get("YAQS_J", "1.0"))
    deltas = parse_anisotropies(os.environ.get("YAQS_DELTAS", "0.5,1.0,1.5"))
    num_states = int(os.environ.get("YAQS_NUM_STATES", "40"))
    t_final = float(os.environ.get("YAQS_T_FINAL", "50.0"))
    dt = float(os.environ.get("YAQS_DT", "0.5"))
    max_bond_dim = int(os.environ.get("YAQS_MAX_BOND_DIM", "16"))
    threshold = float(os.environ.get("YAQS_THRESHOLD", "1e-9"))
    pad = int(os.environ.get("YAQS_PAD", "4"))
    init_mode = os.environ.get("YAQS_INIT", "haar-random").strip().lower()
    parallel = os.environ.get("YAQS_PARALLEL", "1") != "0"
    show_progress = os.environ.get("YAQS_SHOW_PROGRESS", "1") != "0"

    # Dense ED memory scales as 4^L — guardrail.
    if l_ed > 14:
        print(
            f"Warning: YAQS_L_ED={l_ed} builds a dense {2**l_ed}×{2**l_ed} matrix; "
            "reduce L_ED unless you have sufficient RAM.",
            file=sys.stderr,
        )

    bond_ed = center_bond_left(l_ed)
    print(
        f"XXZ spin-current autocorrelator benchmark: L_ED={l_ed}, L_YAQS={l_yaqs}, "
        f"J={j_xy}, Δ∈{deltas}, N_states={num_states}, T={t_final}, dt={dt}, "
        f"init={init_mode}, pad={pad}, max_bond={max_bond_dim}"
    )

    fig_out = os.environ.get(
        "YAQS_FIG_OUT",
        "scripts/figures/xxz_spin_current_autocorr_yaqs_vs_ed.png",
    )
    out_dir = os.path.dirname(fig_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Collect series per delta for plotting.
    series: list[_SeriesEntry] = []

    print("\nΔ        max|YAQS-ED|  RMS|YAQS-ED|   (full $J$, small-$L$ ensemble)")
    print("--------  ------------  -------------")

    for delta in deltas:
        # --- Exact diagonalization at L_ED ---
        mps_ed_master = _make_initial_ensemble(
            length=l_ed, num_states=num_states, init_mode=init_mode, pad=pad
        )
        ed_vecs = [np.asarray(s.to_vec(), dtype=np.complex128) for s in mps_ed_master]
        times_ed = np.arange(0.0, t_final + dt, dt, dtype=np.float64)
        ed_mean_full_j, _ed_stderr = exact_autocorr_spin_current(
            ed_vecs, l_ed, bond_ed, times_ed, j_xy, delta
        )

        # --- YAQS full-J correlator (same initial MPS as ED) ---
        h_ed = MPO.hamiltonian(
            length=l_ed,
            two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
            bc="periodic",
        )
        yaqs_times_s, yaqs_mean_s = run_yaqs_full_periodic_current_autocorr(
            initial_states=mps_ed_master,
            hamiltonian=h_ed,
            j_xy=j_xy,
            elapsed_time=t_final,
            dt=dt,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            show_progress=show_progress,
            parallel=parallel,
        )
        mx, rms = _metrics_vs_ed(yaqs_mean_s, ed_mean_full_j, times_ed, yaqs_times_s)
        print(f"{delta:<8.4g}  {mx:12.4e}  {rms:13.4e}")

        # --- YAQS full-J correlator at L_YAQS (large-L only) ---
        mps_big = _make_initial_ensemble(
            length=l_yaqs, num_states=num_states, init_mode=init_mode, pad=pad
        )
        h_big = MPO.hamiltonian(
            length=l_yaqs,
            two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
            bc="periodic",
        )
        yaqs_times_b, yaqs_mean_b = run_yaqs_full_periodic_current_autocorr(
            initial_states=mps_big,
            hamiltonian=h_big,
            j_xy=j_xy,
            elapsed_time=t_final,
            dt=dt,
            max_bond_dim=max_bond_dim,
            threshold=threshold,
            show_progress=show_progress,
            parallel=parallel,
        )
        series.append(
            {
                "delta": delta,
                "times_ed": times_ed,
                "ed_mean_full_j": ed_mean_full_j,
                "yaqs_times_s": yaqs_times_s,
                "yaqs_mean_s": yaqs_mean_s,
                "yaqs_times_b": yaqs_times_b,
                "yaqs_mean_b": yaqs_mean_b,
            }
        )

    # ---- Plot: 2 rows × len(deltas) columns ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(deltas)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 6.4), sharex="col", sharey=False)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    delta_tex = r"\Delta"

    for col, entry in enumerate(series):
        d_val = entry["delta"]
        times_ed = entry["times_ed"]
        ed_full = entry["ed_mean_full_j"]
        yts = entry["yaqs_times_s"]
        yms = entry["yaqs_mean_s"]
        ytb = entry["yaqs_times_b"]
        ymb = entry["yaqs_mean_b"]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(times_ed, ed_full, "k-", lw=2.0, label="ED (full $J$)")
        ax_top.plot(yts, yms, "--", lw=1.8, color="C0", label=rf"YAQS (full $J$, $L={l_ed}$)")
        ax_top.set_title(rf"${delta_tex}={d_val:g}$  —  small-$L$ check")
        ax_top.grid(True, alpha=0.35)
        ax_top.legend(fontsize=8)
        ax_top.set_xlabel(r"$t$")
        ax_top.set_yscale("symlog", linthresh=1e-6)
        ax_bot.set_yscale("symlog", linthresh=1e-6)

        if col == 0:
            ax_top.set_ylabel(r"$\mathrm{Re}\,\langle J(t)J\rangle/L$")

        ax_bot.plot(ytb, ymb, lw=2.0, color="C1", label=rf"YAQS (full $J$, $L={l_yaqs}$)")
        ax_bot.set_title(rf"${delta_tex}={d_val:g}$  —  $L={l_yaqs}$")
        ax_bot.grid(True, alpha=0.35)
        ax_bot.legend(fontsize=8)
        ax_bot.set_xlabel(r"$t$")
        if col == 0:
            ax_bot.set_ylabel(r"$\mathrm{Re}\,\langle J(t)J\rangle/L$")
    
    fig.suptitle(
        "XXZ spin-current autocorrelator (periodic chain) — "
        + rf"$J={j_xy:g}$, ${num_states}$ states, {init_mode}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(fig_out, dpi=150)
    plt.close()
    print(f"\nSaved figure: {fig_out}")


if __name__ == "__main__":
    main()
