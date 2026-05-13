# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for XXZ + transverse-field dynamical-quantum-typicality benchmark scripts."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate


@lru_cache(maxsize=1)
def _pauli_xyz() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Pauli matrices X, Y, Z as complex128 (same convention as spin benchmarks)."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return x, y, z


def embed_one_site_operator(length: int, site: int, op2: np.ndarray) -> np.ndarray:
    """Embed a single-qubit operator on ``site`` (left index 0 in lexicographic ``|q_0…q_{L-1}⟩``)."""
    if site < 0 or site >= length:
        msg = f"Invalid site {site} for length {length}."
        raise ValueError(msg)
    left_dim = 2**site
    right_dim = 2 ** (length - site - 1)
    mid = op2.astype(np.complex128)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), mid), np.eye(right_dim, dtype=np.complex128))


def embed_two_site_operator(length: int, site_left: int, op4: np.ndarray) -> np.ndarray:
    """Embed a two-qubit operator on ``(site_left, site_left+1)``."""
    if site_left < 0 or site_left + 1 >= length:
        msg = f"Invalid bond {site_left},{site_left + 1} for length {length}."
        raise ValueError(msg)
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    mid = op4.astype(np.complex128)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), mid), np.eye(right_dim, dtype=np.complex128))


def build_xxz_tf_open_chain_dense(length: int, j_xy: float, delta: float, h_x: float) -> np.ndarray:
    r"""Dense open-chain Hamiltonian ``J\sum(S^x S^x + S^y S^y + \Delta S^z S^z) + h_x \sum S^x``.

    Uses ``S^\alpha = \sigma^\alpha / 2``.
    """
    if length < 2:
        msg = "length must be at least 2."
        raise ValueError(msg)
    x, y, z = _pauli_xyz()
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    dim = 2**length
    h_dense = np.zeros((dim, dim), dtype=np.complex128)
    scale = 0.25
    for i in range(length - 1):
        h_dense += scale * j_xy * embed_two_site_operator(length, i, xx)
        h_dense += scale * j_xy * embed_two_site_operator(length, i, yy)
        h_dense += scale * delta * embed_two_site_operator(length, i, zz)
    for r in range(length):
        h_dense += 0.5 * h_x * embed_one_site_operator(length, r, x)
    return h_dense


def max_center_bond_dim(length: int) -> int:
    """Conservative MPS bond ceiling: ``2 ** ceil(L / 2)`` (enough for exact open-chain evolution at small L)."""
    return 2 ** ((length + 1) // 2)


@dataclass(frozen=True)
class DqtBenchmarkConfig:
    """Environment-driven parameters for DQT scripts."""

    n: int
    j_xy: float
    delta: float
    h_x: float
    t_final: float
    dt: float
    threshold: float
    chis_raw: str
    num_states: int | None
    parallel: bool
    show_progress: bool
    fig_out: str


def parse_dqt_env(
    *,
    default_fig_out: str,
    default_chis: str = "full,16,8",
) -> DqtBenchmarkConfig:
    """Parse ``YAQS_*`` environment variables."""
    n = int(os.environ.get("YAQS_N", "8"))
    if n < 2:
        msg = "YAQS_N must be >= 2."
        raise ValueError(msg)
    if n > 14:
        print(
            f"Warning: YAQS_N={n} builds a dense {2**n}×{2**n} matrix; reduce N unless you have sufficient RAM.",
            file=sys.stderr,
        )
    num_states_raw = os.environ.get("YAQS_NUM_STATES", "").strip()
    num_states: int | None
    if not num_states_raw:
        num_states = None
    else:
        k = int(num_states_raw)
        if k < 1:
            msg = "YAQS_NUM_STATES must be >= 1 when set."
            raise ValueError(msg)
        dim = 2**n
        if k > dim:
            print(
                f"Warning: YAQS_NUM_STATES={k} exceeds Hilbert dimension {dim}; using {dim} basis states.",
                file=sys.stderr,
            )
            k = dim
        num_states = k
    return DqtBenchmarkConfig(
        n=n,
        j_xy=float(os.environ.get("YAQS_J", "1.0")),
        delta=float(os.environ.get("YAQS_DELTA", "0.7")),
        h_x=float(os.environ.get("YAQS_HX", "0.5")),
        t_final=float(os.environ.get("YAQS_T_FINAL", "20.0")),
        dt=float(os.environ.get("YAQS_DT", "0.1")),
        threshold=float(os.environ.get("YAQS_THRESHOLD", "1e-12")),
        chis_raw=os.environ.get("YAQS_CHIS", default_chis),
        num_states=num_states,
        parallel=os.environ.get("YAQS_PARALLEL", "1") != "0",
        show_progress=os.environ.get("YAQS_SHOW_PROGRESS", "1") != "0",
        fig_out=os.environ.get("YAQS_FIG_OUT", default_fig_out),
    )


def parse_chi_tokens(raw: str) -> list[str]:
    """Return trimmed non-empty tokens from a comma-separated list."""
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def build_xxz_tf_mpo(length: int, j_xy: float, delta: float, h_x: float) -> MPO:
    """Open-chain XXZ + uniform transverse field as MPO (``S = σ/2``)."""
    return MPO.hamiltonian(
        length=length,
        two_body=[
            (0.25 * j_xy, "X", "X"),
            (0.25 * j_xy, "Y", "Y"),
            (0.25 * delta, "Z", "Z"),
        ],
        one_body=[(0.5 * h_x, "X")],
        bc="open",
    )


def build_full_basis_ensemble(length: int) -> list[MPS]:
    """All ``2^L`` computational product states as MPS (bond dimension 1)."""
    return [MPS(length, state="basis", basis_string=format(i, f"0{length}b")) for i in range(2**length)]


def build_basis_ensemble(length: int, num_states: int | None) -> list[MPS]:
    """Computational-basis product states as MPS.

    If ``num_states`` is ``None``, returns all ``2^L`` states. Otherwise returns the
    first ``num_states`` basis kets in lexicographic bit order (``|00…0⟩, …``),
    capped at ``2^L``.

    Args:
        length: Chain length ``L``.
        num_states: Number of basis states, or ``None`` for the full Hilbert basis.

    Returns:
        List of MPS, length ``min(num_states, 2^L)`` or ``2^L`` when ``num_states`` is ``None``.
    """
    dim = 2**length
    if num_states is None:
        k = dim
    else:
        k = min(int(num_states), dim)
    if k < 1:
        msg = "Ensemble must contain at least one state."
        raise ValueError(msg)
    return [MPS(length, state="basis", basis_string=format(i, f"0{length}b")) for i in range(k)]


def middle_site(length: int) -> int:
    """Index of the middle qubit (integer division)."""
    return length // 2


def pauli_observable(matrix: NDArray[np.complex128], site: int) -> Observable:
    """Observable wrapping a dense 2×2 Pauli (or other single-qubit) matrix."""
    return Observable(BaseGate(np.asarray(matrix, dtype=np.complex128)), sites=[site])


def pauli_two_time_pairs(mid: int) -> list[tuple[Observable, Observable]]:
    """``(σ_α, σ_α)`` at site ``mid`` for α ∈ {x, y, z}; order matches ED stacking."""
    x, y, z = _pauli_xyz()
    ox, oy, oz = pauli_observable(x, mid), pauli_observable(y, mid), pauli_observable(z, mid)
    return [(ox, ox), (oy, oy), (oz, oz)]


def run_yaqs_dqt_basis_sum(
    *,
    states: list[MPS],
    hamiltonian: MPO,
    elapsed_time: float,
    dt: float,
    max_bond_dim: int,
    threshold: float,
    mid_site: int,
    parallel: bool,
    show_progress: bool,
    track_max_bond: bool,
) -> tuple[np.ndarray, NDArray[np.float64]]:
    """Run full basis ensemble; return ``(times, yaqs_xyz)`` with ``yaqs_xyz`` shape ``(3, n_times)`` real."""
    pairs = pauli_two_time_pairs(mid_site)
    observables: list[Observable] = []
    if track_max_bond:
        observables.append(Observable("max_bond"))

    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=elapsed_time,
        dt=dt,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        order=1,
        sample_timesteps=True,
        show_progress=show_progress,
        compute_autocorrelator=False,
        two_time_correlators=pairs,
    )
    simulator.run(states, hamiltonian, sim_params, noise_model=None, parallel=parallel)
    assert sim_params.two_time_correlator_results is not None
    times = np.asarray(sim_params.times, dtype=np.float64)
    yaqs = np.real(np.asarray(sim_params.two_time_correlator_results, dtype=np.complex128))
    return times, yaqs


def exact_dqt_pauli_autocorr_xyz(
    *,
    length: int,
    j_xy: float,
    delta: float,
    h_x: float,
    times: np.ndarray,
) -> NDArray[np.float64]:
    r"""``Tr[U^\dagger(t) \sigma^\alpha_m U(t) \sigma^\alpha_m] / 2^L`` for α ∈ {x,y,z}; shape ``(3, n_times)``."""
    h_dense = build_xxz_tf_open_chain_dense(length, j_xy, delta, h_x)
    mid = middle_site(length)
    x, y, z = _pauli_xyz()
    ops = [embed_one_site_operator(length, mid, x), embed_one_site_operator(length, mid, y), embed_one_site_operator(length, mid, z)]

    evals, evecs = np.linalg.eigh(h_dense)
    evecs_h = evecs.conj().T
    dim = 2**length
    out = np.zeros((3, len(times)), dtype=np.float64)

    for t_idx, t in enumerate(times):
        phases = np.exp(-1j * evals * t)
        u_t = (evecs * phases[np.newaxis, :]) @ evecs_h
        ud = u_t.conj().T
        for a_idx, op in enumerate(ops):
            corr = np.trace(ud @ op @ u_t @ op) / dim
            out[a_idx, t_idx] = float(np.real(corr))
    return out


def exact_dqt_pauli_autocorr_xyz_first_k_basis_mean(
    *,
    length: int,
    j_xy: float,
    delta: float,
    h_x: float,
    times: np.ndarray,
    k: int,
    mid_site: int | None = None,
) -> NDArray[np.float64]:
    r"""Ensemble mean ``(1/k)\sum_{n=0}^{k-1} \langle n|U^\dagger \sigma U \sigma|n\rangle`` (α ∈ x,y,z).

    Matches the YAQS unitary-ensemble average when initial states are the first ``k``
    computational kets in lexicographic order.
    """
    if k < 1:
        msg = "k must be >= 1."
        raise ValueError(msg)
    dim = 2**length
    k_eff = min(int(k), dim)
    site = middle_site(length) if mid_site is None else int(mid_site)
    h_dense = build_xxz_tf_open_chain_dense(length, j_xy, delta, h_x)
    x, y, z = _pauli_xyz()
    ops = [embed_one_site_operator(length, site, x), embed_one_site_operator(length, site, y), embed_one_site_operator(length, site, z)]

    evals, evecs = np.linalg.eigh(h_dense)
    evecs_h = evecs.conj().T
    out = np.zeros((3, len(times)), dtype=np.float64)

    for t_idx, t in enumerate(times):
        phases = np.exp(-1j * evals * t)
        u_t = (evecs * phases[np.newaxis, :]) @ evecs_h
        ud = u_t.conj().T
        for a_idx, op in enumerate(ops):
            m = ud @ op @ u_t @ op
            acc = 0.0
            for n in range(k_eff):
                acc += float(np.real(m[n, n]))
            out[a_idx, t_idx] = acc / float(k_eff)
    return out


def metrics_vs_ed(yaqs_xyz: NDArray[np.float64], ed_xyz: NDArray[np.float64]) -> tuple[float, float]:
    """Return ``(max_abs_error, rms_error)`` over all three Paulis and times."""
    diff = yaqs_xyz - ed_xyz
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def indices_near_uniform_time_grid(
    times: NDArray[np.float64], *, step: float, t_final: float, tol: float
) -> NDArray[np.intp]:
    """Return indices into ``times`` closest to ``0, step, 2*step, …`` up to ``t_final`` (within ``tol``)."""
    if step <= 0.0:
        msg = "step must be positive."
        raise ValueError(msg)
    targets: list[float] = []
    t = 0.0
    while t <= t_final + tol:
        targets.append(t)
        t += step
    idx: list[int] = []
    for tv in targets:
        j = int(np.argmin(np.abs(times - tv)))
        if float(np.abs(times[j] - tv)) <= tol:
            idx.append(j)
    return np.array(sorted(set(idx)), dtype=np.intp)


def metrics_vs_ed_on_indices(
    yaqs_xyz: NDArray[np.float64], ed_xyz: NDArray[np.float64], idx: NDArray[np.intp]
) -> tuple[float, float]:
    """``metrics_vs_ed`` restricted to time columns ``idx`` (Pauli axis unchanged)."""
    if idx.size == 0:
        return float("nan"), float("nan")
    return metrics_vs_ed(yaqs_xyz[:, idx], ed_xyz[:, idx])
