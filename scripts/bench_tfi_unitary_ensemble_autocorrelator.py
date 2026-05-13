# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark script for unitary TFI autocorrelators with an MPS ensemble.

This script computes the onsite autocorrelator

    C_i(t) = <Z_i(t) Z_i(0)>

for the transverse-field Ising (TFI) model using an ensemble of random initial
MPS states. It reports parallel and serial runtimes, and plots the ensemble mean
with a shaded 1-sigma standard error region.

Run:
    uv run python tests/analog/bench_tfi_unitary_ensemble_autocorrelator.py
"""

from __future__ import annotations

import copy
import importlib
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import numpy as np

from mqt.yaqs.analog.unitary_ensemble import unitary_ensemble_member_worker
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z

_BENCH_CTX: dict[str, object] = {}


def _ensure_quspin() -> None:
    """Ensure QuSpin is available in the current Python environment."""
    try:
        importlib.import_module("quspin")
        print("[ed] quspin is available")
        return
    except Exception as exc:
        print(f"[ed] quspin import failed ({type(exc).__name__}); attempting repair...")
        last_error: Exception = exc
    else:
        return

    try:
        print("[ed] quspin not found or broken; installing in current environment...")
        install_commands = [
            [sys.executable, "-m", "pip", "install", "quspin"],
            ["uv", "pip", "install", "--python", sys.executable, "quspin"],
        ]
        install_error: subprocess.CalledProcessError | None = None
        for command in install_commands:
            try:
                subprocess.check_call(command)
                install_error = None
                break
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                if isinstance(exc, subprocess.CalledProcessError):
                    install_error = exc
                continue
        if install_error is not None:
            msg = "Failed to install quspin automatically. Please run: uv pip install --python <your-python> quspin"
            raise RuntimeError(msg) from install_error

        try:
            importlib.import_module("quspin")
            print("[ed] quspin installation complete")
            return
        except ImportError as exc:
            last_error = exc

        if sys.platform == "darwin" and "libomp" in str(last_error):
            print("[ed] missing libomp detected; attempting: brew install libomp")
            subprocess.check_call(["brew", "install", "libomp"])
            importlib.import_module("quspin")
            print("[ed] quspin import fixed after libomp installation")
            return

        raise last_error
    except Exception as exc:
        msg = "Unable to prepare QuSpin in this environment."
        raise RuntimeError(msg) from exc


def _build_quspin_hamiltonian(length: int, coupling: float, field: float):
    """Build TFI Hamiltonian matching YAQS MPO.ising: -J ZZ - g X."""
    quspin_basis_mod = importlib.import_module("quspin.basis")
    quspin_ops_mod = importlib.import_module("quspin.operators")
    spin_basis_1d = getattr(quspin_basis_mod, "spin_basis_1d")
    hamiltonian = getattr(quspin_ops_mod, "hamiltonian")

    basis = spin_basis_1d(length)
    zz_terms = [[-coupling, i, i + 1] for i in range(length - 1)]
    x_terms = [[-field, i] for i in range(length)]
    static = [["zz", zz_terms], ["x", x_terms]]
    h_qs = hamiltonian(static, [], basis=basis, dtype=np.complex128)
    return h_qs, basis


def _exact_autocorr_quspin(
    initial_states: list[MPS],
    length: int,
    center_site: int,
    times: np.ndarray,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ensemble mean/stderr of C(t)=<Z_i(t)Z_i(0)> exactly via QuSpin ED."""
    h_qs, basis = _build_quspin_hamiltonian(length, coupling, field)
    h_dense = np.asarray(h_qs.toarray(), dtype=np.complex128)
    evals, evecs = np.linalg.eigh(h_dense)
    evecs_dag = evecs.conj().T

    quspin_ops_mod = importlib.import_module("quspin.operators")
    hamiltonian = getattr(quspin_ops_mod, "hamiltonian")
    z_i = hamiltonian([["z", [[1.0, center_site]]]], [], basis=basis, dtype=np.complex128).toarray()

    phases = np.exp(-1j * np.outer(evals, times))
    correlations = np.zeros((len(initial_states), len(times)), dtype=np.complex128)

    for idx, state in enumerate(initial_states):
        psi0 = np.asarray(state.to_vec(), dtype=np.complex128)
        psi0 /= np.linalg.norm(psi0)
        phi0 = z_i @ psi0

        psi_e = evecs_dag @ psi0
        phi_e = evecs_dag @ phi0

        for t_idx in range(len(times)):
            psi_t = evecs @ (phases[:, t_idx] * psi_e)
            phi_t = evecs @ (phases[:, t_idx] * phi_e)
            correlations[idx, t_idx] = np.vdot(psi_t, z_i @ phi_t)

    mean_corr = np.real(np.mean(correlations, axis=0))
    stderr_corr = np.std(np.real(correlations), axis=0, ddof=1) / np.sqrt(len(initial_states))
    return mean_corr, stderr_corr


def _bench_worker_init(payload: dict[str, object]) -> None:
    """Initialize per-process worker context for benchmark parallel mode."""
    _BENCH_CTX.clear()
    _BENCH_CTX.update(payload)


def _bench_parallel_worker(idx: int) -> np.ndarray:
    """Parallel benchmark worker retrieving heavy objects from process-local context."""
    initial_states = _BENCH_CTX["initial_states"]
    sim_params = _BENCH_CTX["sim_params"]
    hamiltonian = _BENCH_CTX["hamiltonian"]
    assert isinstance(initial_states, list)
    assert isinstance(sim_params, AnalogSimParams)
    assert isinstance(hamiltonian, MPO)
    typed_states = cast("list[MPS]", initial_states)
    _obs_results, autocorr = unitary_ensemble_member_worker((idx, typed_states[idx], sim_params, hamiltonian))
    assert autocorr is not None
    return np.asarray(autocorr)


def _run_serial(
    initial_states: list[MPS],
    sim_params: AnalogSimParams,
    hamiltonian: MPO,
) -> np.ndarray:
    """Run the autocorrelator worker serially for all initial states."""
    trajectories: list[np.ndarray] = []
    print(f"[serial] workers=1, states={len(initial_states)}")
    for idx, state in enumerate(initial_states):
        _obs_results, autocorr = unitary_ensemble_member_worker((idx, state, sim_params, hamiltonian))
        assert autocorr is not None
        trajectories.append(np.asarray(autocorr))
        if idx % 10 == 0:
            print(f"[serial] completed {idx + 1}/{len(initial_states)} states")
    return np.asarray(trajectories, dtype=np.complex128)


def _run_parallel(
    initial_states: list[MPS],
    sim_params: AnalogSimParams,
    hamiltonian: MPO,
    max_workers: int,
) -> np.ndarray:
    """Run the autocorrelator worker in parallel across initial states."""
    trajectories: list[np.ndarray] = []
    payload: dict[str, object] = {
        "initial_states": initial_states,
        "sim_params": sim_params,
        "hamiltonian": hamiltonian,
    }

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_bench_worker_init,
        initargs=(payload,),
    ) as executor:
        print(f"[parallel] workers={max_workers}, states={len(initial_states)}")
        for idx, autocorr in enumerate(executor.map(_bench_parallel_worker, range(len(initial_states)))):
            trajectories.append(autocorr)
            if idx % 10 == 0:
                print(f"[parallel] completed {idx + 1}/{len(initial_states)} states")

    return np.asarray(trajectories, dtype=np.complex128)


def _save_plot(
    times: np.ndarray,
    mean_corr: np.ndarray,
    stderr_corr: np.ndarray,
    center_site: int,
    num_states: int,
    output_path: str,
) -> bool:
    """Save autocorrelator plot; return True on success."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")

        plt.figure(figsize=(8, 4.5))
        plt.plot(times, mean_corr, label=r"Ensemble mean $\langle Z_i(t)Z_i(0)\rangle$", linewidth=2)
        plt.fill_between(
            times,
            mean_corr - stderr_corr,
            mean_corr + stderr_corr,
            alpha=0.25,
            label=r"$\pm$ 1 standard error",
        )
        plt.xlabel("time")
        plt.ylabel("autocorrelator")
        plt.title(f"TFI autocorrelator at site i={center_site} (N={num_states} random MPS)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except ModuleNotFoundError:
        return False


def _save_combined_autocorr_plot(
    output_path: str,
    center_site: int,
    num_states: int,
    ed_times: np.ndarray,
    ed_mean_corr: np.ndarray,
    sweep_rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float]],
) -> bool:
    """Save combined YAQS vs ED autocorrelator plot."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        plt.figure(figsize=(9, 5))
        plt.plot(ed_times, ed_mean_corr, color="black", linewidth=2.5, label="ED (QuSpin)")
        for dt, times, mean_corr, _stderr_corr, _runtime, _max_abs_err, _rms_err in sweep_rows:
            plt.plot(times, mean_corr, linewidth=1.8, label=f"YAQS dt={dt:g}")
        plt.xlabel("time")
        plt.ylabel(r"autocorrelator $\langle Z_i(t) Z_i(0) \rangle$")
        plt.title(f"TFI autocorrelator comparison at site i={center_site} (N={num_states} random MPS)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except ModuleNotFoundError:
        return False


def _save_error_vs_ed_plot(
    output_path: str,
    ed_times: np.ndarray,
    ed_mean_corr: np.ndarray,
    sweep_rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float]],
) -> bool:
    """Save absolute error-vs-time plot for each dt against ED."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        plt.figure(figsize=(9, 5))
        for dt, times, mean_corr, _stderr_corr, _runtime, _max_abs_err, _rms_err in sweep_rows:
            ed_interp = np.interp(times, ed_times, ed_mean_corr)
            abs_err = np.abs(mean_corr - ed_interp)
            plt.plot(times, abs_err, linewidth=1.8, label=f"|YAQS-ED| dt={dt:g}")
        plt.xlabel("time")
        plt.ylabel("absolute error")
        plt.title("YAQS autocorrelator absolute error vs ED")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except ModuleNotFoundError:
        return False


def main() -> None:
    """Run benchmark and plot autocorrelator with shaded statistical error."""
    # -------------------------
    # Model / simulation setup
    # -------------------------
    length = 8
    coupling = 1.0
    field = 0.8
    num_states = int(os.environ.get("YAQS_NUM_STATES", "200"))
    t_final = float(os.environ.get("YAQS_T_FINAL", "100.0"))
    dt_sweep_env = os.environ.get("YAQS_DT_SWEEP", "1.0,0.5,0.25")
    dt_sweep = [float(token.strip()) for token in dt_sweep_env.split(",") if token.strip()]
    if any(dt <= 0 for dt in dt_sweep):
        msg = "All dt values in YAQS_DT_SWEEP must be positive."
        raise ValueError(msg)
    dt_sweep = sorted(set(dt_sweep), reverse=True)
    pad = 8  # bond-dimension cap for Haar-random initialization

    hamiltonian = MPO.ising(length, coupling, field)
    center_site = length // 2
    operator = Observable(Z(), center_site)

    # Random initial ensemble for typicality.
    # Build one master list, then deep-copy it for ED and YAQS to guarantee both
    # consume identical initial states while remaining mutation-isolated.
    master_initial_states = [MPS(length, state="random", pad=pad) for _ in range(num_states)]
    for state in master_initial_states:
        state.normalize("B")
    ed_initial_states = [copy.deepcopy(state) for state in master_initial_states]
    yaqs_initial_states = [copy.deepcopy(state) for state in master_initial_states]
    cpu_count = os.cpu_count() or 1
    print(
        "Benchmark setup: "
        f"L={length}, J={coupling}, g={field}, states={num_states}, "
        f"t_final={t_final}, dt_sweep={dt_sweep}"
    )
    print(f"Host CPU count: {cpu_count}")
    print(
        "Thread env hints: "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}, "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}"
    )

    max_workers = max(1, cpu_count - 1)
    reference_dt = min(dt_sweep)
    reference_params = AnalogSimParams(
        observables=[],
        elapsed_time=t_final,
        dt=reference_dt,
        max_bond_dim=64,
        threshold=1e-9,
        order=1,
        sample_timesteps=True,
        show_progress=False,
        compute_autocorrelator=True,
        autocorrelator_observable=operator,
    )
    reference_times = reference_params.times
    print(f"[ed] computing exact reference with quspin (time_points={len(reference_times)})")
    _ensure_quspin()
    ed_start = time.perf_counter()
    ed_mean_corr, ed_stderr_corr = _exact_autocorr_quspin(
        initial_states=ed_initial_states,
        length=length,
        center_site=center_site,
        times=reference_times,
        coupling=coupling,
        field=field,
    )
    ed_time = time.perf_counter() - ed_start
    print(f"[ed] completed in {ed_time:.3f} s")

    print("\n[dt sweep] parallel YAQS vs QuSpin ED")
    print("dt        runtime[s]   max_abs_err   rms_err")
    print("--------  -----------  -----------   -----------")
    sweep_rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float]] = []
    for dt in dt_sweep:
        sim_params = AnalogSimParams(
            observables=[],
            elapsed_time=t_final,
            dt=dt,
            max_bond_dim=64,
            threshold=1e-9,
            order=1,
            sample_timesteps=True,
            show_progress=False,
            compute_autocorrelator=True,
            autocorrelator_observable=operator,
        )
        times = sim_params.times
        print(f"[parallel] launching dt={dt}, max_workers={max_workers}, time_points={len(times)}")
        t0 = time.perf_counter()
        corr_parallel = _run_parallel(yaqs_initial_states, sim_params, hamiltonian, max_workers=max_workers)
        runtime = time.perf_counter() - t0

        corr_real = np.real(corr_parallel)
        mean_corr = np.mean(corr_real, axis=0)
        stderr_corr = np.std(corr_real, axis=0, ddof=1) / np.sqrt(num_states)

        ed_interp = np.interp(times, reference_times, ed_mean_corr)
        error = mean_corr - ed_interp
        max_abs_err = float(np.max(np.abs(error)))
        rms_err = float(np.sqrt(np.mean(error**2)))
        print(f"{dt:<8.4g}  {runtime:>11.3f}  {max_abs_err:>11.3e}   {rms_err:>11.3e}")
        sweep_rows.append((dt, times, mean_corr, stderr_corr, runtime, max_abs_err, rms_err))

    # Keep one serial timing run (on the coarsest dt) for reference
    serial_dt = max(dt_sweep)
    serial_params = AnalogSimParams(
        observables=[],
        elapsed_time=t_final,
        dt=serial_dt,
        max_bond_dim=64,
        threshold=1e-9,
        order=1,
        sample_timesteps=True,
        show_progress=False,
        compute_autocorrelator=True,
        autocorrelator_observable=operator,
    )
    run_serial_full = os.environ.get("YAQS_SERIAL_FULL", "0") == "1"
    default_serial_states = min(10, num_states)
    serial_bench_states = int(os.environ.get("YAQS_SERIAL_BENCH_STATES", str(default_serial_states)))
    serial_bench_states = max(1, min(serial_bench_states, num_states))
    serial_states_count = num_states if run_serial_full else serial_bench_states
    print(
        f"\n[serial] reference run at dt={serial_dt}: "
        f"mode={'full' if run_serial_full else 'subset'}, states={serial_states_count}/{num_states}"
    )
    serial_states = [copy.deepcopy(state) for state in yaqs_initial_states[:serial_states_count]]
    t1 = time.perf_counter()
    corr_serial = _run_serial(serial_states, serial_params, hamiltonian)
    serial_time = time.perf_counter() - t1
    serial_time_per_state = serial_time / serial_states_count
    serial_time_est_full = serial_time_per_state * num_states
    coarse_parallel = next(row for row in sweep_rows if row[0] == serial_dt)
    coarse_parallel_corr = coarse_parallel[2]
    serial_mean_corr = np.mean(np.real(corr_serial), axis=0)
    max_abs_diff = float(np.max(np.abs(coarse_parallel_corr - serial_mean_corr)))
    if run_serial_full:
        speedup = serial_time / coarse_parallel[4]
        print(f"[serial] runtime(full): {serial_time:.3f} s, speedup(parallel): {speedup:.2f}x")
    else:
        speedup_est = serial_time_est_full / coarse_parallel[4]
        print(f"[serial] runtime(subset): {serial_time:.3f} s, est_full: {serial_time_est_full:.3f} s")
        print(f"[serial] estimated speedup(parallel): {speedup_est:.2f}x")
    print(f"[serial] max |C_parallel(dt={serial_dt}) - C_serial| on subset: {max_abs_diff:.3e}")

    # -------------------------
    # Plots
    # -------------------------
    output_path = "tests/analog/bench_tfi_unitary_ensemble_autocorrelator.png"
    best_row = min(sweep_rows, key=lambda row: row[0])
    best_times = best_row[1]
    best_mean = best_row[2]
    best_stderr = best_row[3]
    plot_ok = _save_plot(best_times, best_mean, best_stderr, center_site, num_states, output_path)
    if plot_ok:
        print(f"\nSaved plot to: {output_path} (YAQS mean for dt={best_row[0]})")
    else:
        print("\nPlot skipped: matplotlib is not available in the current environment.")

    ed_plot_output = "tests/analog/bench_tfi_unitary_ensemble_autocorrelator_ed.png"
    ed_plot_ok = _save_plot(reference_times, ed_mean_corr, ed_stderr_corr, center_site, num_states, ed_plot_output)
    if ed_plot_ok:
        print(f"Saved ED plot to: {ed_plot_output}")
    else:
        print("ED plot skipped: matplotlib is not available in the current environment.")

    combined_output = "tests/analog/bench_tfi_unitary_ensemble_vs_ed.png"
    combined_ok = _save_combined_autocorr_plot(
        output_path=combined_output,
        center_site=center_site,
        num_states=num_states,
        ed_times=reference_times,
        ed_mean_corr=ed_mean_corr,
        sweep_rows=sweep_rows,
    )
    if combined_ok:
        print(f"Saved combined YAQS-vs-ED plot to: {combined_output}")
    else:
        print("Combined plot skipped: matplotlib is not available in the current environment.")

    error_output = "tests/analog/bench_tfi_unitary_ensemble_error_vs_ed.png"
    error_ok = _save_error_vs_ed_plot(
        output_path=error_output,
        ed_times=reference_times,
        ed_mean_corr=ed_mean_corr,
        sweep_rows=sweep_rows,
    )
    if error_ok:
        print(f"Saved error-vs-ED plot to: {error_output}")
    else:
        print("Error plot skipped: matplotlib is not available in the current environment.")


if __name__ == "__main__":
    main()
