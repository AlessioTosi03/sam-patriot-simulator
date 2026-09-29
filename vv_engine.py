"""
vv_engine.py - Verification & Validation Engine for SAM Patriot Simulator.

Implements the formal checks from System & Control Theory (Prof. Tronci):
1. LTI Discrete-Time Structural Analysis (Reachability, Eigenvalues, Stability).
2. Empirical Monte Carlo Campaign (Success rate, Miss Distance statistics).
"""

import numpy as np
import time
from environment import get_lti_matrices, simulate_interception


def check_reachability(A: np.ndarray, B: np.ndarray) -> tuple[int, bool, np.ndarray]:
    """
    Computes the Reachability Matrix R = [B, AB, A^2*B, ..., A^(n-1)*B]
    and checks if the system is completely reachable (rank(R) == n).
    """
    n = A.shape[0]
    blocks = [B]
    A_pow = A.copy()

    for _ in range(1, n):
        blocks.append(A_pow @ B)
        A_pow = A_pow @ A

    R = np.hstack(blocks)
    rank = int(np.linalg.matrix_rank(R))
    is_reachable = (rank == n)
    return rank, is_reachable, R


def check_stability(A: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    """
    Computes discrete-time eigenvalues lambda_i of state matrix A.
    - Strictly Asymptotically Stable if |lambda_i| < 1 for all i.
    - Marginally Stable (Lyapunov) if |lambda_i| <= 1.
    """
    eigvals = np.linalg.eigvals(A)
    magnitudes = np.abs(eigvals)

    is_asymptotically_stable = bool(np.all(magnitudes < 1.0 - 1e-6))
    is_marginally_stable = bool(np.all(magnitudes <= 1.0 + 1e-6))

    return eigvals, is_asymptotically_stable, is_marginally_stable


def run_monte_carlo_validation(
    num_runs: int = 50,
    kill_radius: float = 15.0,
    dt: float = 0.05,
    N: float = 4.0,
    verbose: bool = True
) -> dict:
    """
    Runs a batch of Monte Carlo simulations to statistically validate
    the interceptor guidance system against randomized ballistic threats.
    """
    if verbose:
        print(f"\n[V&V] Running Monte Carlo Campaign ({num_runs} simulations)...")

    start_time = time.time()

    miss_distances = []
    intercepted_flags = []
    flight_times = []

    for i in range(num_runs):
        m_traj, t_traj, los_hist, intercepted, min_miss = simulate_interception(
            dt=dt,
            N=N,
            kill_radius=kill_radius
        )
        miss_distances.append(min_miss)
        intercepted_flags.append(intercepted)
        flight_times.append(len(m_traj) * dt)

        if verbose and (i + 1) % 10 == 0:
            print(f"  > Completed {i + 1}/{num_runs} runs...")

    elapsed = time.time() - start_time

    miss_arr = np.array(miss_distances)
    success_rate = (sum(intercepted_flags) / num_runs) * 100.0

    results = {
        "num_runs": num_runs,
        "elapsed_seconds": elapsed,
        "success_rate": success_rate,
        "intercepted_count": sum(intercepted_flags),
        "mean_miss_distance": float(np.mean(miss_arr)),
        "std_miss_distance": float(np.std(miss_arr)),
        "min_miss_distance": float(np.min(miss_arr)),
        "max_miss_distance": float(np.max(miss_arr)),
        "mean_flight_time": float(np.mean(flight_times)),
        "miss_distances": miss_distances
    }

    return results


def run_full_vv_suite(num_monte_carlo: int = 50):
    """
    Executes the complete V&V suite and prints a formatted report
    suitable for academic presentation and project examination.
    """
    print("=" * 70)
    print("VERIFICA E VALIDAZIONE DEI SISTEMI INTELLIGENTI - PATRIOT SAM ENGINE")
    print("=" * 70)

    dt = 0.05
    A, B = get_lti_matrices(dt)

    print("\n1. ANALISI STRUTTURALE LTI (TEMPO DISCRETO)")
    print("-" * 70)
    print(f"Dimensioni dello spazio di stato: n = {A.shape[0]} (Posizione 3D + Velocità 3D)")
    print(f"Dimensioni degli ingressi:         m = {B.shape[1]} (Accelerazione di controllo 3D)")

    rank_R, is_reachable, _ = check_reachability(A, B)
    print(f"\nRaggiungibilità (Controllabilità):")
    print(f"  - Rango Matrice di Raggiungibilità R: {rank_R} / {A.shape[0]}")
    print(f"  - Esito: {'COMPLETAMENTE RAGGIUNGIBILE (Full Rank)' if is_reachable else 'NON RAGGIUNGIBILE'}")

    eigvals, is_asymp, is_marg = check_stability(A)
    print(f"\nStabilità a Ciclo Aperto (Autovalori di A):")
    print(f"  - Autovalori: {np.round(eigvals, 4)}")
    print(f"  - Modulo |lambda_i|: {np.round(np.abs(eigvals), 4)}")
    if is_asymp:
        stab_str = "Asintoticamente Stabile (|lambda| < 1)"
    elif is_marg:
        stab_str = "Marginalmente Stabile (|lambda| = 1, tipico del moto inerziale puro)"
    else:
        stab_str = "Instabile (|lambda| > 1)"
    print(f"  - Esito: {stab_str}")

    print("\n2. VALIDAZIONE STATISTICA MONTE CARLO")
    print("-" * 70)
    mc = run_monte_carlo_validation(num_runs=num_monte_carlo, verbose=True)

    print("\n" + "=" * 70)
    print("REPORT CONCLUSIVO DI VERIFICA E VALIDAZIONE (V&V)")
    print("=" * 70)
    print(f"Numero di simulazioni testate:    {mc['num_runs']}")
    print(f"Tempo di esecuzione:              {mc['elapsed_seconds']:.2f} s ({mc['elapsed_seconds']/mc['num_runs']*1000:.1f} ms/run)")
    print(f"Tasso di Successo (Intercettati):  {mc['success_rate']:.1f}% ({mc['intercepted_count']}/{mc['num_runs']})")
    print(f"Miss Distance Media:              {mc['mean_miss_distance']:.2f} m (± {mc['std_miss_distance']:.2f} m)")
    print(f"Miss Distance Minima:             {mc['min_miss_distance']:.2f} m")
    print(f"Miss Distance Massima:            {mc['max_miss_distance']:.2f} m")
    print(f"Tempo Medio di Volo:              {mc['mean_flight_time']:.2f} s")
    print("=" * 70)


if __name__ == '__main__':
    run_full_vv_suite(num_monte_carlo=50)
