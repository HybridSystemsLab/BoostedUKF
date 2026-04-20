# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

# =========================
# Quaternion / SO(3) utils
# =========================
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_from_omega(ω, dt):
    θ = np.linalg.norm(ω*dt)
    if θ < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = ω / np.linalg.norm(ω)
    half = 0.5*θ
    return np.hstack((np.cos(half), axis*np.sin(half)))

def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

# ===============
# Global scenario
# ===============
np.random.seed(42)
J_TRUE = np.diag([100, 80, 70])  # True inertia matrix (kg·m²)
dt, T_RUN = 0.01, 400.0
steps = int(T_RUN/dt)

# Monte Carlo settings
INITIAL_INERTIA_MEAN = np.array([140.0, 20.0, 36.06])
INITIAL_INERTIA_STD  = np.array([10.0, 10.0, 10.0])
N_MONTE_CARLO = 50

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def torque_profile(t, mode="full"):
    """Torque profile with full, windowed, or multi-pulse (persistent) excitation."""

    def base_signal(tt):
        """Underlying torque signal."""
        return np.array([
            1.0*np.sin(0.1*tt) + 2.5*np.cos(0.3*tt) + 1.0*np.sin(0.7*tt) + 1.0*np.sin(1.5*tt),
            2.6*np.cos(0.15*tt) + 3.0*np.sin(0.4*tt) + 2.4*np.cos(0.8*tt) + 1.8*np.cos(1.8*tt),
            3.4*np.sin(0.12*tt) + 2.1*np.cos(0.5*tt) + 1.0*np.sin(0.9*tt) + 1.5*np.sin(2.0*tt)
        ])

    if mode == "full":
        # Continuous excitation throughout
        return base_signal(t)

    elif mode == "window":
        # Single short excitation bursts — three pulses spread across the run
        in_pulse = ((200.0 <= t < 201.0) or
                    (250.0 <= t < 251.0) or
                    (300.0 <= t < 301.0))
        if not in_pulse:
            return np.zeros(3)
        return base_signal(t)

    elif mode == "multi":
        # Persistent excitation — same amplitude, repeated 20 times
        # Start every 20s from 10s to 390s, each lasting 10s
        pulse_intervals = [(50 + 25*i, 51 + 25*i) for i in range(20)]
        in_pulse = any(a <= t < b for a, b in pulse_intervals)
        if not in_pulse:
            return np.zeros(3)
        return base_signal(t)

    else:
        raise ValueError("Mode must be 'full', 'window', or 'multi'.")

# ==================
# Main UKF pipeline
# ==================
def run_simulation(excitation_mode="full", initial_inertia=None, verbose=True):
    # -------- Forward (truth) simulation --------
    ω = np.array([0.1, 0.1, 0.1])
    q_body_cam = np.array([1., 0., 0., 0.])
    ω_hist = np.empty((steps, 3))
    q_hist = np.empty((steps, 4))
    τ_hist = np.empty((steps, 3))
    t_hist = np.linspace(0.0, T_RUN, steps)

    for k, t in enumerate(t_hist):
        τ = torque_profile(t, mode=excitation_mode)
        ω̇ = np.linalg.inv(J_TRUE) @ (τ - skew(ω) @ (J_TRUE @ ω))
        ω += ω̇ * dt
        q_body_cam = quat_mult(q_body_cam, quat_from_omega(ω, dt))
        q_body_cam /= np.linalg.norm(q_body_cam)

        ω_hist[k] = ω
        q_hist[k] = q_body_cam
        τ_hist[k] = τ

    # -------- Synthetic measurements (keep your parameters) --------
    σ_gyro = 0.005
    σ_quat = 0.005
    ω_meas = ω_hist + np.random.normal(0.0, σ_gyro, ω_hist.shape)
    q_meas = q_hist + np.random.normal(0.0, σ_quat, q_hist.shape)
    q_meas /= np.linalg.norm(q_meas, axis=1, keepdims=True)
    z_meas = np.hstack((q_meas, ω_meas))

    # -------- UKF setup (keep your parameters) --------
    dim_x, dim_z = 10, 7  # State: [q(4), ω(3), J(3)]
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.001, beta=2.0, kappa=0.0)

    def fx(x, dt, torque):
        q = x[:4]
        ω = x[4:7]
        J_diag = x[7:]                 # Diagonal inertia components
        J_diag = np.maximum(J_diag, 1.0)
        J = np.diag(J_diag)
        ω̇ = np.linalg.inv(J) @ (torque - skew(ω) @ (J @ ω))
        ω_new = ω + ω̇ * dt
        q_new = quat_mult(q, quat_from_omega(ω_new, dt))
        q_new /= np.linalg.norm(q_new)
        return np.hstack((q_new, ω_new, J_diag))

    def hx(x):
        return np.hstack((x[:4], x[4:7]))

    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points)

    # Initial state — same poor inertia guess you used
    ukf.x[:4] = q_meas[0]
    ukf.x[4:7] = ω_meas[0]
    if initial_inertia is None:
        ukf.x[7:] = np.array([140.0, 20.0, 36.06])
    else:
        ukf.x[7:] = np.maximum(np.array(initial_inertia, dtype=float), 1.0)

    # Initial covariance (unchanged)
    ukf.P = np.eye(dim_x)
    ukf.P[:4, :4] *= 0.001
    ukf.P[4:7, 4:7] *= 0.01
    ukf.P[7:10, 7:10] = np.diag([1700.0, 20.0, 120])

    # Measurement noise covariance (unchanged)
    ukf.R = np.diag([σ_quat**2]*4 + [σ_gyro**2]*3)

    # Process noise covariance (keep your 1e-5 values)
    Q_quat = 1e-7
    Q_omega = 1e-7
    Q_inertia = 1e-7
    Q_base = np.diag([Q_quat]*4 + [Q_omega]*3 + [Q_inertia]*3)

    # Torque threshold below which inertia is frozen
    torque_threshold = 1e-6

    # -------- Estimation loop --------
    J_hist = np.empty((steps, 3))
    trace_P = np.empty(steps)
    nis_hist = np.empty(steps)
    P_diag_inertia = np.empty((steps, 3))

    for k, t in enumerate(t_hist):
        τ = τ_hist[k]
        torque_active = np.linalg.norm(τ) > torque_threshold

        # Always use base Q; zero inertia process noise when no torque
        ukf.Q = Q_base.copy()
        if not torque_active:
            ukf.Q[7:10, 7:10] = 0.0

        ukf.predict(torque=τ)

        if torque_active:
            # Normal update — let all states be corrected
            ukf.update(z_meas[k])
        else:
            # Freeze inertia: zero cross-covariance between J and everything else,
            # then update (Kalman gain rows 7-9 will be zero), then re-enforce freeze
            J_frozen   = ukf.x[7:].copy()
            PJJ_frozen = ukf.P[7:10, 7:10].copy()

            # Decouple inertia from the rest of P so gain K[7:10,:] = 0
            ukf.P[7:10, :7] = 0.0
            ukf.P[:7, 7:10] = 0.0

            ukf.update(z_meas[k])

            # Hard-restore inertia mean and variance
            ukf.x[7:]         = J_frozen
            ukf.P[7:10, 7:10] = PJJ_frozen
            ukf.P[7:10, :7]   = 0.0
            ukf.P[:7, 7:10]   = 0.0

            # Re-symmetrise and add small ridge to guarantee positive-definiteness
            ukf.P = 0.5 * (ukf.P + ukf.P.T)
            ukf.P += np.eye(ukf.P.shape[0]) * 1e-9

        # Normalize quaternion and ensure positive inertia (keep your choices)
        ukf.x[:4] /= np.linalg.norm(ukf.x[:4])
        ukf.x[7:] = np.maximum(ukf.x[7:], 1.0)

        # Logs
        J_hist[k] = ukf.x[7:]
        trace_P[k] = np.trace(ukf.P[7:10, 7:10])
        P_diag_inertia[k] = np.diag(ukf.P[7:10, 7:10])

        # NIS calculation (same as your approach)
        if hasattr(ukf, 'S') and hasattr(ukf, 'y'):
            try:
                S_inv = np.linalg.inv(ukf.S)
                nis_hist[k] = float(ukf.y.T @ S_inv @ ukf.y)
            except Exception:
                nis_hist[k] = 0.0
        else:
            nis_hist[k] = 0.0

        if verbose and k % 5000 == 0:
            curJ = J_hist[k] if np.all(np.isfinite(J_hist[k])) else "NaN"
            print(f"  Progress: {k/steps*100:.1f}% - Current J: {curJ}")

    if verbose:
        print("UKF simulation completed for mode:", excitation_mode)
    return dict(
        t_hist=t_hist, J_hist=J_hist, nis_hist=nis_hist,
        trace_P=trace_P, P_diag_inertia=P_diag_inertia,
        q_hist=q_hist, ω_hist=ω_hist, τ_hist=τ_hist,
        final_J=J_hist[-1], final_P=ukf.P.copy(),
        ω_meas=ω_meas, q_meas=q_meas, final_x=ukf.x.copy()
    )

# =========================
# Monte Carlo functionality
# =========================
def compute_relative_errors(final_J):
    """Relative absolute error (%) for Jx, Jy, Jz."""
    J_true = np.diag(J_TRUE)
    return np.abs(final_J - J_true) / J_true * 100.0

def run_monte_carlo_study(n_samples=N_MONTE_CARLO,
                          initial_mean=INITIAL_INERTIA_MEAN,
                          initial_std=INITIAL_INERTIA_STD):
    """
    Run Monte Carlo study over sampled initial inertia guesses.
    For each case (full/window/multi), compute mean ± std of relative error (% of nominal).
    Also prints elapsed time and estimated remaining time.
    """
    cases = ["full", "window", "multi"]
    samples = np.random.normal(loc=initial_mean, scale=initial_std, size=(n_samples, 3))
    samples = np.maximum(samples, 1.0)

    mc_results = {
        "samples": samples,
        "full":   {"final_J": np.zeros((n_samples, 3)), "rel_errors": np.zeros((n_samples, 3))},
        "window": {"final_J": np.zeros((n_samples, 3)), "rel_errors": np.zeros((n_samples, 3))},
        "multi":  {"final_J": np.zeros((n_samples, 3)), "rel_errors": np.zeros((n_samples, 3))}
    }

    total_runs = len(cases) * n_samples
    completed_runs = 0
    global_start_time = time.time()

    for case in cases:
        case_start_time = time.time()
        print(f"\nRunning Monte Carlo for case: {case}")

        for s in range(n_samples):
            results = run_simulation(
                excitation_mode=case,
                initial_inertia=samples[s],
                verbose=False
            )

            mc_results[case]["final_J"][s, :] = results["final_J"]
            mc_results[case]["rel_errors"][s, :] = compute_relative_errors(results["final_J"])

            completed_runs += 1

            if (s + 1) % 10 == 0 or (s + 1) == n_samples:
                elapsed_total = time.time() - global_start_time
                elapsed_case = time.time() - case_start_time
                avg_time_per_run = elapsed_total / completed_runs
                remaining_runs = total_runs - completed_runs
                est_remaining_sec = avg_time_per_run * remaining_runs

                print(
                    f"  {case}: completed {s + 1}/{n_samples} samples | "
                    f"elapsed = {elapsed_total/60:.2f} min | "
                    f"case elapsed = {elapsed_case/60:.2f} min | "
                    f"estimated remaining = {est_remaining_sec/60:.2f} min"
                )

        mc_results[case]["mean_rel_error"] = np.mean(mc_results[case]["rel_errors"], axis=0)
        mc_results[case]["std_rel_error"]  = np.std(mc_results[case]["rel_errors"], axis=0, ddof=1)
        mc_results[case]["mean_final_J"]   = np.mean(mc_results[case]["final_J"], axis=0)
        mc_results[case]["std_final_J"]    = np.std(mc_results[case]["final_J"], axis=0, ddof=1)

    total_elapsed = time.time() - global_start_time
    print(f"\nTotal Monte Carlo elapsed time: {total_elapsed/60:.2f} minutes")

    return mc_results

def print_monte_carlo_results(mc_results):
    """Print Monte Carlo mean ± std summaries."""
    label_map = {
        "full": "FULL",
        "window": "WINDOWED",
        "multi": "PERSISTENT"
    }

    print(f"\n{'='*80}")
    print("MONTE CARLO SUMMARY: MEAN ± STD OF RELATIVE ERROR (% OF NOMINAL)")
    print(f"{'='*80}")
    print(f"Initial inertia mean used for sampling: [{INITIAL_INERTIA_MEAN[0]:.2f}, {INITIAL_INERTIA_MEAN[1]:.2f}, {INITIAL_INERTIA_MEAN[2]:.2f}]")
    print(f"Initial inertia std used for sampling:  [{INITIAL_INERTIA_STD[0]:.2f}, {INITIAL_INERTIA_STD[1]:.2f}, {INITIAL_INERTIA_STD[2]:.2f}]")
    print(f"Number of Monte Carlo samples: {mc_results['samples'].shape[0]}")

    for case in ["full", "window", "multi"]:
        mu = mc_results[case]["mean_rel_error"]
        sd = mc_results[case]["std_rel_error"]
        mean_final_J = mc_results[case]["mean_final_J"]
        std_final_J = mc_results[case]["std_final_J"]

        print(f"\n{'-'*80}")
        print(f"{label_map[case]} EXCITATION")
        print(f"{'-'*80}")
        print(f"Mean final J estimates: [{mean_final_J[0]:8.4f}, {mean_final_J[1]:8.4f}, {mean_final_J[2]:8.4f}] kg·m²")
        print(f"Std  final J estimates: [{std_final_J[0]:8.4f}, {std_final_J[1]:8.4f}, {std_final_J[2]:8.4f}] kg·m²")
        print(f"Jx relative error: {mu[0]:.4f} ± {sd[0]:.4f} % of nominal")
        print(f"Jy relative error: {mu[1]:.4f} ± {sd[1]:.4f} % of nominal")
        print(f"Jz relative error: {mu[2]:.4f} ± {sd[2]:.4f} % of nominal")

    print(f"\n{'='*80}")
    print("TABLE-STYLE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Case':<12}{'Jx error (%)':<24}{'Jy error (%)':<24}{'Jz error (%)':<24}")
    for case in ["full", "window", "multi"]:
        mu = mc_results[case]["mean_rel_error"]
        sd = mc_results[case]["std_rel_error"]
        label = label_map[case]
        print(f"{label:<12}"
              f"{f'{mu[0]:.4f} ± {sd[0]:.4f}':<24}"
              f"{f'{mu[1]:.4f} ± {sd[1]:.4f}':<24}"
              f"{f'{mu[2]:.4f} ± {sd[2]:.4f}':<24}")

# ================
# Plotting (uniform; no legends; no inset; no window shading)
# ================
def plot_results(results, mode_name=None):
    LABEL_FS  = 28
    TICK_FS   = 22
    LW        = 2

    t = results['t_hist']
    J = results['J_hist']
    σ = np.sqrt(np.maximum(results['P_diag_inertia'], 0.0))

    # ----------------- 1) Inertia with ±1σ bands -----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, J[:, 0], 'b-', lw=LW)                           # Est. Jx
    ax.fill_between(t, J[:, 0]-σ[:, 0], J[:, 0]+σ[:, 0],
                    color='b', alpha=0.12)
    ax.plot(t, J[:, 1], 'g-', lw=LW)                           # Est. Jy
    ax.fill_between(t, J[:, 1]-σ[:, 1], J[:, 1]+σ[:, 1],
                    color='g', alpha=0.12)
    ax.plot(t, J[:, 2], 'r-', lw=LW)                           # Est. Jz
    ax.fill_between(t, J[:, 2]-σ[:, 2], J[:, 2]+σ[:, 2],
                    color='r', alpha=0.12)

    # True lines
    ax.axhline(100, color='b', ls='--', alpha=0.7)             # True Jx
    ax.axhline(80,  color='g', ls='--', alpha=0.7)             # True Jy
    ax.axhline(70,  color='r', ls='--', alpha=0.7)             # True Jz

    ax.set_xlabel('Time [s]',          fontsize=LABEL_FS)
    ax.set_ylabel('Inertia [kg·m²]',   fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 250) 
    plt.tight_layout()
    plt.show()   # no legend

    # ----------------- 2) Inertia standard deviation --------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t, σ[:, 0], 'b-', lw=LW)
    ax.semilogy(t, σ[:, 1], 'g-', lw=LW)
    ax.semilogy(t, σ[:, 2], 'r-', lw=LW)
    ax.set_xlabel('Time [s]',            fontsize=LABEL_FS)
    ax.set_ylabel('Std. dev. [kg·m²]',   fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()   # no legend

    # ----------------- 3) Angular velocity + torque ----------------
    fig, ax_ω = plt.subplots(figsize=(8, 5))
    ax_τ = ax_ω.twinx()
    ax_ω.plot(t, results['ω_hist'][:, 0], 'b-', lw=LW, alpha=0.8)
    ax_ω.plot(t, results['ω_hist'][:, 1], 'g-', lw=LW, alpha=0.8)
    ax_ω.plot(t, results['ω_hist'][:, 2], 'r-', lw=LW, alpha=0.8)
    ax_τ.plot(t, results['τ_hist'][:, 0], 'b:', lw=LW, alpha=0.6)
    ax_τ.plot(t, results['τ_hist'][:, 1], 'g:', lw=LW, alpha=0.6)
    ax_τ.plot(t, results['τ_hist'][:, 2], 'r:', lw=LW, alpha=0.6)

    ax_ω.set_xlabel('Time [s]',                     fontsize=LABEL_FS)
    ax_ω.set_ylabel('Angular velocity [rad/s]',     fontsize=LABEL_FS, color='k')
    ax_τ.set_ylabel('Torque [N·m]',                 fontsize=LABEL_FS, color='k')
    ax_ω.tick_params(axis='both', labelsize=TICK_FS)
    ax_τ.tick_params(axis='y',   labelsize=TICK_FS)
    ax_ω.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()   # no legend

    # ----------------- 4) Relative error with σ envelope ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    J_true = np.diag(J_TRUE)
    rel_err = np.abs(J - J_true) / J_true * 100.0
    ax.semilogy(t, rel_err[:, 0], 'b-', lw=LW)
    ax.semilogy(t, rel_err[:, 1], 'g-', lw=LW)
    ax.semilogy(t, rel_err[:, 2], 'r-', lw=LW)

    σ_pct = σ / J_true * 100.0
    tiny = 1e-6
    ax.fill_between(t, np.maximum(σ_pct[:, 0], tiny), σ_pct[:, 0], color='b', alpha=0.10)
    ax.fill_between(t, np.maximum(σ_pct[:, 1], tiny), σ_pct[:, 1], color='g', alpha=0.10)
    ax.fill_between(t, np.maximum(σ_pct[:, 2], tiny), σ_pct[:, 2], color='r', alpha=0.10)

    ax.set_xlabel('Time [s]',            fontsize=LABEL_FS)
    ax.set_ylabel('Relative error [%]',  fontsize=LABEL_FS)
    ax.set_ylim(0, 250)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# =================
# Print summaries
# =================
def print_results(results, mode_name):
    """Print detailed results for the given simulation."""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {mode_name.upper()} EXCITATION")
    print(f"{'='*60}")

    J_true = np.diag(J_TRUE)
    errors = np.abs(results['final_J'] - J_true)
    rel_errors = errors / J_true * 100.0
    uncertainties = np.sqrt(np.diag(results['final_P'][7:10, 7:10]))

    print(f"\n=== FINAL ESTIMATION RESULTS ===")
    print(f"True inertia values: [{J_true[0]:.1f}, {J_true[1]:.1f}, {J_true[2]:.1f}]")
    print(f"Final estimates:     [{results['final_J'][0]:.2f}, {results['final_J'][1]:.2f}, {results['final_J'][2]:.2f}]")
    print("Final errors  (%):   " + ", ".join(f"{x:.2f}" for x in rel_errors))
    print("Final 1-σ     (abs): " + ", ".join(f"{x:.2f}" for x in uncertainties))

    print(f"\n=== DETAILED MEAN AND COVARIANCE VALUES ===")
    print("MEAN ESTIMATES (ukf.x):")
    print("  Quaternion (q0, q1, q2, q3):", results['final_x'][:4])
    print("  Angular velocity (ωx, ωy, ωz):", results['final_x'][4:7])
    print(f"  Inertia estimates (Jx,Jy,Jz): {results['final_J']}")

    print(f"\nCOVARIANCE (P[7:10, 7:10]):")
    final_cov = results['final_P'][7:10, 7:10]
    for i in range(3):
        print("  [" + " ".join(f"{final_cov[i, j]:8.2f}" for j in range(3)) + "]")

    print(f"\n=== EXCITATION METRICS ===")
    torque_rms = np.sqrt(np.mean(results['τ_hist']**2, axis=0))
    omega_rms  = np.sqrt(np.mean(results['ω_hist']**2, axis=0))
    print(f"Torque RMS values: [{torque_rms[0]:.3f}, {torque_rms[1]:.3f}, {torque_rms[2]:.3f}] N·m")
    print(f"Omega  RMS values: [{omega_rms[0]:.3f}, {omega_rms[1]:.3f}, {omega_rms[2]:.3f}] rad/s")

# ================
# Main execution
# ================
if __name__ == "__main__":
    print("Running UKF simulations for all three excitation modes...")

    print("\n--- Running FULL excitation simulation ---")
    results_full = run_simulation(excitation_mode="full")

    print("\n--- Running WINDOWED excitation simulation ---")
    results_windowed = run_simulation(excitation_mode="window")

    print("\n--- Running MULTI excitation simulation ---")
    results_multi = run_simulation(excitation_mode="multi")

    # Print results for all modes
    print_results(results_full, "Full")
    print_results(results_windowed, "Windowed")
    print_results(results_multi, "Multi")

    # Generate plots for all modes
    plot_results(results_full, "Full")
    plot_results(results_windowed, "Windowed")
    plot_results(results_multi, "Multi")
    plt.show()

    # Comparison summary (same style as EnKF: full vs windowed, and show improvement sign)
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    Jt = np.diag(J_TRUE)
    full_errors     = np.abs(results_full['final_J']     - Jt) / Jt * 100.0
    windowed_errors = np.abs(results_windowed['final_J'] - Jt) / Jt * 100.0
    print(f"Full excitation errors:     [{full_errors[0]:5.2f}, {full_errors[1]:5.2f}, {full_errors[2]:5.2f}] %")
    print(f"Windowed excitation errors: [{windowed_errors[0]:5.2f}, {windowed_errors[1]:5.2f}, {windowed_errors[2]:5.2f}] %")
    improvement = windowed_errors - full_errors
    print(f"Improvement (+ = better):   [{-improvement[0]:5.2f}, {-improvement[1]:5.2f}, {-improvement[2]:5.2f}] %")

    # Monte Carlo study
    mc_results = run_monte_carlo_study(
        n_samples=N_MONTE_CARLO,
        initial_mean=INITIAL_INERTIA_MEAN,
        initial_std=INITIAL_INERTIA_STD
    )
    print_monte_carlo_results(mc_results)

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")