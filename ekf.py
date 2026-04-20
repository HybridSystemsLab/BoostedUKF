# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# Quaternion / SO(3) utils
# =========================
def quat_mult(q1, q2):
    """Multiply two quaternions: q = q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_from_omega(ω, dt):
    """Convert body angular velocity to a small-rotation quaternion increment over dt."""
    θ = np.linalg.norm(ω*dt)
    if θ < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = ω / np.linalg.norm(ω)
    half = 0.5 * θ
    return np.hstack((np.cos(half), axis*np.sin(half)))

def normalize_quaternion(q):
    """Normalize quaternion; enforce w >= 0 for uniqueness."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1., 0., 0., 0.])
    qn = q / n
    if qn[0] < 0:
        qn = -qn
    return qn

def add_quaternion_noise(q_true, sigma):
    """
    Add small SO(3) noise by left-multiplying with a random-axis small-angle rotation.
    sigma is the std of each axis component before conversion to angle.
    """
    delta_axis = np.random.normal(0.0, sigma, 3)
    delta_angle = np.linalg.norm(delta_axis)
    if delta_angle < 1e-8:
        return normalize_quaternion(q_true)
    axis = delta_axis / delta_angle
    cos_half = np.cos(delta_angle / 2.0)
    sin_half = np.sin(delta_angle / 2.0)
    dq = np.array([cos_half, sin_half*axis[0], sin_half*axis[1], sin_half*axis[2]])
    return normalize_quaternion(quat_mult(q_true, dq))

# =================
# Global scenario
# =================
np.random.seed(42)
J_TRUE = np.diag([100, 80, 70])  # True inertia matrix (kg·m²)
dt, T_RUN = 0.01, 400.0
t_hist = np.arange(0.0, T_RUN + 1e-12, dt)  # exact dt steps; includes T_RUN
steps = len(t_hist)

# Monte Carlo settings
INITIAL_INERTIA_MEAN = np.array([140.0, 20.0, 36.06])
INITIAL_INERTIA_STD  = np.array([10.0, 10.0, 10.0])   # change these if you want a different spread
N_MONTE_CARLO = 50

def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def torque_profile(t, mode="full"):
    """Torque profile with full, windowed, or multi-pulse (persistent) excitation."""

    def base_signal(tt):
        """Underlying torque signal."""
        return np.array([
            2.0*np.sin(0.1*tt) + 4.5*np.cos(0.3*tt) + 2.0*np.sin(0.7*tt) + 2.0*np.sin(1.5*tt),
            3.6*np.cos(0.15*tt) + 4.0*np.sin(0.4*tt) + 3.4*np.cos(0.8*tt) + 3.8*np.cos(1.8*tt),
            4.4*np.sin(0.12*tt) + 3.1*np.cos(0.5*tt) + 2.0*np.sin(0.9*tt) + 2.5*np.sin(2.0*tt)
        ])

    if mode == "full":
        # Continuous excitation throughout
        return base_signal(t)

    elif mode == "window":
        # Single short excitation bursts
        in_pulse = ((200.0 <= t < 201.0) or
                    (250.0 <= t < 251.0) or
                    (300.0 <= t < 301.0))
        if not in_pulse:
            return np.zeros(3)
        return base_signal(t)

    elif mode == "multi":
        # Persistent excitation — same amplitude, repeated 20 times
        # Start every 25s from 50s to 525s, each lasting 1s
        pulse_intervals = [(50 + 25*i, 51 + 25*i) for i in range(20)]
        in_pulse = any(a <= t < b for a, b in pulse_intervals)
        if not in_pulse:
            return np.zeros(3)
        return base_signal(t)

    else:
        raise ValueError("Mode must be 'full', 'window', or 'multi'.")

# =========================
# EKF with stability guards
# =========================
class ExtendedKalmanFilter:
    def __init__(self, dim_x, dim_z, dt):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt

        self.x = np.zeros(dim_x)     # State estimate
        self.P = np.eye(dim_x)       # Covariance
        self.Q = np.eye(dim_x)       # Process noise
        self.R = np.eye(dim_z)       # Measurement noise

        self.F = np.eye(dim_x)       # State Jacobian
        self.H = np.zeros((dim_z, dim_x))  # Meas. Jacobian

        # Analysis buffers
        self.y = np.zeros(dim_z)     # Innovation
        self.S = np.eye(dim_z)       # Innovation covariance
        self.K = np.zeros((dim_x, dim_z))  # Kalman gain

        # Stability knobs
        self._first_update_done = False
        self.min_torque_for_inertia_update = 0.5
        self.cov_regularization = 1e-10
        self.max_condition_number = 1e12

    def _regularize(self, M):
        """Regularize symmetric covariance-like matrix M."""
        M_reg = M + np.eye(M.shape[0]) * self.cov_regularization
        try:
            cond = np.linalg.cond(M_reg)
            if cond > self.max_condition_number:
                w, V = np.linalg.eigh(M_reg)
                w = np.maximum(w, np.max(w) / self.max_condition_number)
                M_reg = V @ np.diag(w) @ V.T
        except Exception:
            M_reg = M + np.eye(M.shape[0]) * 1e-6
        return M_reg

    def predict(self, fx, F_jac, **kwargs):
        """Nonlinear predict step: x_k|k-1, P_k|k-1."""
        self.x = fx(self.x, self.dt, **kwargs)
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.x[7:] = np.maximum(self.x[7:], 10.0)

        self.F = F_jac(self.x, self.dt, **kwargs)
        P_pred = self.F @ self.P @ self.F.T + self.Q
        self.P = self._regularize(P_pred)

    def update(self, z, hx, H_jac, torque=None):
        """Nonlinear update with inertia gating and Joseph-form covariance update."""
        self.H = H_jac(self.x)
        h_x = hx(self.x)
        # Normalize quaternion parts to be safe
        qz = z[:4]; qz = qz / (np.linalg.norm(qz) + 1e-12)
        qh = h_x[:4]; qh = qh / (np.linalg.norm(qh) + 1e-12)
        z_safe = np.hstack((qz, z[4:7]))
        h_safe = np.hstack((qh, h_x[4:7]))

        self.y = z_safe - h_safe
        S = self.H @ self.P @ self.H.T + self.R
        self.S = self._regularize(S)

        try:
            Sinv = np.linalg.inv(self.S)
        except Exception:
            Sinv = np.linalg.pinv(self.S)

        self.K = self.P @ self.H.T @ Sinv

        # Gate inertia updates (like EnKF): first update frozen, and low ||τ|| frozen
        freeze_inertia = (not self._first_update_done) or (
            torque is not None and np.linalg.norm(torque) <= self.min_torque_for_inertia_update
        )
        if freeze_inertia:
            self.K[7:10, :] = 0.0

        # State update
        delta = self.K @ self.y
        self.x = self.x + delta
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.x[7:] = np.maximum(self.x[7:], 10.0)

        # Covariance (Joseph form)
        I = np.eye(self.dim_x)
        IKH = I - self.K @ self.H
        P_new = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T
        self.P = self._regularize(P_new)

        self._first_update_done = True

# ========================
# EKF model (fx, jacobians)
# ========================
def state_transition(x, dt, torque):
    """Nonlinear rigid-body dynamics + quaternion integration; mirrors EnKF's fx (with clamps)."""
    q = x[:4]
    ω = x[4:7]
    J_diag = np.maximum(x[7:], 10.0)
    J = np.diag(J_diag)

    # Clamp angular rate magnitude to avoid explosions
    if np.linalg.norm(ω) > 100.0:
        ω = ω / np.linalg.norm(ω) * 100.0

    try:
        J_inv = np.linalg.inv(J)
        ωdot = J_inv @ (torque - skew(ω) @ (J @ ω))
        if np.linalg.norm(ωdot) > 1000.0:
            ωdot = ωdot / np.linalg.norm(ωdot) * 1000.0
        ω_new = ω + ωdot * dt
    except Exception:
        ω_new = ω + 0.001 * torque / np.mean(J_diag)

    q_new = quat_mult(q, quat_from_omega(ω_new, dt))
    q_new = normalize_quaternion(q_new)
    return np.hstack((q_new, ω_new, J_diag))

def state_jacobian(x, dt, torque):
    """
    Finite-difference Jacobian of state transition.
    Note: uses a small eps and reuses state_transition's clamps and renormalization.
    """
    F = np.eye(10)
    eps = 1e-8
    x0 = np.copy(x)
    x0[:4] = normalize_quaternion(x0[:4])
    x0[7:] = np.maximum(x0[7:], 10.0)
    f0 = state_transition(x0, dt, torque)
    for i in range(10):
        xp = np.copy(x0)
        xp[i] += eps
        fp = state_transition(xp, dt, torque)
        F[:, i] = (fp - f0) / eps
    return F

def measurement_function(x):
    """Direct measurement of quaternion (normalized) and omega."""
    q = normalize_quaternion(x[:4])
    return np.hstack((q, x[4:7]))

def measurement_jacobian(x):
    """Simple identity blocks for [q, ω]."""
    H = np.zeros((7, 10))
    H[:4, :4] = np.eye(4)
    H[4:7, 4:7] = np.eye(3)
    return H

# ==================
# Main EKF pipeline
# ==================
def run_ekf_simulation(excitation_mode="full", initial_inertia=None, verbose=True):
    """Run EKF simulation for a given excitation mode (full/window/multi)."""
    if verbose:
        print(f"Starting FIXED EKF simulation with {excitation_mode} excitation mode...")

    # -------- Forward (truth) simulation --------
    ω = np.array([0.1, 0.1, 0.1])
    q_body_cam = np.array([1., 0., 0., 0.])
    ω_hist = np.empty((steps, 3))
    q_hist = np.empty((steps, 4))
    τ_hist = np.empty((steps, 3))

    for k, t in enumerate(t_hist):
        τ = torque_profile(t, mode=excitation_mode)
        ωdot = np.linalg.inv(J_TRUE) @ (τ - skew(ω) @ (J_TRUE @ ω))
        ω = ω + ωdot * dt
        q_body_cam = quat_mult(q_body_cam, quat_from_omega(ω, dt))
        q_body_cam = normalize_quaternion(q_body_cam)

        ω_hist[k] = ω
        q_hist[k] = q_body_cam
        τ_hist[k] = τ

    # -------- Synthetic measurements (same noise as EnKF) --------
    σ_gyro = 0.005
    σ_quat = 0.005

    ω_meas = ω_hist + np.random.normal(0.0, σ_gyro, ω_hist.shape)

    q_meas = np.zeros_like(q_hist)
    for k in range(steps):
        q_meas[k] = add_quaternion_noise(q_hist[k], σ_quat)

    z_meas = np.hstack((q_meas, ω_meas))

    # -------- EKF setup (same params as EnKF) --------
    dim_x, dim_z = 10, 7
    ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt)

    # Initial state
    ekf.x[:4] = q_meas[0]
    ekf.x[4:7] = ω_meas[0]
    if initial_inertia is None:
        ekf.x[7:] = np.array([140.0, 20.0, 36.06])
    else:
        ekf.x[7:] = np.maximum(np.array(initial_inertia, dtype=float), 10.0)

    # Initial covariance (conservative, mirrors EnKF)
    ekf.P = np.eye(dim_x)
    ekf.P[:4, :4] *= 0.001
    ekf.P[4:7, 4:7] *= 0.01
    ekf.P[7:10, 7:10] = np.diag([1700.0, 20.0, 120])

    # Measurement noise covariance
    ekf.R = np.diag([σ_quat**2]*4 + [σ_gyro**2]*3)

    # Process noise covariance (small, same values)
    Q_quat = 1e-7
    Q_omega = 1e-7
    Q_inertia = 1e-7
    ekf.Q = np.diag([Q_quat]*4 + [Q_omega]*3 + [Q_inertia]*3)

    # -------- Estimation loop --------
    J_hist = np.empty((steps, 3))
    trace_P = np.empty(steps)
    nis_hist = np.empty(steps)
    P_diag_inertia = np.empty((steps, 3))

    for k, t in enumerate(t_hist):
        τ = τ_hist[k]

        # Predict & update
        ekf.predict(state_transition, state_jacobian, torque=τ)
        ekf.update(z_meas[k], measurement_function, measurement_jacobian, torque=τ)

        # Safety: renormalize and keep inertia positive
        ekf.x[:4] = normalize_quaternion(ekf.x[:4])
        ekf.x[7:] = np.maximum(ekf.x[7:], 10.0)

        # Logs
        J_hist[k] = ekf.x[7:]
        trace_P[k] = np.trace(ekf.P[7:10, 7:10])
        P_diag_inertia[k] = np.diag(ekf.P[7:10, 7:10])

        try:
            S_inv = np.linalg.inv(ekf.S)
            nis_hist[k] = ekf.y.T @ S_inv @ ekf.y
        except Exception:
            nis_hist[k] = 0.0

        if verbose and (k % 5000 == 0):
            curJ = J_hist[k] if np.all(np.isfinite(J_hist[k])) else "NaN"
            print(f"  Progress: {k/steps*100:.1f}% - Current J: {curJ}")

    if verbose:
        print("FIXED EKF simulation completed!")
    return dict(
        t_hist=t_hist, J_hist=J_hist, nis_hist=nis_hist,
        trace_P=trace_P, P_diag_inertia=P_diag_inertia,
        q_hist=q_hist, ω_hist=ω_hist, τ_hist=τ_hist,
        final_J=J_hist[-1], final_P=ekf.P.copy(),
        ω_meas=ω_meas, q_meas=q_meas, final_x=ekf.x.copy()
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
    samples = np.maximum(samples, 10.0)

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
            results = run_ekf_simulation(
                excitation_mode=case,
                initial_inertia=samples[s],
                verbose=False
            )

            mc_results[case]["final_J"][s, :] = results["final_J"]
            mc_results[case]["rel_errors"][s, :] = compute_relative_errors(results["final_J"])

            completed_runs += 1

            # Print progress every 10 samples or at the end
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
    plt.show()

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
    plt.show()

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
    plt.show()

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

# ================
# Print summaries
# ================
def print_results(results, mode_name):
    print(f"\n{'='*60}")
    print(f"EKF {mode_name.upper()} EXCITATION RESULTS")
    print(f"{'='*60}")
    J_true = np.diag(J_TRUE)
    errors = np.abs(results['final_J'] - J_true)
    rel_errors = errors / J_true * 100
    uncertainties = np.sqrt(np.maximum(np.diag(results['final_P'][7:10, 7:10]), 0.0))

    print(f"True inertia values:      [{J_true[0]:6.1f}, {J_true[1]:6.1f}, {J_true[2]:6.1f}] kg·m²")
    print(f"Final estimates:          [{results['final_J'][0]:6.1f}, {results['final_J'][1]:6.1f}, {results['final_J'][2]:6.1f}] kg·m²")
    print(f"Absolute errors:          [{errors[0]:6.2f}, {errors[1]:6.2f}, {errors[2]:6.2f}] kg·m²")
    print(f"Relative errors:          [{rel_errors[0]:6.2f}, {rel_errors[1]:6.2f}, {rel_errors[2]:6.2f}] %")
    print(f"Final uncertainties (1σ): [{uncertainties[0]:6.2f}, {uncertainties[1]:6.2f}, {uncertainties[2]:6.2f}] kg·m²")

    print(f"\nCONVERGENCE ASSESSMENT:")
    converged = rel_errors < 5.0
    print(f"Converged (< 5% error):   [{str(converged[0]):>5}, {str(converged[1]):>5}, {str(converged[2]):>5}]")

    torque_rms = np.sqrt(np.mean(results['τ_hist']**2, axis=0))
    omega_rms  = np.sqrt(np.mean(results['ω_hist']**2, axis=0))
    print(f"\nEXCITATION METRICS:")
    print(f"Torque RMS values:        [{torque_rms[0]:6.3f}, {torque_rms[1]:6.3f}, {torque_rms[2]:6.3f}] N·m")
    print(f"Angular velocity RMS:     [{omega_rms[0]:6.3f}, {omega_rms[1]:6.3f}, {omega_rms[2]:6.3f}] rad/s")

# ================
# Main execution
# ================
if __name__ == "__main__":
    print("Starting EKF simulations...")
    try:
        # ---------------- Single-run results (original behavior retained) ----------------
        results_full    = run_ekf_simulation(excitation_mode="full")
        results_window  = run_ekf_simulation(excitation_mode="window")
        results_multi   = run_ekf_simulation(excitation_mode="multi")

        print_results(results_full,   "FULL")
        print_results(results_window, "WINDOWED")
        print_results(results_multi,  "MULTI")

        plot_results(results_full,   "Full")
        plot_results(results_window, "Windowed")
        plot_results(results_multi,  "Multi")
        plt.show()

        # Original comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        full_errors   = np.abs(results_full['final_J']   - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100.0
        window_errors = np.abs(results_window['final_J'] - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100.0
        multi_errors  = np.abs(results_multi['final_J']  - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100.0

        print(f"Full excitation errors:      [{full_errors[0]:5.2f}, {full_errors[1]:5.2f}, {full_errors[2]:5.2f}] %")
        print(f"Windowed excitation errors:  [{window_errors[0]:5.2f}, {window_errors[1]:5.2f}, {window_errors[2]:5.2f}] %")
        print(f"Persistent excitation errors:[{multi_errors[0]:5.2f}, {multi_errors[1]:5.2f}, {multi_errors[2]:5.2f}] %")

        # ---------------- Monte Carlo study (new addition) ----------------
        mc_results = run_monte_carlo_study(
            n_samples=N_MONTE_CARLO,
            initial_mean=INITIAL_INERTIA_MEAN,
            initial_std=INITIAL_INERTIA_STD
        )
        print_monte_carlo_results(mc_results)

        print("\nEKF simulation completed successfully!")
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()