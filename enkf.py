# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import time

# Quaternion Helpers (unchanged)
def quat_mult(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_from_omega(ω, dt):
    """Convert angular velocity to quaternion increment"""
    θ = np.linalg.norm(ω*dt)
    if θ < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = ω / np.linalg.norm(ω)
    half = 0.5*θ
    return np.hstack((np.cos(half), axis*np.sin(half)))

def quat_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

def normalize_quaternion(q):
    """Normalize quaternion ensuring w >= 0 for uniqueness"""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1., 0., 0., 0.])
    q_norm = q / norm
    if q_norm[0] < 0:
        q_norm = -q_norm
    return q_norm

def add_quaternion_noise(q_true, sigma):
    """Add noise to quaternion using proper SO(3) perturbation"""
    delta_axis = np.random.normal(0, sigma, 3)
    delta_angle = np.linalg.norm(delta_axis)
    if delta_angle < 1e-8:
        return q_true
    delta_axis = delta_axis / delta_angle
    cos_half = np.cos(delta_angle / 2)
    sin_half = np.sin(delta_angle / 2)
    delta_q = np.array([cos_half, sin_half * delta_axis[0],
                       sin_half * delta_axis[1], sin_half * delta_axis[2]])
    q_noisy = quat_mult(q_true, delta_q)
    return normalize_quaternion(q_noisy)

# Scenario Setup
np.random.seed(42)
J_TRUE = np.diag([100, 80, 70])  # True inertia matrix (kg·m²)
dt, T_RUN = 0.01, 400.0
steps = int(T_RUN/dt)

# Monte Carlo settings
INITIAL_INERTIA_MEAN = np.array([140.0, 20.0, 36.06])
INITIAL_INERTIA_STD  = np.array([10.0, 10.0, 10.0])
N_MONTE_CARLO = 50

def skew(w):
    """Skew symmetric matrix"""
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

# FIXED EnKF class
class SpacecraftEnKF:
    def __init__(self, x_init, P_init, R, Q, N_ensemble, dt):
        self.x = x_init.copy()
        self.P = P_init.copy()
        self.R = R.copy()
        self.Q = Q.copy()
        self.N = N_ensemble
        self.dt = dt
        self.dim_x = len(x_init)
        self.dim_z = R.shape[0]
        
        # Initialize ensemble with proper distribution
        self.ensemble = np.random.multivariate_normal(self.x, self.P, self.N)
        for i in range(self.N):
            self.ensemble[i, :4] = normalize_quaternion(self.ensemble[i, :4])
            self.ensemble[i, 7:] = np.maximum(self.ensemble[i, 7:], 10.0)  # Higher minimum

        # --- NEW: ensure initial inertia mean equals x_init[7:] exactly ---
        ens_mean = np.mean(self.ensemble, axis=0)
        self.ensemble[:, 7:] += (self.x[7:] - ens_mean[7:])
        # -------------------------------------------------------------------

        self.S = None
        self.y = None
        
        # Conservative inflation
        self.base_inflation = 1.01
        self.additive_inflation = 1e-8
        
        # Observability gating
        self.min_torque_for_inertia_update = 0.5
        
        # Regularization parameters
        self.cov_regularization = 1e-10
        self.max_condition_number = 1e12

        # --- NEW: block inertia update on the very first update call ---
        self.first_update_done = False
        # ----------------------------------------------------------------
        
    def regularize_covariance(self, cov_matrix):
        """Regularize covariance matrix for numerical stability"""
        cov_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * self.cov_regularization
        try:
            cond_num = np.linalg.cond(cov_reg)
            if cond_num > self.max_condition_number:
                eigenvals, eigenvecs = np.linalg.eigh(cov_reg)
                eigenvals = np.maximum(eigenvals, np.max(eigenvals) / self.max_condition_number)
                cov_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except:
            cov_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        return cov_reg

    def predict(self, torque):
        """Enhanced prediction step with better numerical stability"""
        for i in range(self.N):
            try:
                noise = np.random.multivariate_normal(np.zeros(self.dim_x), self.Q)
                new_state = self.fx(self.ensemble[i], torque) + noise
                if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
                    continue
                self.ensemble[i] = new_state
                self.ensemble[i, :4] = normalize_quaternion(self.ensemble[i, :4])
                self.ensemble[i, 7:] = np.maximum(self.ensemble[i, 7:], 10.0)
            except Exception:
                continue
        
        self.x = np.mean(self.ensemble, axis=0)
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            print("Warning: NaN detected in state mean, skipping update")
            return
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.x[7:] = np.maximum(self.x[7:], 10.0)
        
        deviations = self.ensemble - self.x
        if np.any(np.isnan(deviations)) or np.any(np.isinf(deviations)):
            print("Warning: NaN detected in ensemble deviations")
            return
        
        self.P = np.cov(deviations.T)
        self.P = self.regularize_covariance(self.P)
        self.P *= self.base_inflation
        self.P += np.eye(self.dim_x) * self.additive_inflation
        
        ensemble_spread = np.mean(np.linalg.norm(deviations, axis=1))
        if ensemble_spread < 1e-8:
            try:
                L = np.linalg.cholesky(self.P)
                for i in range(self.N):
                    random_vec = np.random.randn(self.dim_x) * 0.1
                    self.ensemble[i] = self.x + L @ random_vec
                    self.ensemble[i, :4] = normalize_quaternion(self.ensemble[i, :4])
                    self.ensemble[i, 7:] = np.maximum(self.ensemble[i, 7:], 10.0)
            except:
                for i in range(self.N):
                    perturbation = np.random.randn(self.dim_x) * 0.01
                    self.ensemble[i] = self.x + perturbation
                    self.ensemble[i, :4] = normalize_quaternion(self.ensemble[i, :4])
                    self.ensemble[i, 7:] = np.maximum(self.ensemble[i, 7:], 10.0)
    
    def update(self, z, torque):
        """Enhanced update step with improved numerical stability"""
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print("Warning: NaN detected in measurement, skipping update")
            return
        
        torque_magnitude = np.linalg.norm(torque)
        update_inertia = torque_magnitude > self.min_torque_for_inertia_update

        # --- NEW: never update inertia on the very first update call ---
        if not self.first_update_done:
            update_inertia = False
        # ----------------------------------------------------------------
        
        Z_pred = np.zeros((self.N, 7))
        for i in range(self.N):
            try:
                Z_pred[i] = self.hx(self.ensemble[i])
            except:
                if i > 0:
                    Z_pred[i] = Z_pred[i-1]
                else:
                    Z_pred[i] = z
        
        z_pred = np.mean(Z_pred, axis=0)
        if np.any(np.isnan(z_pred)) or np.any(np.isinf(z_pred)):
            print("Warning: NaN detected in predicted measurements, skipping update")
            return
        
        self.y = z - z_pred
        
        q_meas = z[:4] / np.linalg.norm(z[:4])
        q_pred = z_pred[:4] / np.linalg.norm(z_pred[:4])
        q_error = q_meas - q_pred
        innovation_7d = np.hstack([q_error, self.y[4:7]])
        
        S_7d = np.cov(Z_pred.T)
        S_7d += np.diag([0.005**2]*4 + [0.005**2]*3)
        S_7d = self.regularize_covariance(S_7d)
        
        deviations_x = self.ensemble - self.x
        deviations_z = Z_pred - z_pred
        
        Pxz = np.zeros((self.dim_x, 7))
        for i in range(self.N):
            Pxz += np.outer(deviations_x[i], deviations_z[i])
        Pxz /= max(self.N - 1, 1)
        
        try:
            K = Pxz @ np.linalg.inv(S_7d)
        except:
            try:
                K = Pxz @ np.linalg.pinv(S_7d)
            except:
                print("Warning: Failed to compute Kalman gain, skipping update")
                return
        
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            print("Warning: NaN detected in Kalman gain, skipping update")
            return
        
        if not update_inertia:
            K[7:, :] = 0.0
        
        for i in range(self.N):
            try:
                innovation_i = z - Z_pred[i]
                update_step = K @ innovation_i
                if np.any(np.isnan(update_step)) or np.any(np.isinf(update_step)):
                    continue
                self.ensemble[i] += update_step
                self.ensemble[i, :4] = normalize_quaternion(self.ensemble[i, :4])
                self.ensemble[i, 7:] = np.maximum(self.ensemble[i, 7:], 10.0)
            except:
                continue
        
        self.x = np.mean(self.ensemble, axis=0)
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            print("Warning: NaN detected after update, reverting")
            return
            
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.x[7:] = np.maximum(self.x[7:], 10.0)
        
        deviations = self.ensemble - self.x
        self.P = np.cov(deviations.T)
        self.P = self.regularize_covariance(self.P)
        
        self.S = S_7d
        self.y = innovation_7d

        # --- NEW: allow inertia updates from now on ---
        self.first_update_done = True
        # ------------------------------------------------
    
    def fx(self, x, torque):
        """State transition function with numerical stability"""
        try:
            q = x[:4]
            ω = x[4:7]
            J_diag = x[7:]
            J_diag = np.maximum(J_diag, 10.0)
            J = np.diag(J_diag)
            
            if np.linalg.norm(ω) > 100.0:
                ω = ω / np.linalg.norm(ω) * 100.0
            
            try:
                J_inv = np.linalg.inv(J)
                ω̇ = J_inv @ (torque - skew(ω) @ (J @ ω))
                if np.linalg.norm(ω̇) > 1000.0:
                    ω̇ = ω̇ / np.linalg.norm(ω̇) * 1000.0
                ω_new = ω + ω̇ * self.dt
            except:
                ω_new = ω + 0.001 * torque / np.mean(J_diag)
            
            q_new = quat_mult(q, quat_from_omega(ω_new, self.dt))
            q_new = normalize_quaternion(q_new)
            
            result = np.hstack((q_new, ω_new, J_diag))
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return x
            return result
            
        except Exception:
            return x
    
    def hx(self, x):
        """Measurement function"""
        return np.hstack((x[:4], x[4:7]))

# Main Simulation - FIXED
def run_simulation(excitation_mode="full", initial_inertia=None, verbose=True):
    """Run FIXED EnKF simulation"""
    if verbose:
        print(f"Starting FIXED EnKF simulation with {excitation_mode} excitation mode...")
    
    # Forward Simulation
    ω = np.array([0.1, 0.1, 0.1])
    q_body_cam = np.array([1., 0., 0., 0.])
    ω_hist = np.empty((steps, 3))
    q_hist = np.empty((steps, 4))
    τ_hist = np.empty((steps, 3))
    t_hist = np.linspace(0.0, T_RUN, steps)

    if verbose:
        print("Running forward simulation...")
    for k, t in enumerate(t_hist):
        τ = torque_profile(t, mode=excitation_mode)
        ω̇ = np.linalg.inv(J_TRUE) @ (τ - skew(ω) @ (J_TRUE @ ω))
        ω += ω̇ * dt
        q_body_cam = quat_mult(q_body_cam, quat_from_omega(ω, dt))
        q_body_cam /= np.linalg.norm(q_body_cam)
        
        ω_hist[k] = ω
        q_hist[k] = q_body_cam
        τ_hist[k] = τ

    # Synthetic Measurements
    σ_gyro = 0.005   
    σ_quat = 0.005
    
    ω_meas = ω_hist + np.random.normal(0.0, σ_gyro, ω_hist.shape)
    
    q_meas = np.zeros_like(q_hist)
    for k in range(steps):
        q_meas[k] = add_quaternion_noise(q_hist[k], σ_quat)
    
    z_meas = np.hstack((q_meas, ω_meas))

    # EnKF Setup - FIXED parameters
    dim_x, dim_z = 10, 7
    N_ensemble = 100  # Smaller for stability

    # Initial state
    x_init = np.zeros(dim_x)
    x_init[:4] = q_meas[0]
    x_init[4:7] = ω_meas[0]
    if initial_inertia is None:
        x_init[7:] = np.array([140.0,20.0,36.06])  # <-- requested initial inertia
    else:
        x_init[7:] = np.maximum(np.array(initial_inertia, dtype=float), 10.0)

    # Initial covariance - more conservative
    P_init = np.eye(dim_x)
    P_init[:4, :4] *= 0.001
    P_init[4:7, 4:7] *= 0.01
    P_init[7:10, 7:10] = np.diag([1700.0, 20.0, 120])

    # Measurement noise covariance
    R = np.diag([σ_quat**2]*4 + [σ_gyro**2]*3)

    # FIXED Process noise covariance - more conservative
    Q_quat = 1e-7      
    Q_omega = 1e-7     
    Q_inertia = 1e-7  
    Q = np.diag([Q_quat]*4 + [Q_omega]*3 + [Q_inertia]*3)

    # Create EnKF
    enkf = SpacecraftEnKF(x_init, P_init, R, Q, N_ensemble, dt)

    # Estimation Loop
    if verbose:
        print("Running FIXED EnKF estimation...")
    J_hist = np.empty((steps, 3))
    trace_P = np.empty(steps)
    nis_hist = np.empty(steps)
    P_diag_inertia = np.empty((steps, 3))

    for k, t in enumerate(t_hist):
        τ = τ_hist[k]
        
        try:
            enkf.predict(torque=τ)
            enkf.update(z_meas[k], torque=τ)
            
            # Store results
            J_hist[k] = enkf.x[7:]
            trace_P[k] = np.trace(enkf.P[7:10, 7:10])
            P_diag_inertia[k] = np.diag(enkf.P[7:10, 7:10])
            
            # NIS calculation
            if hasattr(enkf, 'S') and hasattr(enkf, 'y') and enkf.S is not None:
                try:
                    S_inv = np.linalg.inv(enkf.S)
                    nis_hist[k] = enkf.y.T @ S_inv @ enkf.y
                except:
                    nis_hist[k] = 0
            else:
                nis_hist[k] = 0
                
        except Exception as e:
            if verbose:
                print(f"Error at step {k}: {e}")
            if k > 0:
                J_hist[k] = J_hist[k-1]
                trace_P[k] = trace_P[k-1]
                P_diag_inertia[k] = P_diag_inertia[k-1]
                nis_hist[k] = 0
        
        # Progress indicator
        if verbose and k % 5000 == 0:
            current_J = J_hist[k] if not np.any(np.isnan(J_hist[k])) else "NaN detected"
            print(f"  Progress: {k/steps*100:.1f}% - Current J: {current_J}")

    if verbose:
        print("FIXED EnKF simulation completed!")
    
    # Calculate final errors
    if verbose:
        if not np.any(np.isnan(J_hist[-1])):
            J_error = np.abs(J_hist[-1] - np.diag(J_TRUE))
            print(f"Final inertia estimates: {J_hist[-1]}")
            print(f"True inertia values: {np.diag(J_TRUE)}")
            print(f"Final errors: {J_error}")
            rel = 100 * J_error / np.diag(J_TRUE)
            print("Final relative errors: [{:.2f}, {:.2f}, {:.2f}] %".format(*rel))
            print(f"Final uncertainties: {np.sqrt(np.diag(enkf.P[7:10, 7:10]))}")
        else:
            print("Warning: Final estimates contain NaN values")

    return dict(
        t_hist=t_hist, J_hist=J_hist, nis_hist=nis_hist, 
        trace_P=trace_P, P_diag_inertia=P_diag_inertia,
        q_hist=q_hist, ω_hist=ω_hist, τ_hist=τ_hist, 
        final_J=J_hist[-1], final_P=enkf.P.copy(),
        ω_meas=ω_meas, q_meas=q_meas, final_x=enkf.x.copy()
    )

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
    ax.set_ylim(0, 250)   # <-- Added line: fixes y-axis between 0 and 250
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

def print_results(results, mode_name):
    """Enhanced results printing"""
    print(f"\n{'='*60}")
    print(f"EnKF {mode_name.upper()} EXCITATION RESULTS")
    print(f"{'='*60}")
    
    J_true = np.diag(J_TRUE)
    errors = np.abs(results['final_J'] - J_true)
    rel_errors = errors / J_true * 100
    uncertainties = np.sqrt(np.diag(results['final_P'][7:10, 7:10]))
    
    print(f"True inertia values:     [{J_true[0]:6.1f}, {J_true[1]:6.1f}, {J_true[2]:6.1f}] kg·m²")
    print(f"Final estimates:         [{results['final_J'][0]:6.1f}, {results['final_J'][1]:6.1f}, {results['final_J'][2]:6.1f}] kg·m²")
    print(f"Absolute errors:         [{errors[0]:6.2f}, {errors[1]:6.2f}, {errors[2]:6.2f}] kg·m²")
    print(f"Relative errors:         [{rel_errors[0]:6.2f}, {rel_errors[1]:6.2f}, {rel_errors[2]:6.2f}] %")
    print(f"Final uncertainties (1σ): [{uncertainties[0]:6.2f}, {uncertainties[1]:6.2f}, {uncertainties[2]:6.2f}] kg·m²")
    
    print(f"\nCONVERGENCE ASSESSMENT:")
    converged = rel_errors < 5.0  # 5% threshold
    print(f"Converged (< 5% error):   [{str(converged[0]):>5}, {str(converged[1]):>5}, {str(converged[2]):>5}]")
    
    torque_rms = np.sqrt(np.mean(results['τ_hist']**2, axis=0))
    omega_rms = np.sqrt(np.mean(results['ω_hist']**2, axis=0))
    print(f"\nEXCITATION METRICS:")
    print(f"Torque RMS values:       [{torque_rms[0]:6.3f}, {torque_rms[1]:6.3f}, {torque_rms[2]:6.3f}] N·m")
    print(f"Angular velocity RMS:    [{omega_rms[0]:6.3f}, {omega_rms[1]:6.3f}, {omega_rms[2]:6.3f}] rad/s")

if __name__ == "__main__":
    # Run simulations
    print("Starting EnKF simulations...")
    try:
        results_full = run_simulation(excitation_mode="full")
        results_windowed = run_simulation(excitation_mode="window")
        results_multi = run_simulation(excitation_mode="multi")   # <-- NEW: multi case

        # Display results
        print_results(results_full, "FULL")
        print_results(results_windowed, "WINDOWED")
        print_results(results_multi, "MULTI")                     # <-- NEW

        # Generate plots
        plot_results(results_full, "Full")
        plot_results(results_windowed, "Windowed")
        plot_results(results_multi, "Multi")                      # <-- NEW
        plt.show()
        
        # Comparison summary (unchanged; still compares full vs window)
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        full_errors = np.abs(results_full['final_J'] - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100
        windowed_errors = np.abs(results_windowed['final_J'] - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100
        multi_errors = np.abs(results_multi['final_J'] - np.diag(J_TRUE)) / np.diag(J_TRUE) * 100
        
        print(f"Full excitation errors:     [{full_errors[0]:5.2f}, {full_errors[1]:5.2f}, {full_errors[2]:5.2f}] %")
        print(f"Windowed excitation errors: [{windowed_errors[0]:5.2f}, {windowed_errors[1]:5.2f}, {windowed_errors[2]:5.2f}] %")
        print(f"Persistent excitation errors:[{multi_errors[0]:5.2f}, {multi_errors[1]:5.2f}, {multi_errors[2]:5.2f}] %")
        
        improvement = windowed_errors - full_errors
        print(f"Improvement (+ = better):   [{-improvement[0]:5.2f}, {-improvement[1]:5.2f}, {-improvement[2]:5.2f}] %")

        # Monte Carlo study
        mc_results = run_monte_carlo_study(
            n_samples=N_MONTE_CARLO,
            initial_mean=INITIAL_INERTIA_MEAN,
            initial_std=INITIAL_INERTIA_STD
        )
        print_monte_carlo_results(mc_results)
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()