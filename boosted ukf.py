# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
import pandas as pd
import scipy.linalg

# --------------------------
# Enhanced robust SPD projector with better regularization
# --------------------------
def _spd(A, floor=1e-8, max_cond=1e12):
    """Project A to the nearest SPD by eigenvalue flooring with condition number control."""
    A = 0.5 * (A + A.T)
    try:
        vals, vecs = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(A)
        vals = s
        vecs = U

    vals = np.maximum(vals, floor)
    max_val = np.max(vals)
    min_val = max_val / max_cond
    vals = np.maximum(vals, min_val)

    A_spd = (vecs * vals) @ vecs.T
    A_spd = 0.5 * (A_spd + A_spd.T)
    A_spd += floor * np.eye(A.shape[0])
    return A_spd


# --------------------------
# Enhanced safe Cholesky decomposition with multiple fallbacks
# --------------------------
def safe_cholesky(A, regularization=1e-8, max_attempts=15):
    """Compute Cholesky decomposition with automatic regularization and fallbacks."""
    A_work = A.copy()
    for attempt in range(max_attempts):
        try:
            L = np.linalg.cholesky(A_work)
            if np.allclose(L @ L.T, A_work, rtol=1e-10):
                return L
        except np.linalg.LinAlgError:
            pass
        reg = regularization * (2 ** attempt)
        A_work = _spd(A + reg * np.eye(A.shape[0]))

    # Ultimate fallback: SVD-based square root
    try:
        U, s, Vt = np.linalg.svd(A_work)
        s = np.maximum(s, regularization)
        return U @ np.diag(np.sqrt(s))
    except:
        return np.sqrt(regularization) * np.eye(A.shape[0])


# --------------------------
# Enhanced Custom Sigma Points with better numerical stability
# --------------------------
class RobustMerweScaledSigmaPoints(MerweScaledSigmaPoints):
    """Merwe Scaled Sigma Points with enhanced robustness."""

    def sigma_points(self, x, P):
        """Compute sigma points with multiple fallback mechanisms."""
        if self.n != np.size(x):
            raise ValueError(f"Size of x ({np.size(x)}) must equal self.n ({self.n})")

        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        P = _spd(P, floor=1e-10)

        scale_factor = lambda_ + n
        if scale_factor <= 0:
            scale_factor = 1.0

        try:
            U = safe_cholesky(scale_factor * P)
        except:
            vals, vecs = np.linalg.eigh(P)
            vals = np.maximum(vals, 1e-12)
            U = vecs @ np.diag(np.sqrt(vals * scale_factor))

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x

        for k in range(n):
            col = U[:, k]
            sigmas[k+1] = x + col
            sigmas[n+k+1] = x - col

        return sigmas


# --------------------------
# Safe matrix inversion with fallbacks
# --------------------------
def safe_inv(A, regularization=1e-10):
    """Safely invert a matrix with fallback to pseudoinverse."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        try:
            A_reg = A + regularization * np.eye(A.shape[0])
            return np.linalg.inv(A_reg)
        except:
            return np.linalg.pinv(A)


# --------------------------
# Helpers
# --------------------------
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_omega(w, dt):
    th = np.linalg.norm(w*dt)
    if th < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = w / np.linalg.norm(w)
    half = 0.5 * th
    return np.hstack((np.cos(half), axis*np.sin(half)))

def skew(w):
    return np.array([
        [0,    -w[2],  w[1]],
        [w[2],  0,    -w[0]],
        [-w[1], w[0],  0]
    ])

def make_prior_from_excel(path):
    """Create prior from Excel file with error handling."""
    try:
        df = pd.read_excel(path)
        J = df[["Jxx_gen","Jyy_gen","Jzz_gen"]].values
        J_unit = J / np.linalg.norm(J, axis=1, keepdims=True)
        log_shape = np.log(np.maximum(J_unit, 1e-15))
        mu = log_shape.mean(axis=0)
        Sigma = np.cov(log_shape.T)
        Sigma = _spd(Sigma)
        return mu, Sigma
    except FileNotFoundError:
        print(f"Warning: File {path} not found. Using default prior.")
        mu_default = np.log(np.ones(3)/np.linalg.norm(np.ones(3)))
        Sigma_default = np.diag([0.05, 0.05, 0.05])
        return mu_default, Sigma_default


# --------------------------
# Scenario
# --------------------------
np.random.seed(42)
J_TRUE = np.diag([100., 80., 70.])
J_TRUE_DIAG = np.diag(J_TRUE)
NORM_TRUE = np.linalg.norm(J_TRUE_DIAG)

dt, T_RUN = 0.01, 400.0
steps = int(T_RUN / dt)


# --------------------------
# Excitation profiles (full / window / multi)
# --------------------------
def torque_profile(t, mode="full"):
    def base_signal(tt):
        return np.array([
            1.0*np.sin(0.1*tt) + 2.5*np.cos(0.3*tt) + 1.0*np.sin(0.7*tt) + 1.0*np.sin(1.5*tt),
            2.6*np.cos(0.15*tt) + 3.0*np.sin(0.4*tt) + 2.4*np.cos(0.8*tt) + 1.8*np.cos(1.8*tt),
            3.4*np.sin(0.12*tt) + 2.1*np.cos(0.5*tt) + 1.0*np.sin(0.9*tt) + 1.5*np.sin(2.0*tt)
        ])

    if mode == "full":
        return base_signal(t)

    if mode == "window":
        in_pulse = ((200.0 <= t < 201.0) or
                    (250.0 <= t < 251.0) or
                    (300.0 <= t < 301.0))
        return base_signal(t) if in_pulse else np.zeros(3)

    if mode == "multi":
        pulse_intervals = [(50 + 25*i, 51 + 25*i) for i in range(20)]
        in_pulse = any(a <= t < b for a, b in pulse_intervals)
        return base_signal(t) if in_pulse else np.zeros(3)

    raise ValueError("Mode must be 'full', 'window', or 'multi'.")


# --------------------------
# UKF model
# State: x = [ q(4), w(3), s(3)=log(unit-shape) ]
# --------------------------
def build_ukf(alpha=0.1, beta=2.0, kappa=0.0):
    dim_x, dim_z = 10, 7
    sp = RobustMerweScaledSigmaPoints(n=dim_x, alpha=alpha, beta=beta, kappa=kappa)

    def fx(x, dt, torque, ref_norm=1.0):
        q = x[:4]
        w = x[4:7]
        s = x[7:10]

        q = q / (np.linalg.norm(q) + 1e-15)

        s = np.clip(s, -10, 10)
        u = np.exp(s)
        u /= (np.linalg.norm(u) + 1e-15)
        J = np.diag(ref_norm * u)

        J_inv = safe_inv(J)
        wdot = J_inv @ (torque - skew(w) @ (J @ w))
        wdot = np.clip(wdot, -100, 100)

        w_new = w + wdot * dt
        q_new = quat_mult(q, quat_from_omega(w_new, dt))
        q_new /= (np.linalg.norm(q_new) + 1e-15)

        return np.hstack((q_new, w_new, s))

    def hx(x):
        return np.hstack((x[:4], x[4:7]))

    ukf = UnscentedKalmanFilter(dim_x=10, dim_z=7, dt=dt, fx=fx, hx=hx, points=sp)
    ukf.sp = sp
    return ukf


# --------------------------
# Virtual sensor scheduler (kept as-is)
# --------------------------
def vs_weight(t, mode):
    if mode == "full":
        return 1.0
    if mode == "window":
        if t < 200.0:
            return 0.0
        if t <= 391.0:
            return 1.0
        return max(0.0, 1.0 - (t - 391.0) / 40.0)
    if mode == "multi":
        if t < 50.0:
            return 0.0
        if t <= 391.0:
            return 1.0
        return max(0.0, 1.0 - (t - 391.0) / 40.0)
    return 0.0


# --------------------------
# Determine freeze time
# --------------------------
def get_freeze_time(mode):
    if mode == "full":
        return 0.0
    if mode == "window":
        return 200.0
    if mode == "multi":
        return 50.0
    return 0.0


# --------------------------
# CONTROL-SAFE FILTER (v2): hold during pulse + cooldown after pulse
# --------------------------
def smooth_hold_during_pulses(t, J_raw, tau_hist, dt,
                              tau_smooth=5.0,
                              tau_thresh=1e-6,
                              hold_after_pulse_sec=1.0):
    """
    Control-safe inertia:
      - HOLD during pulses
      - HOLD for hold_after_pulse_sec after each pulse ends (cooldown)
      - otherwise, low-pass smooth toward raw estimate
    """
    alpha = 1.0 - np.exp(-dt / tau_smooth)
    Jc = np.zeros_like(J_raw)
    Jc[0] = J_raw[0]

    hold_after_steps = int(max(0.0, hold_after_pulse_sec) / dt)
    cooldown = 0

    for k in range(1, len(t)):
        in_pulse = np.linalg.norm(tau_hist[k]) > tau_thresh

        if in_pulse:
            cooldown = hold_after_steps
            Jc[k] = Jc[k-1]
        elif cooldown > 0:
            cooldown -= 1
            Jc[k] = Jc[k-1]
        else:
            Jc[k] = (1 - alpha) * Jc[k-1] + alpha * J_raw[k]

    return Jc


# --------------------------
# Run one mode
# --------------------------
def run_mode(prior_mu, prior_Sigma, use_vs=True, mu_vs=None, sigma_vs0=0.02, mode="full"):
    freeze_time = get_freeze_time(mode)

    # Truth rollout
    w = np.array([0.1, 0.1, 0.1])
    q = np.array([1., 0., 0., 0.])
    w_hist = np.empty((steps, 3))
    q_hist = np.empty((steps, 4))
    tau_hist = np.empty((steps, 3))
    t_hist = np.linspace(0.0, T_RUN, steps)

    for k, t in enumerate(t_hist):
        tau = torque_profile(t, mode=mode)
        wdot = np.linalg.inv(J_TRUE) @ (tau - skew(w) @ (J_TRUE @ w))
        w += wdot * dt
        q = quat_mult(q, quat_from_omega(w, dt))
        q /= np.linalg.norm(q)
        w_hist[k] = w
        q_hist[k] = q
        tau_hist[k] = tau

    # Measurements
    sigma_gyro = 0.005
    sigma_quat = 0.005
    w_meas = w_hist + np.random.normal(0.0, sigma_gyro, w_hist.shape)
    q_meas = q_hist + np.random.normal(0.0, sigma_quat, q_hist.shape)
    q_meas /= np.linalg.norm(q_meas, axis=1, keepdims=True)
    z_meas = np.hstack((q_meas, w_meas))

    # UKF init
    ukf = build_ukf(alpha=1e-3, beta=2.0, kappa=0.0)
    ukf.x[:4] = q_meas[0]
    ukf.x[4:7] = w_meas[0]

    # Initial inertia guess
    J0 = np.array([140.0, 20.0, 36.06])
    Ibar0 = J0 / np.linalg.norm(J0)
    ukf.x[7:10] = np.log(np.maximum(Ibar0, 1e-15))

    # Covariances
    ukf.P = np.eye(10) * 1e-4
    ukf.P[7:10, 7:10] = np.diag([100.0, 300.0, 300.0])
    ukf.P = _spd(ukf.P)

    ukf.R = np.diag([sigma_quat**2]*4 + [sigma_gyro**2]*3)
    ukf.R += 1e-10 * np.eye(7)

    ukf.Q = np.diag([1e-7]*4 + [1e-7]*3 + [1e-7]*3)

    # Storage
    s_hist = np.empty((steps, 3))
    P_s_hist = np.empty((steps, 3, 3))

    s_prev = ukf.x[7:10].copy()
    P_s_prev = ukf.P[7:10, 7:10].copy()

    for k, t in enumerate(t_hist):
        tau = tau_hist[k]

        # Predict
        try:
            ukf.P = _spd(ukf.P)
            ukf.predict(torque=tau, ref_norm=NORM_TRUE)
            ukf.P = _spd(ukf.P)
        except Exception as e:
            print(f"Predict failed at step {k}: {e}")
            ukf.P = _spd(ukf.P + 1e-6*np.eye(10))
            try:
                ukf.predict(torque=tau, ref_norm=NORM_TRUE)
                ukf.P = _spd(ukf.P)
            except:
                pass

        # Update
        try:
            ukf.update(z_meas[k])
            ukf.P = _spd(ukf.P)
        except Exception as e:
            print(f"Update failed at step {k}: {e}")
            ukf.R += 1e-8*np.eye(7)
            ukf.P = _spd(ukf.P + 1e-7*np.eye(10))
            try:
                ukf.update(z_meas[k])
                ukf.P = _spd(ukf.P)
            except:
                pass

        # HARD FREEZE before excitation starts
        if t < freeze_time:
            ukf.x[7:10] = s_prev
            ukf.P[7:10, 7:10] = P_s_prev
            ukf.P[:7, 7:10] = 0.0
            ukf.P[7:10, :7] = 0.0
            ukf.P = _spd(ukf.P)
        else:
            s_prev = ukf.x[7:10].copy()
            P_s_prev = ukf.P[7:10, 7:10].copy()

        # Virtual Sensor update
        if mode != "full" and use_vs and (mu_vs is not None):
            ρ = vs_weight(t, mode)
            if ρ > 1e-6:
                try:
                    sigma_vs = sigma_vs0 / np.sqrt(ρ)
                    z_vs = mu_vs

                    def hx_vs(x):
                        return x[7:10]

                    R_vs = np.eye(3) * (sigma_vs**2) + 1e-12*np.eye(3)

                    ukf.P = _spd(ukf.P)
                    sigmas = ukf.sp.sigma_points(ukf.x, ukf.P)

                    Wm, Wc = ukf.sp.Wm, ukf.sp.Wc
                    Zsig = np.array([hx_vs(s) for s in sigmas])
                    z_pred = np.sum(Wm.reshape(-1, 1) * Zsig, axis=0)

                    S = np.zeros((3, 3))
                    Pxz = np.zeros((10, 3))
                    for i in range(Zsig.shape[0]):
                        dz = (Zsig[i] - z_pred).reshape(3, 1)
                        dx = (sigmas[i] - ukf.x).reshape(10, 1)
                        S += Wc[i] * (dz @ dz.T)
                        Pxz += Wc[i] * (dx @ dz.T)

                    S += R_vs
                    S = _spd(S)
                    S_inv = safe_inv(S)

                    K = Pxz @ S_inv
                    innov = (z_vs - z_pred)
                    ukf.x = ukf.x + K @ innov
                    ukf.P = ukf.P - K @ S @ K.T
                    ukf.P = _spd(ukf.P)

                except Exception as e:
                    print(f"Virtual sensor update failed at step {k}: {e}")
                    pass

        # Constraints
        qn = ukf.x[:4]
        ukf.x[:4] = qn / (np.linalg.norm(qn) + 1e-15)

        ukf.x[7:10] = np.clip(ukf.x[7:10], -5, 5)
        u = np.exp(ukf.x[7:10])
        u = u / (np.linalg.norm(u) + 1e-15)
        ukf.x[7:10] = np.log(np.maximum(u, 1e-15))

        ukf.P = _spd(ukf.P)

        s_hist[k] = ukf.x[7:10]
        P_s_hist[k] = ukf.P[7:10, 7:10]

    # Convert to absolute inertia
    u_all = np.exp(s_hist)
    u_all /= np.linalg.norm(u_all, axis=1, keepdims=True)
    J_abs_hist = NORM_TRUE * u_all

    return dict(
        t=t_hist, tau=tau_hist,
        s_hist=s_hist, P_s_hist=P_s_hist,
        J_abs_hist=J_abs_hist,
        final_J=J_abs_hist[-1].copy(),
        final_P=ukf.P.copy(),
        freeze_time=freeze_time
    )


# --------------------------
# Uncertainty propagation
# --------------------------
def _compute_sigma_J_improved(res):
    t = res['t']
    J = res['J_abs_hist']
    s_hist = res['s_hist']
    P_s_hist = res['P_s_hist']

    sig = np.zeros_like(J)
    for i in range(len(t)):
        s = s_hist[i]
        P_s = P_s_hist[i]

        u = np.exp(s)
        u_unit = u / (np.linalg.norm(u) + 1e-15)

        eps = 1e-8
        J_center = NORM_TRUE * u_unit

        J_jac = np.zeros((3, 3))
        for j in range(3):
            s_pert = s.copy()
            s_pert[j] += eps
            u_pert = np.exp(s_pert)
            u_pert_unit = u_pert / (np.linalg.norm(u_pert) + 1e-15)
            J_pert = NORM_TRUE * u_pert_unit
            J_jac[:, j] = (J_pert - J_center) / eps

        try:
            Sigma_J = J_jac @ P_s @ J_jac.T
            Sigma_J = _spd(Sigma_J)
            diag_vals = np.diag(Sigma_J)
        except:
            diag_vals = np.diag(P_s) * (J_center**2)

        diag_vals = np.maximum(diag_vals, 1e-15)
        sig[i] = np.sqrt(diag_vals)

        max_allowed_sigma = 0.3 * np.abs(J[i])
        sig[i] = np.minimum(sig[i], max_allowed_sigma)

    return sig


def _shade_mode(ax, mode):
    m = mode.lower()
    if "window" in m:
        for a, b in [(200, 201), (250, 251), (300, 301)]:
            ax.axvspan(a, b, color='gray', alpha=0.15, lw=0)
    elif "multi" in m:
        for a, b in [(50 + 25*i, 51 + 25*i) for i in range(20)]:
            ax.axvspan(a, b, color='gray', alpha=0.15, lw=0)
    elif "full" in m:
        pass


# --------------------------
# Plotting (control-safe by default if available)
# --------------------------
def plot_mode(ax, res, mode_name, title="", use_control_safe=True, show_raw=False):
    t = res['t']
    J_raw = res['J_abs_hist']
    J = res.get('J_abs_hist_ctrl', J_raw) if use_control_safe else J_raw

    try:
        sig = _compute_sigma_J_improved(res)
    except Exception:
        sig = np.zeros_like(J_raw)

    freeze_time = res.get('freeze_time', 200.0)

    colors = ['b', 'g', 'r']
    labels = ['Jx', 'Jy', 'Jz']

    if show_raw:
        for k, c in enumerate(colors):
            ax.plot(t, J_raw[:, k], c=c, lw=1.0, alpha=0.30)

    for k, (c, lbl) in enumerate(zip(colors, labels)):
        ax.plot(t, J[:, k], c=c, lw=2.2)
        mask = sig[:, k] > 0
        if np.any(mask):
            t_masked = t[mask]
            J_masked = J[:, k][mask]
            sig_masked = sig[:, k][mask]
            lower_bound = np.maximum(J_masked - sig_masked, 0.1)
            upper_bound = J_masked + sig_masked
            ax.fill_between(t_masked, lower_bound, upper_bound, color=c, alpha=0.12)
        ax.axhline(J_TRUE_DIAG[k], c=c, ls='--', alpha=0.7)

    _shade_mode(ax, mode_name)

    if freeze_time < 1000.0:
        ax.axvline(freeze_time, color='black', linestyle=':', alpha=0.5)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Inertia [kg·m²]')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(150, np.max(J) * 1.1))


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    prior_file = "gen10k_noise_0.0001_gen.xlsx"
    mu_wfm, Sigma_wfm = make_prior_from_excel(prior_file)
    print(f"Using WFM prior from: {prior_file}")
    print("μ_wfm (log-shape):", mu_wfm)
    print("diag(Σ_wfm):      ", np.diag(Sigma_wfm))

    mu_neutral = np.log(np.ones(3) / np.linalg.norm(np.ones(3)))
    Sigma_neutral = np.diag([0.08, 0.08, 0.08])

    print("\n=== Running FULL excitation:WFM + Virtual Sensor ===")
    res_full = run_mode(mu_neutral, Sigma_neutral, False, None, 0.02, "full")

    print("\n=== Running WINDOWED excitation: WFM + Virtual Sensor ===")
    res_window = run_mode(mu_neutral, Sigma_neutral, True, mu_wfm, 0.02, "window")

    print("\n=== Running MULTI excitation: WFM + Virtual Sensor ===")
    res_multi = run_mode(mu_neutral, Sigma_neutral, True, mu_wfm, 0.02, "multi")

    # ---- Control-safe smoothing (v2) ----
    tau_smooth = 5.0
    hold_after_pulse_sec = 1.0
    tau_thresh = 1e-6

    res_full["J_abs_hist_ctrl"] = smooth_hold_during_pulses(
        res_full["t"], res_full["J_abs_hist"], res_full["tau"], dt,
        tau_smooth=tau_smooth, tau_thresh=tau_thresh,
        hold_after_pulse_sec=hold_after_pulse_sec
    )
    res_window["J_abs_hist_ctrl"] = smooth_hold_during_pulses(
        res_window["t"], res_window["J_abs_hist"], res_window["tau"], dt,
        tau_smooth=tau_smooth, tau_thresh=tau_thresh,
        hold_after_pulse_sec=hold_after_pulse_sec
    )
    res_multi["J_abs_hist_ctrl"] = smooth_hold_during_pulses(
        res_multi["t"], res_multi["J_abs_hist"], res_multi["tau"], dt,
        tau_smooth=tau_smooth, tau_thresh=tau_thresh,
        hold_after_pulse_sec=hold_after_pulse_sec
    )

    # Sanity check
    print("\nSanity check (should be > 0):")
    print("max |raw - ctrl| FULL   :", np.max(np.abs(res_full["J_abs_hist"] - res_full["J_abs_hist_ctrl"])))
    print("max |raw - ctrl| WINDOW :", np.max(np.abs(res_window["J_abs_hist"] - res_window["J_abs_hist_ctrl"])))
    print("max |raw - ctrl| MULTI  :", np.max(np.abs(res_multi["J_abs_hist"] - res_multi["J_abs_hist_ctrl"])))


# --------------------------
# Plot styling
# --------------------------
LABEL_FS  = 28
TICK_FS   = 22
LW        = 2

def finalize_figure(ax, fig, filename_prefix):
    ax.set_title("")
    ax.set_xlabel('Time [s]', fontsize=LABEL_FS)
    ax.set_ylabel('Inertia [kg·m²]', fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 250)
    ax.set_xticks(np.arange(0, 401, 50))

    for line in ax.lines:
        line.set_linewidth(LW)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{filename_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename_prefix}.png", bbox_inches="tight")


# --------- Plot set 1: control-safe only ----------
fig_f, ax_f = plt.subplots(figsize=(8.6, 5.2))
plot_mode(ax_f, res_full, "Full", title="", use_control_safe=False, show_raw=False)
finalize_figure(ax_f, fig_f, "ukf_wfm_vs_full")

fig_w, ax_w = plt.subplots(figsize=(8.6, 5.2))
plot_mode(ax_w, res_window, "Windowed", title="", use_control_safe=True, show_raw=False)
finalize_figure(ax_w, fig_w, "ukf_wfm_vs_windowed")

fig_m, ax_m = plt.subplots(figsize=(8.6, 5.2))
plot_mode(ax_m, res_multi, "Multi", title="", use_control_safe=True, show_raw=False)
finalize_figure(ax_m, fig_m, "ukf_wfm_vs_multi")

plt.show()