import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import higher
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import autograd
from scipy.integrate import solve_ivp
import copy  
# Set fixed random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create a directory for results if it doesn't exist
os.makedirs("lrw_results", exist_ok=True)
_rng = np.random.default_rng()   
#############################################
# 1. Dataset Generation (Inertia Simulation)
#############################################
# ===============================================================
#  Weighted‑Flow‑Matching / LRW Pipeline – FULL SCRIPT
#  • Fixed nominal inertia   J_c  = diag(100, 80, 70)  kg·m²
#  • Synthetic inertia       J_s  = diag(U(±10 %)  for each principal moment)
#  • Generates dataset entries with:
#       time grid, ω_c_true(t), ω_c_synthetic(t), ω_c_measured(t)
# ===============================================================

# ---------------------------------------------------------------
# 0.  Globals & reproducibility
# ---------------------------------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("lrw_results", exist_ok=True)

# ---------------------------------------------------------------
# 1.  Dynamics helpers
# ---------------------------------------------------------------
def chief_rotational_dynamics(t, omega_c, J_c):
    """
    Euler’s rotational‑dynamics equation for a torque‑free rigid body.
    """
    omega_c = np.asarray(omega_c)
    Jw      = J_c @ omega_c
    cross   = np.cross(omega_c, Jw)
    return np.linalg.inv(J_c) @ (-cross)

def simulate_chief_motion(J_c, omega_c0,
                          T: float = 30.0,
                          num_points: int = 600):
    """
    Integrate the rotational dynamics with SciPy’s solve_ivp.
    Returns time grid (num_points,) and trajectory (num_points, 3).
    """
    t_span = (0.0, T)
    t_eval = np.linspace(*t_span, num_points)

    sol = solve_ivp(
        lambda t, y: chief_rotational_dynamics(t, y, J_c),
        t_span,
        y0=omega_c0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
    return sol.t, sol.y.T


def measurement_model_omega_c(omega_c_true, noise_std: float = 0.001, rng=_rng):
    """
    Add zero‑mean Gaussian noise to the true ω_c trajectory.
    """

    noise = rng.normal(0.0, noise_std, size=omega_c_true.shape)
    return omega_c_true + noise

def compute_error(omega_c_1, omega_c_2):
    # Mean squared error between two trajectories.
    return np.mean((omega_c_1 - omega_c_2)**2)
# ---------------------------------------------------------------
# 2.  Dataset generator  (NEW ±5 % uniform jitter in J_s)
# ---------------------------------------------------------------
def generate_base_dataset(num_samples: int = 2000,
                          T: float = 30.0,
                          num_points: int = 600,
                          noise_std: float = 0.001):
    # ---- Nominal constants --------------------------------------------------
    J_xx_nom, J_yy_nom, J_zz_nom = 100.0, 80.0, 70.0              # kg·m²
    J_c_true = np.diag([J_xx_nom, J_yy_nom, J_zz_nom])            # fixed
    omega_c0 = np.array([1, 1, 1])                # rad/s IC
    
    # ---- Pre‑simulate the true trajectory (shared for speed) ----------------
    time_grid, omega_c_true = simulate_chief_motion(
        J_c_true, omega_c0, T=T, num_points=num_points
    )

    # ---- Build samples ------------------------------------------------------
    dataset = []
    rng = np.random.default_rng(42)  # reproducible jitter
    for i in range(num_samples):
        # 2.1 Draw J_s using Gaussian distribution (std dev of ~5% of nominal value)
        J_xx_s = rng.normal(J_xx_nom, 0.07 * J_xx_nom)
        J_yy_s = rng.normal(J_yy_nom, 0.07 * J_yy_nom)
        J_zz_s = rng.normal(J_zz_nom, 0.07 * J_zz_nom)
        
        
        J_s = np.diag([J_xx_s, J_yy_s, J_zz_s])

        # 2.2  Synthetic trajectory
        _, omega_c_synthetic = simulate_chief_motion(
            J_s, omega_c0, T=T, num_points=num_points
        )

        # 2.3  Noisy measurement of the *true* trajectory
        omega_c_measured = measurement_model_omega_c(
            omega_c_true, noise_std=noise_std
        )

        # 2.4  Assemble sample
        dataset.append({
            "id": i,
            "time": time_grid,
            "J_c_true": J_c_true,
            "J_s": J_s,
            "omega_c_true": omega_c_true,
            "omega_c_synthetic": omega_c_synthetic,
            "omega_c_measured": omega_c_measured,
        })

    return dataset



# ---------------------------------------------------------------
# 3.  (Optional) quick diagnostic plot – first sample
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Generating a tiny dataset for sanity-check …")
    data   = generate_base_dataset(num_samples=1, num_points=600)
    sample = data[-1]                                # one (ω_true, ω_syn) pair

    # ------------------------------------------------------------------
    # Build noisy trajectories and inspect the raw MSE vs. sensor noise
    # ------------------------------------------------------------------
    noise_levels   = [0.0001, 0.001, 0.01]
    measured_trajs = {
        sigma: measurement_model_omega_c(sample["omega_c_true"], noise_std=sigma)
        for sigma in noise_levels
    }

    print("\nRaw MSE between ω_synthetic and ω_measured:")
    ω_syn = sample["omega_c_synthetic"]

    for sigma in noise_levels:
        mse = compute_error(ω_syn, measured_trajs[sigma])
        print(f"  σ = {sigma:0.3f}  →  raw MSE = {mse:.3e}")

    # ------------------------------------------------------------------
    # 3-D Plot with consistent styling
    # ------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401



import matplotlib.pyplot as plt


fig = plt.figure(figsize=(18, 12))                 # BIGGER figure
ax  = fig.add_subplot(111, projection='3d')

# --- your data plots ---
ax.plot(*sample["omega_c_true"].T,      lw=3.0, label="True")
ax.plot(*sample["omega_c_synthetic"].T, lw=2.8, ls='--', label="Synthetic")
styles = {0.0001: {'ls':'-.', 'lw':2.6},
          0.001:  {'ls':'--', 'lw':2.6},
          0.01:   {'ls':':',  'lw':2.6}}
for s, traj in measured_trajs.items():
    ax.plot(*traj.T, label=f"Measured (σ={s})", **styles[s])

# Bigger axis labels
ax.set_xlabel(r"$\omega_x$ [rad/s]", fontsize=30, labelpad=36)
ax.set_ylabel(r"$\omega_y$ [rad/s]", fontsize=30, labelpad=36)
ax.set_zlabel(r"$\omega_z$ [rad/s]", fontsize=30, labelpad=42)

# Bigger tick numbers (all axes)
ax.tick_params(axis='both', which='major', labelsize=22, pad=12)
ax.tick_params(axis='z',   which='major', labelsize=22, pad=14)  # optional: nudge z separately


# Make ticks and box look balanced
ax.tick_params(pad=6)
ax.set_box_aspect([1, 1, 1])               # nicer 3D proportions
ax.set_xlim([0.8, 1.15])
ax.set_ylim([-1.3, 1.3])
ax.set_zlim([-1.3, 1.3])


ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    frameon=True, fancybox=True,
    framealpha=0.9, borderpad=0.6,
    fontsize=18   # <-- increase legend text size
)

# Squeeze outer margins
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96)

plt.show()





def apply_noise_to_dataset(base_dataset, noise_std):
    """
    Add Gaussian noise to ω_true, compute a *noise-normalised* error, and
    assign class labels.  Returns the noisy dataset list.
    """
    noisy_dataset = []
    dyn_errors    = []          # physics-only error used for stats / labels


    # ---------- 1st pass: inject noise & compute errors ---------------------
    for sample in base_dataset:
        s = copy.deepcopy(sample)

        # Gaussian measurement noise
        s['omega_c_measured'] = measurement_model_omega_c(
            s['omega_c_true'], noise_std=noise_std)

        # raw mean-squared error between synthetic & measured
        raw_err = compute_error(s['omega_c_synthetic'], s['omega_c_measured'])

        # -------- NEW: noise-compensated *ratio* error ----------------------
        dyn_err          = raw_err 
        s['error']       = dyn_err
        # optional auxiliary feature
        s['err_l2']      = np.sqrt(dyn_err)     # you can keep your old formula
        # --------------------------------------------------------------------

        dyn_errors.append(dyn_err)
        noisy_dataset.append(s)

    # ---------- 2nd pass: z-score & median split ---------------------------
    err_mu   = np.mean(dyn_errors)
    err_std  = np.std (dyn_errors) + 1e-12     # avoid divide-by-zero
    thresh   = np.median(dyn_errors)

    for s in noisy_dataset:
        s['err_z']      = (s['error'] - err_mu) / err_std
        s['class_label'] = 0 if s['error'] <= thresh else 1

    return noisy_dataset


# build the base dataset once
base_dataset = generate_base_dataset(noise_std=0.0)


def prepare_datasets(dataset):
    # Classification by angular-velocity error

    random.seed(None)

    # Meta-set (balanced, extreme)
    meta_size_total = 100
    meta_size_per_class = meta_size_total // 2  # 50 each

    class0_samples = [s for s in dataset if s['class_label'] == 0]
    class1_samples = [s for s in dataset if s['class_label'] == 1]

    class0_sorted = sorted(class0_samples, key=lambda x: x['error'])  # easiest c0
    class1_sorted = sorted(class1_samples, key=lambda x: x['error'], reverse=True)  # hardest c1

    meta_set = class0_sorted[:meta_size_per_class] + class1_sorted[:meta_size_per_class]

    meta_ids = {s['id'] for s in meta_set}
    remaining_samples = [s for s in dataset if s['id'] not in meta_ids]

    # Imbalanced training set (10% / 90%)
    n_train = 700
    n_train_class0 = int(0.10 * n_train)  # 140 desired
    n_train_class1 = n_train - n_train_class0

    remaining_class0 = [s for s in remaining_samples if s['class_label'] == 0]
    remaining_class1 = [s for s in remaining_samples if s['class_label'] == 1]

    # never ask for more than we have
    n_train_class0 = min(n_train_class0, len(remaining_class0))
    n_train_class1 = min(n_train_class1, len(remaining_class1))

    training_set = random.sample(remaining_class0, n_train_class0) + random.sample(remaining_class1, n_train_class1)
    random.shuffle(training_set)

    training_ids = {s['id'] for s in training_set}

    # Balanced test set (250 / 250 if possible)
    n_test = 250
    n_test_per_class = n_test // 2  # 250 desired

    remaining_after_train = [s for s in remaining_samples if s['id'] not in training_ids]
    test_class0 = [s for s in remaining_after_train if s['class_label'] == 0]
    test_class1 = [s for s in remaining_after_train if s['class_label'] == 1]

    # keep it perfectly balanced with what is available
    n_test_per_class = min(n_test_per_class, len(test_class0), len(test_class1))

    test_set = random.sample(test_class0, n_test_per_class) + random.sample(test_class1, n_test_per_class)
    random.shuffle(test_set)

    # Reset random seed
    random.seed(None)

    print(f"Meta set size: {len(meta_set)} samples (balanced).")
    print(f"Training set size: {len(training_set)} samples "
          f"(class 0: {n_train_class0}, class 1: {n_train_class1}).")
    print(f"Test set size: {len(test_set)} samples "
          f"(class 0: {n_test_per_class}, class 1: {n_test_per_class}).")
    
    return meta_set, training_set, test_set



def plot_trajectories(sample, noise_std):
    # Local styling constants
    LABEL_FS  = 18
    TICK_FS   = 16
    LEGEND_FS = 14
    LW        = 2

    t_vals = sample['time']
    omega_true = sample['omega_c_true']
    omega_measured = sample['omega_c_measured']
    omega_synthetic = sample['omega_c_synthetic']

    axis_labels = ['ωₓ', 'ωᵧ', 'ω_z']
    colors = ['C0', 'C1', 'C2']

    # ========== Trajectories ==========
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for idx, label in enumerate(axis_labels):
        ax.plot(t_vals, omega_true[:, idx], label=f"True {label}", linestyle='-', color=colors[idx], linewidth=LW)
        ax.plot(t_vals, omega_measured[:, idx], label=f"Measured {label}", linestyle='--', color=colors[idx], linewidth=LW)
        ax.plot(t_vals, omega_synthetic[:, idx], label=f"Synthetic {label}", linestyle=':', color=colors[idx], linewidth=LW)

    ax.set_xlabel("Time [s]", fontsize=LABEL_FS)
    ax.set_ylabel("Angular Velocity [rad/s]", fontsize=LABEL_FS)
    ax.set_title(f"Angular Velocity Trajectories (noise_std={noise_std})", fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.grid(alpha=0.3)

    ax.legend(
        loc='upper right',
        fontsize=LEGEND_FS,
        frameon=True, fancybox=True,
        framealpha=0.9, borderpad=0.6, labelspacing=0.4
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout(pad=0.4)
    plt.show()


def plot_training_meas_vs_syn_three(
        training_sets,          # dict  {σ: training_set}
        noise_order,            # list [σ1, σ2, σ3]  (display order)
        folder="lrw_results"):
    """
    Creates a single 1×3 figure:
        left   = training set for σ1
        middle = training set for σ2
        right  = training set for σ3

    Each subplot overlays ω_measured (thin coloured) and ω_synthetic
    (thick black) for *all* samples in that training subset.

    Saves  lrw_results/train_meas_vs_syn_all_noise_levels.png
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    os.makedirs(folder, exist_ok=True)

    fig = plt.figure(figsize=(18, 5))

    for idx, σ in enumerate(noise_order, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        sub = training_sets[σ]

        colours = plt.cm.viridis(
            np.linspace(0, 1, len(sub), endpoint=False))

        for col, s in zip(colours, sub):
            # thin measured
            ax.plot(*s['omega_c_measured'].T,  lw=0.6, c=col, alpha=.4)
            # thick synthetic
            ax.plot(*s['omega_c_synthetic'].T, lw=1.5, c=col, alpha=.9)

        ax.set_xlabel(r"$\omega_x$"); ax.set_ylabel(r"$\omega_y$")
        ax.set_zlabel(r"$\omega_z$")
        ax.set_title(f"Training set  σ={σ:g}")

    plt.tight_layout()
    f_name = f"{folder}/train_meas_vs_syn_all_noise_levels.png"
    plt.savefig(f_name, dpi=200)
    plt.close()
    print(f"saved  {f_name}")


class AngularVelocityDataset(Dataset):
    def __init__(self, samples, mu_err, sd_err, sd_l2):
        self.samples = samples
        self.mu_err  = mu_err
        self.sd_err  = sd_err
        self.sd_l2   = sd_l2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # z-score with σ-specific statistics
        err_norm = (s['error'] - self.mu_err) / self.sd_err
        feat = torch.tensor([err_norm,
                             s['err_l2'] / self.sd_l2], dtype=torch.float32)
        label = torch.tensor(s['class_label'], dtype=torch.float32)
        return feat, label, idx


class MLP3D(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)      # ← back to 1
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)   # (B,) logit


def get_loss_n_accuracy(model, criterion, data_loader, device, num_classes=2, return_confusion=False):
    old_reduction = criterion.reduction
    criterion.reduction = 'mean'
    model.eval()
    total_loss = 0.0
    total_correct = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / (confusion_matrix.sum(dim=1) + 1e-8)
    criterion.reduction = old_reduction
    if return_confusion:
        return avg_loss, (accuracy, per_class_accuracy), confusion_matrix
    else:
        return avg_loss, (accuracy, per_class_accuracy)

def plot_confusion_matrix(conf_mat, title="Confusion Matrix", filename="confusion_matrix.png"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat.cpu().numpy(), annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(filename)
    plt.close()

def train_and_evaluate(meta_set, training_set, test_set,
                       mu_err, sd_err, sd_l2,          # NEW
                       noise_std, device, n_epochs=200):
    # Set seed for reproducibility in training
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    train_dataset = AngularVelocityDataset(training_set, mu_err, sd_err, sd_l2)
    meta_dataset  = AngularVelocityDataset(meta_set,    mu_err, sd_err, sd_l2)
    test_dataset  = AngularVelocityDataset(test_set,    mu_err, sd_err, sd_l2)

    # Create data loaders
    bs = 32
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    meta_bs = min(len(meta_dataset), 8)
    meta_loader = DataLoader(meta_dataset, batch_size=meta_bs, shuffle=True)
    meta_loader = itertools.cycle(meta_loader)

    criterion = nn.BCEWithLogitsLoss().to(device)

    # Baseline training
    print(f"\n=== Baseline Training for noise_std={noise_std} ===")
    baseline_model = MLP3D(input_dim=2).to(device)
    optimizer_baseline = optim.SGD(baseline_model.parameters(), lr=1e-4)
    baseline_train_losses = []
    baseline_test_losses = []
    baseline_train_accs = []
    baseline_test_accs = []

    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        baseline_model.train()
        epoch_loss = 0.0
        correct = 0
        n_samples = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_baseline.zero_grad()
            outputs = baseline_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_baseline.step()
            batch_size_ = inputs.size(0)
            epoch_loss += loss.item() * batch_size_
            n_samples += batch_size_
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
        avg_loss = epoch_loss / n_samples
        train_acc = correct / n_samples
        test_loss, (test_acc, _) = get_loss_n_accuracy(baseline_model, criterion, test_loader, device)
        baseline_train_losses.append(avg_loss)
        baseline_test_losses.append(test_loss)
        baseline_train_accs.append(train_acc)
        baseline_test_accs.append(test_acc)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Train Loss {avg_loss:.4f}, Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

    baseline_time = time.time() - start_time
    print(f"Baseline training took {baseline_time:.2f} seconds.")
    test_loss, (test_acc, _), conf_mat = get_loss_n_accuracy(baseline_model, criterion, test_loader, device, return_confusion=True)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    plot_confusion_matrix(conf_mat, 
                         title=f"Baseline Confusion Matrix (noise_std={noise_std})",
                         filename=f"lrw_results/baseline_confusion_noise_{noise_std}.png")

    # Meta-learning reweighting
    print(f"\n=== Meta-Learning Reweighting for noise_std={noise_std} ===")
    meta_model = MLP3D(input_dim=2).to(device)
    optimizer_meta = optim.SGD(meta_model.parameters(), lr=3e-4)
    sample_weights_record = {i: [] for i in range(len(train_dataset))}
    meta_train_losses = []
    meta_test_losses = []
    meta_train_accs = []
    meta_test_accs = []

    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        meta_model.train()
        epoch_loss = 0.0
        correct = 0
        n_samples = 0
        for inputs, labels, indices in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_meta.zero_grad()
            with higher.innerloop_ctx(meta_model, optimizer_meta, copy_initial_weights=False) as (m_model, m_opt):
                criterion.reduction = 'none'
                meta_outputs = m_model(inputs)
                meta_loss = criterion(meta_outputs, labels)
                eps = torch.zeros_like(meta_loss, requires_grad=True)
                weighted_meta_loss = torch.sum(eps * meta_loss)
                m_opt.step(weighted_meta_loss)
                
                meta_inputs, meta_labels, _ = next(meta_loader)
                meta_inputs, meta_labels = meta_inputs.to(device), meta_labels.to(device)
                criterion.reduction = 'mean'
                meta_val_outputs = m_model(meta_inputs)
                meta_val_loss = criterion(meta_val_outputs, meta_labels)
                eps_grads = autograd.grad(meta_val_loss, eps)[0].detach()
            w_tilde = torch.clamp(-eps_grads, min=0.0)
            l1_norm = w_tilde.sum()
            if l1_norm.item() == 0:
               w = torch.ones_like(w_tilde) / w_tilde.numel()
            else:
               w = w_tilde / l1_norm
            
            # Record weights per sample
            for idx, weight in zip(indices, w.cpu().numpy()):
                sample_weights_record[idx.item()].append(weight)
            
            criterion.reduction = 'none'
            final_outputs = meta_model(inputs)
            final_loss = criterion(final_outputs, labels)
            weighted_loss = torch.sum(w * final_loss)
            weighted_loss.backward()
            optimizer_meta.step()
            
            batch_size_ = inputs.size(0)
            epoch_loss += weighted_loss.item() * batch_size_
            n_samples += batch_size_
            preds = (torch.sigmoid(final_outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
        avg_loss = epoch_loss / n_samples
        train_acc = correct / n_samples
        test_loss, (test_acc, _) = get_loss_n_accuracy(meta_model, criterion, test_loader, device)
        meta_train_losses.append(avg_loss)
        meta_train_accs.append(train_acc)
        meta_test_losses.append(test_loss)
        meta_test_accs.append(test_acc)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: Train Loss {avg_loss:.4f}, Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

    meta_time = time.time() - start_time
    print(f"Meta-learning training took {meta_time:.2f} seconds.")
    test_loss, (test_acc, _), meta_conf_mat = get_loss_n_accuracy(meta_model, criterion, test_loader, device, return_confusion=True)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    plot_confusion_matrix(meta_conf_mat, 
                         title=f"Meta-Learning Confusion Matrix (noise_std={noise_std})",
                         filename=f"lrw_results/meta_confusion_noise_{noise_std}.png")
    # ----- Consistent styling (match your other figures) -----
    LABEL_FS  = 28
    TICK_FS   = 22
    LEGEND_FS = 20
    LW        = 2
    FIG_W, FIG_H, DPI = 7, 3.8, 400  # identical canvas for every figure

    def _plot_curve(epochs, series_list, labels, y_label, title, out_path, logy=False):
        """Generic curve plot with publication settings and fixed canvas."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

        # Consistent styles (extend if you pass more than 2 series)
        styles = [
            dict(lw=LW, color='C0', ls='-'),   # Baseline
            dict(lw=LW, color='C1', ls='--'),  # Meta/LRW
            dict(lw=LW, color='C2', ls='-.'),  # (optional) 3rd
            dict(lw=LW, color='C3', ls=':'),   # (optional) 4th
            ]

        for i, y in enumerate(series_list):
            if logy:
                ax.semilogy(epochs, y, label=labels[i], **styles[i % len(styles)])
            else:
                ax.plot(epochs, y, label=labels[i], **styles[i % len(styles)])

        # Labels, ticks, grid
        ax.set_xlabel('Epoch', fontsize=LABEL_FS)
        ax.set_ylabel(y_label, fontsize=LABEL_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.grid(alpha=0.3, which='both')

        # Legend outside (no overlap) + fixed margins for identical layout
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=LEGEND_FS, frameon=True, borderaxespad=0.)
        # Fixed margins remove variability across titles/labels and keep size identical
        fig.subplots_adjust(left=0.14, right=0.78, bottom=0.16, top=0.90)

        if title:
            ax.set_title(title, fontsize=LABEL_FS, pad=10)

        # Ensure limits are consistent (prevents leftover interactive zoom issues)
        ax.set_xlim(min(epochs), max(epochs))
        # (Optional) uncomment to lock y limits across runs
        # ax.set_ylim(0.0, 1.0)  # for accuracy
        # ax.set_ylim(0.45, 0.75)  # for loss, example

        # Save with fixed canvas (avoid bbox_inches='tight' which can vary sizing)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)

    # ------------------ Use the helper for your four plots ------------------
    epochs_list = range(1, n_epochs + 1)

    # 1) Test Accuracy
    _plot_curve(
        epochs_list,
        [baseline_test_accs, meta_test_accs],
        ['Baseline Test Acc', 'Meta Test Acc'],
        y_label='Test Accuracy',
        title=f'Test Accuracy vs. Epoch (noise_std={noise_std})',
        out_path=f"lrw_results/test_acc_noise_{noise_std}.png",
        logy=False
        )

    # 2) Test Loss
    _plot_curve(
        epochs_list,
        [baseline_test_losses, meta_test_losses],
        ['Baseline Test Loss', 'Meta Test Loss'],
        y_label='Test Loss',
        title=f'Test Loss vs. Epoch (noise_std={noise_std})',
        out_path=f"lrw_results/test_loss_noise_{noise_std}.png",
        logy=False  # set True if you prefer semilogy for loss
        )

    # 3) Train Accuracy
    _plot_curve(
        epochs_list,
        [baseline_train_accs, meta_train_accs],
        ['Baseline Train Acc', 'Meta Train Acc'],
        y_label='Train Accuracy',
        title=f'Train Accuracy vs. Epoch (noise_std={noise_std})',
        out_path=f"lrw_results/train_acc_noise_{noise_std}.png",
        logy=False
        )

    # 4) Train Loss
    _plot_curve(
        epochs_list,
        [baseline_train_losses, meta_train_losses],
        ['Baseline Train Loss', 'Meta Train Loss'],
        y_label='Train Loss',
        title=f'Train Loss vs. Epoch (noise_std={noise_std})',
        out_path=f"lrw_results/train_loss_noise_{noise_std}.png",
        logy=False  # set True if you want log-scale here
        )


    # Plot training curves
    epochs_list = range(1, n_epochs + 1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_list, baseline_test_accs, label='Baseline Test Acc')
    plt.plot(epochs_list, meta_test_accs, label='Meta Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy vs. Epoch (noise_std={noise_std})')
    plt.legend()
    plt.savefig(f"lrw_results/test_acc_noise_{noise_std}.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_list, baseline_test_losses, label='Baseline Test Loss')
    plt.plot(epochs_list, meta_test_losses, label='Meta Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(f'Test Loss vs. Epoch (noise_std={noise_std})')
    plt.legend()
    plt.savefig(f"lrw_results/test_loss_noise_{noise_std}.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_list, baseline_train_accs, label='Baseline Train Acc')
    plt.plot(epochs_list, meta_train_accs, label='Meta Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title(f'Train Accuracy vs. Epoch (noise_std={noise_std})')
    plt.legend()
    plt.savefig(f"lrw_results/train_acc_noise_{noise_std}.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs_list, baseline_train_losses, label='Baseline Train Loss')
    plt.plot(epochs_list, meta_train_losses, label='Meta Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'Train Loss vs. Epoch (noise_std={noise_std})')
    plt.legend()
    plt.savefig(f"lrw_results/train_loss_noise_{noise_std}.png")
    plt.close()

    # Average sample weights and save
    avg_sample_weights = {}
    for idx, w_list in sample_weights_record.items():
        avg_sample_weights[idx] = sum(w_list) / len(w_list) if len(w_list) > 0 else 0.0

    torch.save(avg_sample_weights, f'lrw_results/avg_sample_weights_noise_{noise_std}.pt')
    
    # Build a DataFrame for the training set with sample IDs for consistent tracking
    data_list = []
    raw_w_list = []
    norm_w_list = []
    id_list = []
    total_weight = sum(avg_sample_weights.values())
    
    for idx in range(len(train_dataset)):
        # Get the original sample ID for consistent tracking across noise levels
        original_id = training_set[idx]['id']
        id_list.append(original_id)
        
        # Feature: normalized error
        feature = train_dataset[idx][0].numpy()
        # Retrieve the corresponding synthetic inertia diagonal
        Js_vector = np.diag(training_set[idx]['J_s'])  # [J_xx, J_yy, J_zz]
        data_list.append(feature.tolist() + Js_vector.tolist())
        
        rw = avg_sample_weights[idx]
        raw_w_list.append(rw)
        nw = rw / total_weight if total_weight != 0 else 0.0
        norm_w_list.append(nw)

    # Create DataFrame with columns for features, weights, and sample IDs
    columns = ['err_norm', 'err_l2', 'Js_xx', 'Js_yy', 'Js_zz']
    df_weights = pd.DataFrame(data_list, columns=columns)
    df_weights['Sample_ID'] = id_list  # Add sample ID column for tracking
    df_weights['Raw_Weight'] = raw_w_list
    df_weights['Normalized_Weight'] = norm_w_list
    df_weights.to_excel(f'lrw_results/training_data_weights_noise_{noise_std}.xlsx', index=False)
    print(f"training_data_weights_noise_{noise_std}.xlsx saved.")
    
    return {
        "baseline_model"       : baseline_model,
        "meta_model"           : meta_model,
        "baseline_accuracies"  : baseline_test_accs,   # (= test accuracy curve)
        "meta_accuracies"      : meta_test_accs,
        #  add the six lists below ⬇
        "baseline_train_accs"  : baseline_train_accs,
        "meta_train_accs"      : meta_train_accs,
        "baseline_test_losses" : baseline_test_losses,
        "meta_test_losses"     : meta_test_losses,
        "baseline_train_losses": baseline_train_losses,
        "meta_train_losses"    : meta_train_losses,
        # leave the rest unchanged
        "avg_sample_weights"   : avg_sample_weights,
        "training_sample_ids"  : id_list
    }

def main():
    # Check for GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate base dataset once (with no noise)
    print("Generating base dataset...")
    base_dataset = generate_base_dataset()
    
    # List of noise standard deviations to test
    noise_std_values = [0.0001, 0.001, 0.01]

    # MODIFICATION: Prepare splits only once with a moderate noise level
    reference_noise = 0.001  # Use middle noise level for initial split
    tmp_ds = apply_noise_to_dataset(base_dataset, reference_noise)

    GLOBAL_ERR_MEAN = np.mean([s['error'] for s in tmp_ds])
    GLOBAL_ERR_STD  = np.std( [s['error'] for s in tmp_ds]) + 1e-12
    print(f"Global error μ={GLOBAL_ERR_MEAN:.4e}, σ={GLOBAL_ERR_STD:.4e}")
    print("\nPreparing reference dataset splits using noise_std =", reference_noise)
    reference_dataset = apply_noise_to_dataset(base_dataset, reference_noise)
    meta_set_ref, training_set_ref, test_set_ref = prepare_datasets(reference_dataset)
    
    # Store original dataset splits for all noise levels
    dataset_splits = {}
    results = {}
    
# Loop over each noise level
    for noise_std in noise_std_values:
        print("\n" + "="*50)
        print(f"Evaluating for noise_std = {noise_std}")
        
        # Apply noise to the base dataset
        print(f"Applying noise with std = {noise_std} to dataset...")
        noisy_dataset = apply_noise_to_dataset(base_dataset, noise_std)
        
        
        # 1) split into meta / train / test for *this* σ
        meta_set, training_set, test_set = prepare_datasets(noisy_dataset)
        
        
        # ----- local statistics for the CURRENT σ -----
        err_vals = np.asarray([s['error']  for s in noisy_dataset])
        l2_vals  = np.asarray([s['err_l2'] for s in noisy_dataset])

        mu_err   = err_vals.mean()
        sd_err   = err_vals.std() + 1e-12
        sd_l2    = l2_vals .std() + 1e-12


       
        dataset_splits[noise_std] = {
            'meta':  meta_set,
            'train': training_set,
            'test':  test_set
        }
        
        # Plot example trajectories
        print("Plotting example trajectories...")
        plot_trajectories(noisy_dataset[0], noise_std)
        
        # Train and evaluate
        print("Training and evaluating models...")
        results[noise_std] = train_and_evaluate(
               meta_set, training_set, test_set,
               mu_err, sd_err, sd_l2,                 # NEW
               noise_std, device, n_epochs=200
        )

    
    # Compare final test accuracies across noise levels
    print("\n" + "="*50)
    print("Comparing final test accuracies across noise levels:")
    

    training_sets_dict = {σ: dataset_splits[σ]['train']
                          for σ in noise_std_values}
    plot_training_meas_vs_syn_three(
        training_sets_dict,
        noise_order=noise_std_values)  
    
    
    baseline_accs = []
    meta_accs = []
    noise_labels = []
    
    for noise_std in noise_std_values:
        final_baseline_acc = results[noise_std]['baseline_accuracies'][-1]
        final_meta_acc = results[noise_std]['meta_accuracies'][-1]
        baseline_accs.append(final_baseline_acc)
        meta_accs.append(final_meta_acc)
        noise_labels.append(f"{noise_std:.5f}")
        print(f"noise_std = {noise_std:.5f}:")
        print(f"  Baseline final acc: {final_baseline_acc:.4f}")
        print(f"  Meta-learning final acc: {final_meta_acc:.4f}")
        print(f"  Improvement: {(final_meta_acc - final_baseline_acc) * 100:.2f}%")
    
    # Plot comparison of final accuracies
    plt.figure(figsize=(10, 6))
    x = range(len(noise_std_values))
    width = 0.35
    plt.bar([i - width/2 for i in x], baseline_accs, width, label='Baseline')
    plt.bar([i + width/2 for i in x], meta_accs, width, label='Meta-Learning')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Final Test Accuracy')
    plt.title('Comparison of Final Test Accuracies')
    plt.xticks(x, noise_labels)
    plt.legend()
    plt.savefig("lrw_results/final_accuracy_comparison.png")
    plt.close()
    
    # Analyze sample weights across noise levels
    print("\n" + "="*50)
    print("Analyzing sample weight consistency across noise levels...")
    
    # Create a DataFrame to track weights across noise levels
    weight_comparison = {}
    
    # Get a common set of sample IDs from the first noise level
    reference_noise_level = noise_std_values[0]
    reference_ids = results[reference_noise_level]['training_sample_ids']
    
    # For each sample ID, collect weights across noise levels
    for i, sample_id in enumerate(reference_ids):
        weights_across_noise = []
        for noise_std in noise_std_values:
            avg_weights = results[noise_std]['avg_sample_weights']
            # Find the index for this sample_id in the current noise level
            try:
                idx = results[noise_std]['training_sample_ids'].index(sample_id)
                weight = avg_weights[idx] if idx in avg_weights else 0.0
            except ValueError:
                weight = 0.0  # Sample not found in this noise level
            weights_across_noise.append(weight)
        weight_comparison[sample_id] = weights_across_noise
    
    # Convert to DataFrame
    columns = [f"weight_noise_{std}" for std in noise_std_values]
    weight_df = pd.DataFrame.from_dict(weight_comparison, orient='index', columns=columns)
    weight_df.to_excel("lrw_results/weight_comparison_across_noise.xlsx")
    print("Saved weight comparison across noise levels.")
    
    
    print("\nExperiment completed successfully!")
    
    
    
    #############################################################
    # Box-and-Whisker Chart of weight vs noise levels
    import seaborn as sns
    
    rows = []
    
    for sigma in noise_std_values:                 # e.g. [0.0001, 0.001, 0.01]
        for idx, w_raw in results[sigma]['avg_sample_weights'].items():
            rows.append({'σ': sigma, 'weight_raw': w_raw})

    df_w = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.violinplot(data=df_w, x='σ', y='weight_raw',
                   order=sorted(noise_std_values),
                   palette='Set2', inner=None, cut=0, ax=ax)
    sns.stripplot(data=df_w, x='σ', y='weight_raw',
                  order=sorted(noise_std_values),
                  color='k', alpha=0.25, size=2, ax=ax)

# overlay median & mean
    group_stats = df_w.groupby('σ')['weight_raw'].agg(['median', 'mean'])
    for x_pos, (σ, row) in enumerate(group_stats.iterrows()):
        ax.text(x_pos-0.18, row['median'], f"median\n{row['median']:.3e}",
                ha='right', va='center', fontsize=8, color='midnightblue')
        ax.text(x_pos+0.18, row['mean'  ], f"mean\n{row['mean']:.3e}",
                ha='left',  va='center', fontsize=8, color='firebrick')

    ax.set_yscale('log')
    ax.set_xlabel("Noise standard deviation σ")
    ax.set_ylabel("Raw LRW weight")
    ax.set_title("Distribution of LRW sample weights vs sensor noise")
    ax.grid(axis='y', which='both', ls='--', alpha=.3)

    fname = "lrw_results/weight_distribution_vs_noise_violin.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"saved  {fname}")

    #################################################


    
    subplot_titles = ["Test accuracy", "Train accuracy",
                  "Test loss",     "Train loss"]
    curve_keys     = [
        ('baseline_accuracies', 'meta_accuracies'),        # test acc
        ('baseline_train_accs', 'meta_train_accs'),        # train acc
        ('baseline_test_losses','meta_test_losses'),       # test loss
        ('baseline_train_losses','meta_train_losses')      # train loss
    ]
    
    ylabel_map = {
    "Test accuracy":  "Accuracy",
    "Train accuracy": "Accuracy",
    "Test loss":      "Loss",
    "Train loss":     "Loss",
    }

    colour_cycle = ['tab:blue', 'tab:orange', 'tab:green',
                    'tab:red',  'tab:purple', 'tab:brown']

    for title, key_pair in zip(subplot_titles, curve_keys):
        plt.figure(figsize=(8, 4))
        # no title
        plt.xlabel("Epoch")
        plt.ylabel(ylabel_map[title])
        plt.grid(True, alpha=.3)

        # ---------- move this loop INSIDE ----------
        for c, σ in zip(colour_cycle, noise_std_values):
            b_curve = results[σ][key_pair[0]]
            m_curve = results[σ][key_pair[1]]
            epochs  = range(1, len(b_curve) + 1)

            plt.plot(epochs, b_curve, color=c, linestyle='--',
                     label=f"Baseline σ={σ:g}")
            plt.plot(epochs, m_curve, color=c, linestyle='-',
                     label=f"Meta σ={σ:g}")
        # -------------------------------------------

        if "loss" in title.lower():
            plt.yscale('log')

        plt.legend(
            ncol=1,
            fontsize=12,
            frameon=True, fancybox=True, framealpha=0.95,
            borderpad=1.2, labelspacing=0.8, handlelength=2.2,
            loc="best"
        )

        fname = f"lrw_results/{title.lower().replace(' ', '_')}_all_noise_levels.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=200)

        # SHOW and keep window open (no plt.close())
        plt.show()            # use plt.show(block=True) if needed
    
    

if __name__ == "__main__":
    main()
    
    