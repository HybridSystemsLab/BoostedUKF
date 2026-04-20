# -*- coding: utf-8 -*-
"""
Updated script to run weighted flow matching on three different noise levels
and compare their convergence through KDE analysis.

@author: Yasar Yanik (with updates)
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import List
from tqdm import trange
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# Set device
device = torch.device("cpu")
print("CPU will be used.")

    
###############################################################################
#                          OT Flow Matching Classes                           #
###############################################################################
class OTFlowMatching:
    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation in time t from x_0 to x_1:
        psi_t = (1 - alpha(t)) * x_0 + alpha(t) * x_1,
        where alpha(t) = 1 - (1 - sig_min)*t.
        """
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted flow matching loss.
        - t is drawn in [0, 1-eps)
        - x_0 ~ Normal(0, I) in R^3
        - Computes the weighted MSE between the estimated derivative (v_psi)
          and the analytical derivative (d_psi).
        """
        # Create t in [0, 1-eps) for each sample in the batch.
        t = (torch.rand(1, device=x_1.device) + 
             torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        # Expand t to the same shape as x_1 (batch_size, 3)
        t = t[:, None].expand(x_1.shape)
        
        # x_0 ~ N(0,I) in R^3
        x_0 = torch.randn_like(x_1)
        
        # Evaluate the vector field at psi_t
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        
        # Analytical derivative: d_psi/dt = x_1 - (1 - sig_min)*x_0
        d_psi = x_1 - (1 - self.sig_min) * x_0
        
        # Compute per-sample squared error (summing over the 3 dimensions)
        mse = (v_psi - d_psi).pow(2).sum(dim=1)
        w = weights / (weights.sum() + 1e-9)     # re‑normalise
        w = torch.clamp(w, min=1e-4)             # keep > ~20 active pts
        weights_sum = w.sum()

        # ─── mix a bit of un‑weighted loss (α ≈ 0.9) ─────────────
        alpha = 0.9
        weighted_mse = (w * mse).sum() / weights_sum
        plain_mse    =  mse.mean()
        return alpha * weighted_mse + (1 - alpha) * plain_mse


###############################################################################
#                              Conditional VF                                 #
###############################################################################
class CondVF(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Helper for odeint: expands scalar t -> shape (batch_size,).
        """
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def encode(self, x_1: torch.Tensor) -> torch.Tensor:
        from zuko.utils import odeint
        return odeint(self.wrapper, x_1, 1., 0., self.parameters())

    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        from zuko.utils import odeint
        return odeint(self.wrapper, x_0, 0., 1., self.parameters())

    def decode_t0_t1(self, x_0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        from zuko.utils import odeint
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

###############################################################################
#                              Network Definition                             #
###############################################################################
class Net(nn.Module):
    """
    A time-dependent vector field for R^3 -> R^3 with time encoded via Fourier features.
    """
    def __init__(self, in_dim: int, out_dim: int, h_dims: List[int], n_frequencies: int) -> None:
        super().__init__()
        # The input has dimension: (x, y, z) + (2*n_frequencies) from time encoding
        ins = [in_dim + 2 * n_frequencies] + h_dims
        outs = h_dims + [out_dim]
        self.n_frequencies = n_frequencies

        # Hidden layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU())
            for in_d, out_d in zip(ins, outs)
        ])
        # Final layer
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal encoding of scalar t.
        t has shape (batch_size,).
        Output shape => (batch_size, 2*n_frequencies).
        """
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t_freq = freq * t[..., None]  # shape: (batch_size, n_frequencies)
        return torch.cat((t_freq.cos(), t_freq.sin()), dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Encode time t
        t_enc = self.time_encoder(t)
        # Concat x with encoded t
        in_vec = torch.cat((x, t_enc), dim=-1)

        # Pass through the MLP
        for layer in self.layers:
            in_vec = layer(in_vec)

        return self.top(in_vec)

###############################################################################
#                           Helper Functions                                  #
###############################################################################


def plot_kde_comparison(data, weights, x_1_hat, title=None):
    """
    Create KDE plots for real vs generated data (2D projection).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # ---------- styling constants ----------
    AXIS_LABEL_FONTSIZE = 22
    TICK_FONTSIZE       = 20
    TITLE_FONTSIZE      = 18
    CBAR_TICK_FONTSIZE  = 16
    # --------------------------------------

    # Project to 2D (take the first two dimensions)
    data_2d    = data[:, :2]
    x_1_hat_2d = x_1_hat[:, :2]

    # Define grid
    x_min = min(data_2d[:, 0].min(), x_1_hat_2d[:, 0].min())
    x_max = max(data_2d[:, 0].max(), x_1_hat_2d[:, 0].max())
    y_min = min(data_2d[:, 1].min(), x_1_hat_2d[:, 1].min())
    y_max = max(data_2d[:, 1].max(), x_1_hat_2d[:, 1].max())

    resolution = 100
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    X1_grid, X2_grid = np.meshgrid(x_range, y_range)
    coords = np.vstack([X1_grid.ravel(), X2_grid.ravel()])

    # Weighted KDE for 2D data
    kde_weighted = gaussian_kde(data_2d.T, weights=weights)
    n_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)  # Effective sample size
    bw_factor = (n_eff * np.product(data_2d.std(axis=0))) ** (-1/6)  # Silverman-like
    kde_weighted.set_bandwidth(bw_method=bw_factor)
    pdf_weighted = kde_weighted(coords).reshape(X1_grid.shape)

    # Generated KDE for 2D data
    kde_gen = gaussian_kde(x_1_hat_2d.T)
    pdf_gen = kde_gen(coords).reshape(X1_grid.shape)

    # Normalize PDFs to integrate to 1 on the grid
    cell_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    pdf_weighted /= (np.sum(pdf_weighted) * cell_area)
    pdf_gen      /= (np.sum(pdf_gen) * cell_area)

    # File suffix based on title
    suffix = f"_{title}" if title else ""

    # ------------------- figure -------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # >>> WHITE BACKGROUND (figure + axes) <<<
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    # >>> OPTIONAL: make very-low density show as white in the colormap <<<
    cmap_blues = plt.cm.Blues.copy()
    cmap_blues.set_under("white")
    cmap_reds = plt.cm.Reds.copy()
    cmap_reds.set_under("white")

    # Choose a vmin so "under" actually triggers (pick a small positive floor)
    eps1 = max(np.min(pdf_weighted[pdf_weighted > 0]), 1e-12)
    eps2 = max(np.min(pdf_gen[pdf_gen > 0]), 1e-12)

    # Weighted KDE
    c1 = axes[0].contourf(
        X1_grid, X2_grid, pdf_weighted,
        levels=20, cmap=cmap_blues, vmin=eps1, extend="min"
    )
    axes[0].scatter(data_2d[:, 0], data_2d[:, 1], s=5, alpha=0.3, color='blue')
    cb1 = fig.colorbar(c1, ax=axes[0])
    cb1.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    axes[0].set_xlabel('$J_x$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel('$J_y$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    # Generated KDE
    c2 = axes[1].contourf(
        X1_grid, X2_grid, pdf_gen,
        levels=20, cmap=cmap_reds, vmin=eps2, extend="min"
    )
    axes[1].scatter(x_1_hat_2d[:, 0], x_1_hat_2d[:, 1], s=5, alpha=0.3, color='red')
    cb2 = fig.colorbar(c2, ax=axes[1])
    cb2.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    axes[1].set_xlabel('$J_x$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel('$J_y$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(f"kde_comparison{suffix}.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()

    # Compute Jensen-Shannon Divergence
    js_div = jensenshannon(pdf_weighted.ravel(), pdf_gen.ravel())
    print(f"Jensen-Shannon Divergence{(' for ' + title) if title else ''}: {js_div:.4f}")

    # --------------------- Y–Z projection ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(x=data[:, 1], y=data[:, 2], weights=weights,
                fill=True, thresh=0, levels=20, cmap='Blues', ax=axes[0])
    axes[0].set_title("Original Data (Y–Z)", fontsize=TITLE_FONTSIZE)
    axes[0].set_xlabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel('Z', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    sns.kdeplot(x=x_1_hat[:, 1], y=x_1_hat[:, 2],
                fill=True, thresh=0, levels=20, cmap='Reds', ax=axes[1])
    axes[1].set_title("Generated Data (Y–Z)", fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel('Y', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel('Z', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(f"kde_yz_comparison{suffix}.png", dpi=300)
    plt.show()

    # --------------------- X–Z projection ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(x=data[:, 0], y=data[:, 2], weights=weights,
                fill=True, thresh=0, levels=20, cmap='Blues', ax=axes[0])
    axes[0].set_title("Original Data (X–Z)", fontsize=TITLE_FONTSIZE)
    axes[0].set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel('Z', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    sns.kdeplot(x=x_1_hat[:, 0], y=x_1_hat[:, 2],
                fill=True, thresh=0, levels=20, cmap='Reds', ax=axes[1])
    axes[1].set_title("Generated Data (X–Z)", fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel('X', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel('Z', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(f"kde_xz_comparison{suffix}.png", dpi=300)
    plt.show()

    return js_div



def plot_3d_comparison(data, x_1_hat, weights, title):
    """
    Create 3D scatter plots to compare real vs generated data
    (publication styling: bigger fonts, no title)
    """
    # ---- styling knobs ----
    FIGSIZE          = (14, 10)
    AXIS_LABEL_FS    = 22   # axis names
    TICK_FS          = 18   # axis numbers
    LEGEND_FS        = 18
    POINT_SIZE       = 30
    HILO_POINT_SIZE  = 110
    ANNOTATE_FS      = 14
    LABELPAD         = 24
    # -----------------------

    fig = plt.figure(figsize=FIGSIZE)
    ax  = fig.add_subplot(111, projection='3d')

    # Original (real) data
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        s=POINT_SIZE, color='black', alpha=0.5, label='Original Data'
    )

    # Generated data
    ax.scatter(
        x_1_hat[:, 0], x_1_hat[:, 1], x_1_hat[:, 2],
        s=POINT_SIZE, color='blue', alpha=0.5, label='Generated Data'
    )

    # Highest/lowest weight highlights
    sorted_indices  = np.argsort(weights)
    lowest_indices  = sorted_indices[:3]
    highest_indices = sorted_indices[-3:]

    for idx in highest_indices:
        ax.scatter(
            data[idx, 0], data[idx, 1], data[idx, 2],
            s=HILO_POINT_SIZE, color='red', marker='^'
        )
        ax.text(
            data[idx, 0], data[idx, 1], data[idx, 2],
            f'W={weights[idx]:.4f}', color='red', fontsize=ANNOTATE_FS
        )

    for idx in lowest_indices:
        ax.scatter(
            data[idx, 0], data[idx, 1], data[idx, 2],
            s=HILO_POINT_SIZE, color='green', marker='s'
        )
        ax.text(
            data[idx, 0], data[idx, 1], data[idx, 2],
            f'W={weights[idx]:.4f}', color='green', fontsize=ANNOTATE_FS
        )

    # Axis labels (no figure title)
    ax.set_xlabel('X', fontsize=AXIS_LABEL_FS, labelpad=LABELPAD)
    ax.set_ylabel('Y', fontsize=AXIS_LABEL_FS, labelpad=LABELPAD)
    ax.set_zlabel('Z', fontsize=AXIS_LABEL_FS, labelpad=LABELPAD)

    # Tick label sizes
    ax.tick_params(axis='x', labelsize=TICK_FS, pad=10)
    ax.tick_params(axis='y', labelsize=TICK_FS, pad=10)
    ax.tick_params(axis='z', labelsize=TICK_FS, pad=10)

    # Legend
    ax.scatter([], [], [], s=HILO_POINT_SIZE, color='red',   marker='^', label='Highest Weight')
    ax.scatter([], [], [], s=HILO_POINT_SIZE, color='green', marker='s', label='Lowest Weight')
    leg = ax.legend(fontsize=LEGEND_FS, frameon=True, fancybox=True, framealpha=0.9, borderpad=0.6, loc='upper left')

    # Slightly nicer 3D proportions and tighter layout
    ax.set_box_aspect((1.2, 1.0, 0.9))
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.98)

    plt.savefig(f"3d_scatter_{title}.png", dpi=300)
    plt.show()


def train_and_evaluate(file_name, n_epochs=10000, batch_size=256, n_gen_samples=10000):
    """
    Run the complete training and evaluation pipeline for a given dataset
    """
    print(f"\n{'='*80}\nProcessing dataset: {file_name}\n{'='*80}")
    
    # Load data
    df = pd.read_excel(file_name)
    data_np = df[['Js_xx', 'Js_yy', 'Js_zz']].values
    weights_np = df['Normalized_Weight'].values
    
    # Convert to torch tensors
    data = torch.from_numpy(data_np).float().to(device)
    weights = torch.from_numpy(weights_np).float().to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = OTFlowMatching()
    net = Net(in_dim=3, out_dim=3, h_dims=[256]*5, n_frequencies=10).to(device)
    v_t = CondVF(net)
    
    # Training
    losses = []
    optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-5)
    
    for epoch in trange(n_epochs, ncols=88, desc=f"Training {file_name}"):
        epoch_losses = []
        for x_batch, w_batch in dataloader:
            loss = model.loss(v_t, x_batch, w_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Record average loss for this epoch
        losses.append(np.mean(epoch_losses))
        
        # Print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {losses[-1]:.6f}")
    
    # Generate samples
    with torch.no_grad():
        x_0 = torch.randn(n_gen_samples, 3, device=device)
        x_1_hat = v_t.decode(x_0)
    
    x_1_hat = x_1_hat.cpu().numpy()
    

    gen_df = pd.DataFrame(
        x_1_hat, columns=["Jxx_gen", "Jyy_gen", "Jzz_gen"]
    )
    out_name = file_name.replace("training_data_weights_",              # e.g.
                                 "gen10k_").replace(".xlsx", "_gen.xlsx")  # → gen10k_noise_0.005_gen.xlsx
    gen_df.to_excel(out_name, index=False)
    print(f"Generated samples saved to: {out_name}")

    data_cpu = data.cpu().numpy()
    weights_cpu = weights.cpu().numpy()
    

    
    # Create visualization plots
    dataset_name = file_name.replace("training_data_weights_", "").replace(".xlsx", "")
    plot_3d_comparison(data_cpu, x_1_hat, weights_cpu, dataset_name)
    js_div = plot_kde_comparison(data_cpu, weights_cpu, x_1_hat, dataset_name)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f"Training Loss - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"training_loss_{dataset_name}.png")
    plt.show()
    
    return {
        "dataset": dataset_name,
        "losses": losses,
        "final_loss": losses[-1],
        "js_div": js_div,
        "generated_samples": x_1_hat
    }


###############################################################################
#                           Run All Experiments                               #
############################################################################### 
# Define file names
file_names = [
    "training_data_weights_noise_0.01.xlsx",
    "training_data_weights_noise_0.001.xlsx",
    "training_data_weights_noise_0.0001.xlsx"
]

# For testing or faster runs, set to a smaller number (e.g., 1000)
n_epochs = 10000  
n_gen_samples = 700
batch_size = 256

# Run experiments for all datasets
results = []
for file_name in file_names:
    result = train_and_evaluate(
        file_name=file_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_gen_samples=n_gen_samples
    )
    results.append(result)
    

###############################################################################
#                           Compare Convergence                               #
###############################################################################
# Plot comparison of training losses
plt.figure(figsize=(12, 6))
for result in results:
    plt.plot(result["losses"], label=f"Noise: {result['dataset']}")

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig("loss_comparison.png")
plt.show()

# Create a summary bar chart of JS divergence and EMD values
noise_levels = [result["dataset"] for result in results]
js_div_values = [result["js_div"] for result in results]


x = np.arange(len(noise_levels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, js_div_values, width, label='JS Divergence')


ax.set_xlabel('Noise Level')
ax.set_ylabel('Metric Value')
ax.set_title('Convergence Metrics by Noise Level')
ax.set_xticks(x)
ax.set_xticklabels(noise_levels)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)


fig.tight_layout()
plt.savefig("convergence_metrics.png")
plt.show()

# Print summary table
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Noise Level':<15} | {'Final Loss':<15} | {'JS Divergence':<15} | {'EMD':<15}")
print("-" * 80)
for result in results:
    print(f"{result['dataset']:<15} | {result['final_loss']:<15.6f} | {result['js_div']:<15.6f} | ")
print("-" * 80)

# Plot 3D scatter comparison of all generated distributions
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue']
markers = ['o', '^', 's']

for i, result in enumerate(results):
    samples = result["generated_samples"]
    # Plot a subset of samples for clarity (1000 points)
    indices = np.random.choice(len(samples), size=1000, replace=False)
    ax.scatter(
        samples[indices, 0], 
        samples[indices, 1], 
        samples[indices, 2],
        s=30, 
        color=colors[i], 
        marker=markers[i], 
        alpha=0.7, 
        label=f"Noise: {result['dataset']}"
    )

ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
ax.set_title("Comparison of Generated Distributions", fontsize=16)
ax.legend(fontsize=12)

plt.savefig("generated_distribution_comparison.png")
plt.show()

print("\nAnalysis completed. All visualizations have been saved.")


# --------------------------------------------------------------
#  Evaluate ⟨‖Ĵ−J_true‖₂⟩   *and*   its spread for every σ
# --------------------------------------------------------------
true_J = np.array([100.0, 80.0, 70.0])   # nominal diagonal

sigma_vals, mean_dists, std_dists = [], [], []

for res in results:
    # dataset strings look like  "noise_0.008"  →  get 0.008
    sigma = float(res["dataset"].split("_")[1])

    J_samples = res["generated_samples"]            # shape (N, 3)

    # Euclidean distance for every generated sample
    d_all = np.linalg.norm(J_samples - true_J, axis=1)

    mu_d  = d_all.mean()                            # mean ‖Ĵ−J‖₂
    sd_d  = d_all.std(ddof=1)                       # std  ‖Ĵ−J‖₂

    sigma_vals.append(sigma)
    mean_dists.append(mu_d)
    std_dists.append(sd_d)

    print(f"σ = {sigma:7.4f}   →   "
          f"mean‖Ĵ−J‖₂ = {mu_d:.4f} kg·m²,   "
          f"std = {sd_d:.4f}")

# Sort by σ for tidy curves
sigma_vals, mean_dists, std_dists = zip(
    *sorted(zip(sigma_vals, mean_dists, std_dists))
)

# ------------------------------------------------------------------
# (1) Mean distance vs. σ (log–log)  — same size as before
# ------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.loglog(sigma_vals, mean_dists, '-o', lw=3, ms=8, mec='k', mfc='none')
plt.gca().invert_xaxis()  # optional: show smaller σ on the right

# Axis labels (bigger fonts)
plt.xlabel(r"Sensor noise $\sigma$ [rad s$^{-1}$]", fontsize=22, labelpad=8)
plt.ylabel(r"Difference mean [kg m$^{2}$]", fontsize=22, labelpad=10)

# Bigger tick numbers
plt.tick_params(axis='both', which='both', labelsize=20, width=1.2, length=6)

# No grid
plt.grid(False)

# Layout & save (keep figure open)
plt.tight_layout()
plt.savefig("mean_distance_vs_sigma.png", dpi=300)
plt.show()  

# ------------------------------------------------------------------
# (2) Std of distance vs. σ (log–log)  — same size as before
# ------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.loglog(sigma_vals, std_dists, 's-', lw=3, ms=8, color='darkorange', mec='k', mfc='none')
plt.gca().invert_xaxis()

plt.xlabel(r"Sensor noise $\sigma$ [rad s$^{-1}$]", fontsize=22, labelpad=8)
plt.ylabel(r"Difference variance [kg m$^{2}$]", fontsize=22, labelpad=10)

plt.tick_params(axis='both', which='both', labelsize=20, width=1.2, length=6)

# No grid
plt.grid(False)

plt.tight_layout()
plt.savefig("std_distance_vs_sigma.png", dpi=300)
plt.show()  