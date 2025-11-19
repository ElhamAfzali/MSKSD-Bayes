import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import Dict
import math

try:
    from ..run_kef import run_KEF
    from ..nearestSPD import nearestSPD
    from ..pdf_KEF import PDF_KEF
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from msksd.Bimodal.run_kef import run_KEF
    from msksd.Bimodal.nearestSPD import nearestSPD
    from msksd.Bimodal.pdf_KEF import PDF_KEF

###############################################################################
#                           Configuration & Prior                             #
###############################################################################
@dataclass
class ExperimentConfig:
    """Configuration for KEF0 experiments (no contamination)."""
    p: int = 25          # number of basis functions
    L: float = 3.0       # width of reference measure
    seed: int = 0        # random seed
    prior_scale: float = 100.0   # scale for prior covariance
    prior_decay: float = 1.1    # decay rate for prior covariance
    output_dir: Path = field(default_factory=lambda: Path("gene_results"))

    def get_result_path(self) -> Path:
        """Get path for saving results with parameters in filename."""
        params = f"p{self.p}_L{self.L}"
        return self.output_dir / f"kef_results_{params}_prior_scale{self.prior_scale}_prior_decay{self.prior_decay}.pt"


def initialize_prior(config: ExperimentConfig) -> Dict[str, torch.Tensor]:
    """Initialize prior parameters based on configuration."""
    Sig0 = config.prior_scale * torch.diag(torch.tensor(
        [(i + 1) ** (-config.prior_decay) for i in range(config.p)]
    ))
    mu0 = torch.zeros(config.p, 1)
    A0 = (1 / 2) * torch.linalg.inv(Sig0)
    v0 = -2 * A0 @ mu0
    theta0 = -(1 / 2) * torch.linalg.solve(A0, v0)

    return {
        'Sig0': Sig0,
        'mu0': mu0,
        'A0': A0,
        'v0': v0,
        'theta0': theta0
    }

###############################################################################
#                           Basis & Log-p Functions                           #
###############################################################################
def phi_Tx(x: torch.Tensor, max_i: int) -> torch.Tensor:
    """Compute phi(x) as a vector of basis functions for x."""
    phi_values = []
    for i in range(max_i):
        factorial_i = math.sqrt(math.factorial(i))
        phi_i = (x ** i) / factorial_i * torch.exp(-x ** 2 / 2)
        phi_values.append(phi_i)
    return torch.stack(phi_values, dim=1)


def b_x(x: torch.Tensor, L: float) -> torch.Tensor:
    """Compute b(x) = -x^2 / (2 * L^2)."""
    if L == 0:
        raise ValueError("Parameter L must be non-zero")
    b = - (x ** 2) / (2 * L ** 2)
    return b

def compute_log_p(x: torch.Tensor, theta: torch.Tensor, max_i: int, L: float) -> torch.Tensor:
    """Compute log p(x) = theta * phi(x) + b(x)."""
    phi_values = phi_Tx(x, max_i)
    b_values = b_x(x, L)
    return torch.matmul(phi_values.squeeze(-1), theta) + b_values


def setup_plotting():
    """Configure matplotlib settings for a clean plot."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': True,
        'figure.figsize': (7.5, 6.5),
        'lines.linewidth': 2.0,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.edgecolor': 'black',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

def plot_histogram(ax, X_data, binwidth=0.5):
    """Plot histogram of the data with professional color scheme."""
    data_np = X_data.numpy().flatten()
    data_range = np.max(data_np) - np.min(data_np)
    bins = int(np.ceil(data_range / binwidth))

    counts, bin_edges = np.histogram(data_np, bins=bins,
                                     range=(np.min(data_np), np.max(data_np)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    color = '#4DBBD5'  # blue
    edge_color = 'black'

    ax.bar(bin_centers, counts, width=bin_width,
           color=color, alpha=0.8,
           label='Observed Data', edgecolor=edge_color,
           linewidth=1.0)

    ax.set_xlim([0, 14])
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_ylim([0, 200])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_yticklabels(['0', '50', '100', '150', '200'])
    ax.grid(True, linestyle='--', alpha=0.5, color='#666666')
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.set_xlabel('Gene Expression(log2 scale)', fontsize=12)


def plot_kef_posterior(ax,
                       X_original: torch.Tensor,
                       mun: torch.Tensor,
                       Sign: torch.Tensor,
                       center: float,
                       scale: float,
                       config: ExperimentConfig,
                       n_samples: int = 100,
                       title: str = "",
                       show_xlabel: bool = True):
    """
    Plots the posterior samples and posterior mean for a single KEF0 run.
    """

    sample_color = '#9A8FBF'  # purple
    mean_color = '#191970'
    torch.manual_seed(0)

    X_grid = torch.linspace(-3 * config.L, 3 * config.L, 1000).reshape(-1, 1)

    for i in range(n_samples):
        coeff = torch.distributions.MultivariateNormal(
            mun.squeeze(), Sign
        ).sample().unsqueeze(-1)
        pdf_estimator = PDF_KEF(coeff, config.L)
        pdf_vals = pdf_estimator.compute_pdf(X_grid)
        if i == 0:
            ax.plot(center + scale * X_grid,
                    pdf_vals / scale,
                    color=sample_color, linestyle='-', linewidth=0.85, alpha=0.4,
                    label="Samples")
        else:
            ax.plot(center + scale * X_grid,
                    pdf_vals / scale,
                    color=sample_color, linestyle='-', linewidth=0.85, alpha=0.4)

    # Posterior mean curve
    pdf_estimator = PDF_KEF(mun, config.L)
    pdf_vals = pdf_estimator.compute_pdf(X_grid)
    ax.plot(center + scale * X_grid,
            pdf_vals / scale,
            color=mean_color, linewidth=2.0,
            label='Mean')

    ax.set_xlim([0, 14])
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_title(title, fontsize=12, pad=5)
    ax.set_ylabel("Generalized Posterior", fontsize=12, weight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5, color='#666666')

def plot_results(all_results: Dict[str, Dict], n_samples: int = 100):
    """
    Creates a single figure with 1 row × 3 columns:
       - Col 1: KSD-Bayes (non-weighted, non-robust)
       - Col 2: MSKSD-Bayes (non-weighted, robust)
       - Col 3: Data histogram
    """
    setup_plotting()

    # Extract config & baseline info
    first_key = next(iter(all_results))
    config = all_results[first_key]['config']

    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)  # Adjust spacing

    # Extract results
    nonrobust_results = all_results['nonrobust_nonweighted']
    robust_results = all_results['robust_nonweighted']
    center = nonrobust_results['data']['center']
    scale = nonrobust_results['data']['scale']
    X_data = nonrobust_results['X']

    # Column 1: KSD-Bayes (non-weighted, non-robust)
    plot_kef_posterior(axes[0],
                       X_data,
                       nonrobust_results['mun'],
                       nonrobust_results['Sign'],
                       center,
                       scale,
                       config,
                       n_samples=n_samples,
                       title="KSD-Bayes",
                       show_xlabel=True)

    # Column 2: MSKSD-Bayes (non-weighted, robust)
    plot_kef_posterior(axes[1],
                       X_data,
                       robust_results['mun'],
                       robust_results['Sign'],
                       center,
                       scale,
                       config,
                       n_samples=n_samples,
                       title="MSKSD-Bayes",
                       show_xlabel=True)

    # Column 3: Histogram
    plot_histogram(axes[2], center + scale * X_data)
    axes[2].set_title("Observed Data Histogram", fontsize=12, pad=5)

    # Add x-axis label "Theta" to the left and middle subplots
    axes[0].set_xlabel(r"$\theta$", fontsize=12, weight='bold')
    axes[1].set_xlabel(r"$\theta$", fontsize=12, weight='bold')

    # Save plot
    plot_path = config.output_dir / f"kef_results_plot_p{config.p}_L{config.L}_prior_scale{config.prior_scale}_prior_decay{config.prior_decay}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=1200, transparent=False)
    plt.close(fig)

###############################################################################
#                         Main Experiment (No Contamination)                  #
###############################################################################
def run_single_configuration(config: ExperimentConfig,
                             X_data: torch.Tensor,
                             prior: Dict,
                             robust: bool) -> Dict:
    """
    Run KEF0 for a single configuration of robust/weighted parameters
    using original data only (X_data).
    """
    # Run KEF0 (non-weighted)
    kef_result = run_KEF(
        X=X_data,
        p=config.p,
        L=config.L,
        robust=robust,
        weighted=False,
        log_p=None
    )

    # Combine with prior
    An = prior['A0'] + kef_result.An
    vn = prior['v0'] + kef_result.vn

    Sign = (1 / 2) * nearestSPD(torch.linalg.inv(An))
    mun = -(1 / 2) * torch.linalg.solve(An, vn)

    return {
        'Sign': Sign,
        'mun': mun
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Dict]:
    """Run the complete KEF0 experiment (robust & non-robust, non-weighted only) with original data only."""
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    dataset = pd.read_csv("gene.csv")
    X_data = torch.tensor(dataset.values, dtype=torch.float32)
    X_data = X_data[:, 0:1]

    # Center and scale
    center = torch.mean(X_data)
    scale = torch.std(X_data)
    Z1 = (X_data - center) / scale  # Original data (standardized)

    # Initialize prior
    prior = initialize_prior(config)

    # Dictionary to store results for each setting
    all_results = {}

    # Run non-weighted experiments (robust and non-robust)
    for robust in [True, False]:
        print(f"Running combination => robust={robust}, weighted=False")

        key = f"{'robust' if robust else 'nonrobust'}_nonweighted"

        # Run KEF0
        kef_results = run_single_configuration(
            config, Z1, prior, robust
        )

        # Store results
        all_results[key] = {
            'config': config,
            'data': {
                'center': center.item(),
                'scale': scale.item()
            },
            'X': Z1,                    # The standardized original data
            'Sign': kef_results['Sign'],
            'mun': kef_results['mun']
        }

    # Save results to a single file
    result_path = config.get_result_path()
    torch.save(all_results, result_path)

    return all_results

###############################################################################
#                               Script Entry                                  #
###############################################################################
if __name__ == "__main__":
    # Example usage:
    config = ExperimentConfig()
    all_results = run_experiment(config)
    plot_results(all_results)
    print(f"Results and plot saved to: {config.output_dir}")
