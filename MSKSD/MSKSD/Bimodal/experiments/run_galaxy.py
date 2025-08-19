import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import List, Dict
import math
from Bimodal.run_kef import run_KEF
from Bimodal.nearestSPD import nearestSPD


@dataclass
class ExperimentConfig:
    """Configuration for KEF0 experiments."""
    p: int = 25  # number of basis functions
    L: float = 3.0  # width of reference measure
    D: float = 5.0  # location for displaced data
    eps_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    seed: int = 0  # random seed
    prior_scale: float = 100.0  # scale for prior covariance
    prior_decay: float = 1.1  # decay rate for prior covariance
    output_dir: Path = field(default_factory=lambda: Path("galaxy_results"))

    def get_result_path(self) -> Path:
        """Get path for saving results with parameters in filename."""
        params = f"p{self.p}_L{self.L}_D{self.D}"
        eps_str = '-'.join(map(str, self.eps_levels))

        return self.output_dir / f"kef_galaxy_results_{params}_eps{eps_str}.pt"


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


# Basis functions
def phi_Tx(x: torch.Tensor, max_i: int) -> torch.Tensor:
    """Compute phi(x) as a vector of basis functions for x."""
    phi_values = []
    for i in range(max_i):
        factorial_i = math.sqrt(math.factorial(i))
        phi_i = ((x ** i) / factorial_i) * torch.exp(-x ** 2 / 2)
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
    """Configure matplotlib settings for ICML paper."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'font.weight': 'bold',  # Added bold font weight
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.labelweight': 'bold',  # Bold axis labels
        'axes.titleweight': 'bold',  # Bold title
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


def plot_histogram(ax, X_cont, X_orig, binwidth=1000, eps=0.0):
    """Plot histogram with professional color scheme."""
    # Combine data to calculate range
    all_data = np.concatenate([X_cont.numpy(), X_orig.numpy()])
    data_range = np.max(all_data) - np.min(all_data)

    # Calculate the number of bins based on binwidth
    bins = int(np.ceil(data_range / binwidth))

    # Compute histogram for contaminated and original data
    counts_cont, bin_edges = np.histogram(X_cont.numpy(), bins=bins, range=(np.min(all_data), np.max(all_data)))
    counts_orig, _ = np.histogram(X_orig.numpy(), bins=bin_edges)

    # Calculate bin centers and width
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Professional color scheme
    contaminated_color = '#E64B35'  # Red for contaminated data
    original_color = '#4DBBD5'  # Blue for original data
    edge_color = 'black'  # Dark gray for edges

    # Plot histograms with enhanced styling
    ax.bar(bin_centers, counts_cont, width=bin_width,
           color=contaminated_color, alpha=0.6,
           label='contaminated', edgecolor=edge_color,
           linewidth=1.0)
    ax.bar(bin_centers, counts_orig, width=bin_width,
           color=original_color, alpha=0.6,
           label='original', edgecolor=edge_color,
           linewidth=1.0)

    # Customize plot appearance
    ax.set_xlim([0, 50000])
    ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
    ax.set_ylim([0, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(['0', '5', '10', '15', '20'])

    if eps == 0:
        ax.set_ylabel("Frequency", fontsize=12)


def plot_results(all_results: Dict[str, Dict], n_samples: int = 100):
    """Plot results with enhanced professional styling."""
    setup_plotting()

    # Get common parameters
    first_result = next(iter(all_results.values()))
    config = first_result['config']
    center = first_result['data']['center']
    scale = first_result['data']['scale']
    eps_levels = config.eps_levels

    # Create subplot grid with better spacing
    fig, axes = plt.subplots(3, len(eps_levels), figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.88, bottom=0.1)

    X_grid = torch.linspace(-3 * config.L,
                            max(3 * config.L, config.D + 3),
                            1000).reshape(-1, 1)

    # Row titles with better positioning
    row_titles = ['Dataset', 'KSD-Bayes', 'MSKSD-Bayes']
    middle_col = len(eps_levels) // 2

    y_positions = [1.15, 1.05, 1.05]  # Different y-coordinates for each row

    for row in range(3):
        axes[row, middle_col].text(
            0.5, y_positions[row],  # Use different y-coordinates for each row
            row_titles[row],
            ha='center',
            va='bottom',
            transform=axes[row, middle_col].transAxes,
            fontsize=14,
            weight='bold'
        )

    # Plot data histograms
    for i, eps in enumerate(eps_levels):
        ax = axes[0, i]
        mask = first_result['contaminate'][i].squeeze()
        X_cont = first_result['X'][i][mask]
        X_orig = first_result['X'][i][~mask]

        X_cont_rescaled = center + scale * X_cont
        X_orig_rescaled = center + scale * X_orig

        plot_histogram(ax, X_cont_rescaled, X_orig_rescaled, eps=eps)
        ax.set_title(f'$\\varepsilon = {eps}$', pad=4)

    # Professional colors for mean and samples
    mean_color = '#3C5488'  # Deep blue for mean
    sample_color = '#00A087'  # Teal for samples

    # Plot results for KSD-Bayes (non-robust)
    results = all_results['nonrobust_nonweighted']
    for i, eps in enumerate(eps_levels):
        ax = axes[1, i]

        mun = results['mun'][i]
        Sign = results['Sign'][i]

        # Plot samples first (background)
        from Bimodal.pdf_KEF import PDF_KEF
        for _ in range(n_samples):
            coeff = torch.distributions.MultivariateNormal(
                mun.squeeze(), Sign
            ).sample().unsqueeze(-1)
            pdf_estimator = PDF_KEF(coeff, config.L)
            pdf_vals = pdf_estimator.compute_pdf(X_grid)
            ax.plot(center + scale * X_grid, pdf_vals / scale,
                    color=sample_color, linestyle='-', alpha=0.2)

        # Plot mean (foreground)
        pdf_estimator = PDF_KEF(mun, config.L)
        pdf_vals = pdf_estimator.compute_pdf(X_grid)
        ax.plot(center + scale * X_grid, pdf_vals / scale,
                color=mean_color, linewidth=2.0)

        # Customize axes
        ax.set_xlim([0, 50000])
        ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
        ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
        ax.set_ylim([0, 0.00030])
        ax.set_yticks([0, 0.00010, 0.00020, 0.00030])
        ax.set_yticklabels(['0', '1', '2', '3'])

        if i == 0:
            ax.set_ylabel("Generalized Posterior ($\\times 10^{-4}$)", fontsize=11)
        if row == 2:
            ax.set_xlabel("Velocity (km/sec) ($\\times 10^{4}$)", fontsize=11)

    # Plot results for MSKSD-Bayes (robust)
    results = all_results['robust_nonweighted']
    for i, eps in enumerate(eps_levels):
        ax = axes[2, i]

        mun = results['mun'][i]
        Sign = results['Sign'][i]

        # Plot samples first (background)
        from Bimodal.pdf_KEF import PDF_KEF
        for _ in range(n_samples):
            coeff = torch.distributions.MultivariateNormal(
                mun.squeeze(), Sign
            ).sample().unsqueeze(-1)
            pdf_estimator = PDF_KEF(coeff, config.L)
            pdf_vals = pdf_estimator.compute_pdf(X_grid)
            ax.plot(center + scale * X_grid, pdf_vals / scale,
                    color=sample_color, linestyle='-', alpha=0.2)

        # Plot mean (foreground)
        pdf_estimator = PDF_KEF(mun, config.L)
        pdf_vals = pdf_estimator.compute_pdf(X_grid)
        ax.plot(center + scale * X_grid, pdf_vals / scale,
                color=mean_color, linewidth=2.0)

        # Customize axes
        ax.set_xlim([0, 50000])
        ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
        ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
        ax.set_ylim([0, 0.00030])
        ax.set_yticks([0, 0.00010, 0.00020, 0.00030])
        ax.set_yticklabels(['0', '1', '2', '3'])

        if i == 0:
            ax.set_ylabel("Generalized Posterior ($\\times 10^{-4}$)", fontsize=11)
        if row == 2:
            ax.set_xlabel("Velocity (km/sec) ($\\times 10^{4}$)", fontsize=11)

    # Enhanced legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#E64B35', ec='#333333', alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, fc='#4DBBD5', ec='#333333', alpha=0.6),
        plt.Line2D([0], [0], color=mean_color, linewidth=2.0),
        plt.Line2D([0], [0], color=sample_color, alpha=0.4)
    ]

    fig.legend(
        legend_elements,
        ['Contaminated', 'Original', 'Mean', 'Samples'],
        bbox_to_anchor=(0.5, 0.02),
        loc='center',
        ncol=4,
        frameon=True,
        fontsize=10,
        borderaxespad=1.5
    )

    # Save plot
    plot_path = config.output_dir / f"kef_galaxy_results_plot_p{config.p}_L{config.L}_D{config.D}_eps{'-'.join(map(str, eps_levels))}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.close()


def run_single_configuration(config: ExperimentConfig, X_contaminated: torch.Tensor,
                             contaminate: torch.Tensor, prior: Dict,
                             robust: bool) -> Dict:
    """Run KEF0 for a single configuration of robust/weighted parameters."""

    # Run KEF0
    kef_result = run_KEF(
        X=X_contaminated,
        p=config.p,
        L=config.L,
        robust=robust,
        weighted=False,
        log_p=None
    )

    # Compute posterior
    An = prior['A0'] + kef_result.An
    vn = prior['v0'] + kef_result.vn

    Sign = (1 / 2) * nearestSPD(torch.linalg.inv(An))
    mun = -(1 / 2) * torch.linalg.solve(An, vn)

    return {
        'Sign': Sign,
        'mun': mun
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Dict]:
    """Run the complete KEF0 experiment with all combinations."""
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    dataset = pd.read_csv("galaxy.csv", header=None)
    X_data = torch.tensor(dataset.values, dtype=torch.float32)

    # Center and scale
    center = torch.mean(X_data)
    scale = torch.std(X_data)
    Z1 = (X_data - center) / scale

    # Create displaced data
    n = X_data.shape[0]
    Z2 = config.D + 0.1 * torch.randn(n, 1)
    rnd = torch.rand(n, 1)

    # Initialize prior
    prior = initialize_prior(config)

    # Store all results
    all_results = {}

    # Initialize dictionaries to store results for robust and non-robust cases
    robust_results = {}
    nonrobust_results = {}

    for robust in [True, False]:
        # Storage for this configuration
        results = {
            'config': config,
            'data': {
                'center': center.item(),
                'scale': scale.item()
            },
            'X': [],
            'contaminate': [],
            'Sign': [],
            'mun': []
        }

        # Process each contamination level
        for eps in config.eps_levels:
            print(f"Processing contamination level ε = {eps} "
                  f"(robust={robust})")

            # Create contaminated data
            contaminate = (rnd < eps)
            X_contaminated = torch.where(contaminate, Z2, Z1)

            # Store data
            results['X'].append(X_contaminated)
            results['contaminate'].append(contaminate)

            # Run KEF0 with current configuration
            kef_results = run_single_configuration(
                config, X_contaminated, contaminate, prior, robust
            )

            results['Sign'].append(kef_results['Sign'])
            results['mun'].append(kef_results['mun'])

        # Store in all_results dictionary
        config_key = f"{'robust' if robust else 'nonrobust'}_nonweighted"
        all_results[config_key] = results

    # Save results to a single file
    result_path = config.get_result_path()
    torch.save(all_results, result_path)

    return all_results


if __name__ == "__main__":
    # Run experiment
    config = ExperimentConfig()
    # all_results = run_experiment(config)
    all_results = torch.load('galaxy_results/kef_galaxy_results_p25_L3.0_D5.0_eps0.0-0.1-0.2.pt')

    # Plot all results
    plot_results(all_results)
    print(f"Results saved to: {config.output_dir}")
