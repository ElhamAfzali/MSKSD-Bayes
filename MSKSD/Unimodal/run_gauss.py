

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    from ..src.kernel import Scaled_PIMQ
    from ..src.KSD_Bayes import KSD_Bayes
    from ..src.rscm_torch import RSCMEstimator
    from .M_Gauss import MGauss
    from .Gauss import Gauss
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.kernel import Scaled_PIMQ
    from src.KSD_Bayes import KSD_Bayes
    from src.rscm_torch import RSCMEstimator
    from Unimodal.M_Gauss import MGauss
    from Unimodal.Gauss import Gauss


def compute_log_p(x, mu, sigma):
    """Compute the log probability of x under a Gaussian distribution."""
    log_p_x = -0.5 * np.log(2 * np.pi * sigma ** 2) - ((x - mu) ** 2) / (2 * sigma ** 2)
    return log_p_x

def run_experiment(weight_factor: float = 1.0):
    """Run the experiment and save the results in PyTorch format."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulation parameters
    n = 100  # number of samples
    Sig = 1.0  # measurement variance
    d = 1  # dimension

    # Set random seed for consistency
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate data
    Z1 = torch.empty(n, d).normal_(mean=1.0, std=np.sqrt(Sig))
    Z2 = torch.empty(n, d).normal_(mean=0.0, std=np.sqrt(Sig))
    rnd = torch.empty(n).uniform_()

    # Prior parameters
    Sig0 = 1.0  # prior standard deviation
    mu0 = 0.0  # prior mean
    A0 = (1 / 2) * (1 / Sig0)  # posterior precision
    v0 = -2 * A0 * mu0

    # Initialize models
    model = Gauss()
    M_nonrobust = MGauss(robust=False)

    # Experiment parameters
    eps_levels = [0.0, 0.1, 0.2]  # contamination proportions
    y_levels = [1.0, 10.0, 20.0]  # displacement values
    theta_grid = np.linspace(-0.5, 3.5, 1000)  # Adjusted range

    # Results storage
    results = {
        "weight_factor": weight_factor,
        "eps_levels": eps_levels,
        "y_levels": y_levels,
        "theta_grid": theta_grid,
        "row1": [],  # Results for varying eps (fixed y = 10)
        "row2": [],  # Results for varying y (fixed eps = 0.1)
    }

    # First row: varying contamination levels (eps) for fixed y = 10
    y = 10.0  # fixed displacement
    for eps in eps_levels:
        # Create contaminated dataset
        contaminate = (rnd < eps).float()
        X = (1 - contaminate).unsqueeze(1) * Z1 + contaminate.unsqueeze(1) * (y + Z2)

        # Compute S using RSCM
        rscm = RSCMEstimator()
        S = rscm.fit(X, approach='ell1')
        kernel = Scaled_PIMQ(S=S)

        # Standard Bayes posterior
        Sign_Bayes = 1 / ((1 / Sig0) + (n / Sig))
        mun_Bayes = Sign_Bayes * ((mu0 / Sig0) + (X.sum() / Sig))

        # Compute log probabilities for weighted KSD
        log_p = torch.tensor(compute_log_p(X.numpy(), mu0, Sig0), dtype=torch.float32, device=device)

        # KSD-Bayes (Non-Weighted, Non-Robust)
        ksd_result_nonrobust = KSD_Bayes(
            X=X,
            grad_T=model.grad_T,
            grad_b=model.grad_b,
            M=M_nonrobust,
            K=kernel,
            log_p=None,  # Non-weighted
            weighted=False,
            weight_factor=weight_factor,
        )
        beta = min(1.0, ksd_result_nonrobust.w)
        print('Beta:', beta)
        An_nonrobust = A0 + beta * ksd_result_nonrobust.An
        vn_nonrobust = v0 + beta * ksd_result_nonrobust.vn
        Sign_KSD_Bayes = (1 / 2) * torch.inverse(An_nonrobust).item()
        mun_KSD_Bayes = -(1 / 2) * torch.linalg.solve(An_nonrobust, vn_nonrobust).item()

        # MSKSD-Bayes (Weighted, Non-Robust)
        ksd_result_weighted = KSD_Bayes(
            X=X,
            grad_T=model.grad_T,
            grad_b=model.grad_b,
            M=M_nonrobust,
            K=kernel,
            log_p=log_p,  # Weighted
            weighted=True,
            weight_factor=weight_factor,
        )
        beta = min(1.0, ksd_result_weighted.w)
        print('Beta:', beta)
        An_weighted = A0 + beta * ksd_result_weighted.An
        vn_weighted = v0 + beta * ksd_result_weighted.vn
        Sign_MSKSD_Bayes = (1 / 2) * torch.inverse(An_weighted).item()
        mun_MSKSD_Bayes = -(1 / 2) * torch.linalg.solve(An_weighted, vn_weighted).item()

        # Save results for this eps
        results["row1"].append({
            "eps": eps,
            "mun_Bayes": mun_Bayes.item(),
            "Sign_Bayes": Sign_Bayes,
            "mun_KSD_Bayes": mun_KSD_Bayes,
            "Sign_KSD_Bayes": Sign_KSD_Bayes,
            "mun_MSKSD_Bayes": mun_MSKSD_Bayes,
            "Sign_MSKSD_Bayes": Sign_MSKSD_Bayes,
        })

    # Second row: varying displacements (y) for fixed eps = 0.1
    eps = 0.1  # fixed contamination
    for y in y_levels:
        # Create contaminated dataset
        contaminate = (rnd < eps).float()
        X = (1 - contaminate).unsqueeze(1) * Z1 + contaminate.unsqueeze(1) * (y + Z2)

        # Compute S using RSCM
        rscm = RSCMEstimator()
        S = rscm.fit(X, approach='ell1')
        kernel = Scaled_PIMQ(S=S)

        # Standard Bayes posterior
        Sign_Bayes = 1 / ((1 / Sig0) + (n / Sig))
        mun_Bayes = Sign_Bayes * ((mu0 / Sig0) + (X.sum() / Sig))

        # Compute log probabilities for weighted KSD
        log_p = torch.tensor(compute_log_p(X.numpy(), mu0, Sig0), dtype=torch.float32, device=device)

        # KSD-Bayes (Non-Weighted, Non-Robust)
        ksd_result_nonrobust = KSD_Bayes(
            X=X,
            grad_T=model.grad_T,
            grad_b=model.grad_b,
            M=M_nonrobust,
            K=kernel,
            log_p=None,  # Non-weighted
            weighted=False,
            weight_factor=weight_factor,
        )
        beta = min(1.0, ksd_result_nonrobust.w)
        print('Beta:', beta)
        An_nonrobust = A0 + beta * ksd_result_nonrobust.An
        vn_nonrobust = v0 + beta * ksd_result_nonrobust.vn
        Sign_KSD_Bayes = (1 / 2) * torch.inverse(An_nonrobust).item()
        mun_KSD_Bayes = -(1 / 2) * torch.linalg.solve(An_nonrobust, vn_nonrobust).item()

        # MSKSD-Bayes (Weighted, Non-Robust)
        ksd_result_weighted = KSD_Bayes(
            X=X,
            grad_T=model.grad_T,
            grad_b=model.grad_b,
            M=M_nonrobust,
            K=kernel,
            log_p=log_p,  # Weighted
            weighted=True,
            weight_factor=weight_factor,
        )
        beta = min(1.0, ksd_result_weighted.w)
        print('Beta:', beta)
        An_weighted = A0 + beta * ksd_result_weighted.An
        vn_weighted = v0 + beta * ksd_result_weighted.vn
        Sign_MSKSD_Bayes = (1 / 2) * torch.inverse(An_weighted).item()
        mun_MSKSD_Bayes = -(1 / 2) * torch.linalg.solve(An_weighted, vn_weighted).item()
        print('MSKSD Bayes_mu:', mun_MSKSD_Bayes)

        # Save results for this y
        results["row2"].append({
            "y": y,
            "mun_Bayes": mun_Bayes.item(),
            "Sign_Bayes": Sign_Bayes,
            "mun_KSD_Bayes": mun_KSD_Bayes,
            "Sign_KSD_Bayes": Sign_KSD_Bayes,
            "mun_MSKSD_Bayes": mun_MSKSD_Bayes,
            "Sign_MSKSD_Bayes": Sign_MSKSD_Bayes,
        })

    # Save results to a file in PyTorch format
    filename = f"experiment_results_weight_factor_{weight_factor}.pt"
    torch.save(results, filename)

    print(f"Experiment completed and results saved to {filename}.")


def plot_results(weight_factor: float = 1.0):
    """Load the results and generate high-quality plots for a statistical journal."""
    # Load results from file
    filename = f"experiment_results_weight_factor_{weight_factor}.pt"
    results = torch.load(filename)

    # Setup for plots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']  # Modern and visually distinct colors
    styles = ['-', '--', ':']
    theta_grid = results["theta_grid"]

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'lines.linewidth': 2.0,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
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

    # First row: varying contamination levels (eps) for fixed y = 10
    for i, res in enumerate(results["row1"]):
        axs[0, 0].plot(theta_grid, norm.pdf(theta_grid, res["mun_Bayes"], np.sqrt(res["Sign_Bayes"])),
                       linestyle=styles[i], color=colors[i], label=f'$\epsilon = {res["eps"]}$')
        axs[0, 1].plot(theta_grid, norm.pdf(theta_grid, res["mun_KSD_Bayes"], np.sqrt(res["Sign_KSD_Bayes"])),
                       linestyle=styles[i], color=colors[i])
        axs[0, 2].plot(theta_grid, norm.pdf(theta_grid, res["mun_MSKSD_Bayes"], np.sqrt(res["Sign_MSKSD_Bayes"])),
                       linestyle=styles[i], color=colors[i])

    # Second row: varying displacements (y) for fixed eps = 0.1
    for i, res in enumerate(results["row2"]):
        axs[1, 0].plot(theta_grid, norm.pdf(theta_grid, res["mun_Bayes"], np.sqrt(res["Sign_Bayes"])),
                       linestyle=styles[i], color=colors[i], label=f'$y = {res["y"]}$')
        axs[1, 1].plot(theta_grid, norm.pdf(theta_grid, res["mun_KSD_Bayes"], np.sqrt(res["Sign_KSD_Bayes"])),
                       linestyle=styles[i], color=colors[i])
        axs[1, 2].plot(theta_grid, norm.pdf(theta_grid, res["mun_MSKSD_Bayes"], np.sqrt(res["Sign_MSKSD_Bayes"])),
                       linestyle=styles[i], color=colors[i])

    # Configure plots
    titles = [r'Bayesian Posterior', r'KSD-Bayes', r'MSKSD-Bayes']
    for i in range(3):
        for j in range(2):
            ax = axs[j, i]
            ax.set_ylim(0, 5)
            ax.set_xlim(0.0, 3.5)
            ax.grid(True, linestyle=':', alpha=0.7)
            if j == 1:
                ax.set_xlabel(r'$\theta$', fontsize=14)
            if i == 0 and j == 0:
                ax.set_ylabel('Generalized Posterior\n(Varying $\epsilon$, $y=10$)', fontsize=12)
            if i == 0 and j == 1:
                ax.set_ylabel('Generalized Posterior\n(Varying $y$, $\\epsilon=0.1$)', fontsize=12)
            if j == 0:
                ax.set_title(titles[i], fontsize=16)

    # Add a single legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles2, labels2 = axs[1, 0].get_legend_handles_labels()
    axs[0, 2].legend(handles, labels, loc='upper right', fontsize=12, frameon=True, shadow=True)
    axs[1, 2].legend(handles2, labels2, loc='upper right', fontsize=12, frameon=True, shadow=True)

    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0, rect=[0.02, 0.02, 0.98, 0.98])
    save_path = f"experiment_results_plot_weight_factor_{weight_factor}.pdf"
    plt.savefig(save_path, dpi=1200, bbox_inches="tight", format='pdf')  # High-resolution PDF
    print(f"Plot saved as {save_path}")
    plt.show()


if __name__ == "__main__":
    # # Run the experiment with a specific weight_factor
    weight_factor = 1.0 # Example weight factor
    run_experiment(weight_factor=weight_factor)

    # Plot the results
    plot_results(weight_factor=weight_factor)
