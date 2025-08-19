import torch
import math


class PDF_KEF:
    def __init__(self, coeff, L):
        """
        Initialize the PDF_KEF class.

        Args:
            coeff (torch.Tensor): Coefficient vector of shape (p,) or (p, 1).
            L (float): Width of the reference measure.
        """
        # Ensure coeff is the right shape
        self.coeff = coeff.squeeze()
        self.L = L
        self.p = coeff.numel()  # Number of coefficients

    def factorial_sqrt_inv(self):
        """
        Compute the inverse square root of factorials for 0 to (p-1).

        Returns:
            torch.Tensor: A tensor of shape (p,) with values (1 / sqrt(factorial(i))).
        """
        return torch.tensor([
            1 / math.sqrt(math.factorial(i)) for i in range(self.p)
        ], dtype=torch.float32, device=self.coeff.device)

    def compute_vandermonde(self, X):
        """
        Construct the Vandermonde matrix with Gaussian weighting.

        Args:
            X (torch.Tensor): Input tensor of shape (n, 1).

        Returns:
            torch.Tensor: Vandermonde matrix of shape (n, p).
        """
        X = X.squeeze()
        powers = torch.arange(self.p, dtype=X.dtype, device=X.device)  # Shape: (p,)
        X_powers = X.unsqueeze(-1).pow(powers)  # Shape: (n, p)

        # Compute Gaussian weighting
        gaussian_weight = torch.exp(-0.5 * X.pow(2))  # Shape: (n,)

        # Vandermonde matrix with weighting
        inv_sqrt_factorials = self.factorial_sqrt_inv().to(X.device)  # Shape: (p,)
        return X_powers * inv_sqrt_factorials * gaussian_weight.unsqueeze(-1)  # Shape: (n, p)

    def compute_pdf(self, X):
        """
        Compute the normalized probability density function values.

        Args:
            X (torch.Tensor): Input tensor of shape (n, 1).

        Returns:
            torch.Tensor: Normalized PDF values of shape (n, 1).
        """
        x_vals = X.squeeze()

        # Compute Vandermonde matrix
        mat = self.compute_vandermonde(X)  # Shape: (n, p)

        # Log of unnormalized PDF
        log_pdf = -0.5 * (x_vals / self.L).pow(2) + torch.matmul(mat, self.coeff)

        # Prevent overflow
        log_pdf = log_pdf - torch.max(log_pdf)

        # Unnormalized PDF
        out = torch.exp(log_pdf)  # Shape: (n,)

        # Numerical normalization using the trapezoidal rule
        dx = x_vals[1] - x_vals[0]  # Assuming uniform grid
        integral = torch.trapz(out, dx=dx)

        return (out / integral).unsqueeze(-1)  # Return shape (n, 1)

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Load input data
    # X = pd.read_csv('galaxy.csv', header=None).to_numpy()  # Load data from CSV
    # X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)  # Ensure shape (n, 1)

    x_min = -3 * 3
    x_max = max(3 * 3, 5 + 3)
    X = torch.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # Define coefficients and scalar
    coeff = torch.tensor([0.5, -0.2, 0.1, -2.0, 0.08], dtype=torch.float32).reshape(-1, 1)  # (p, 1)
    L = 2.0  # Scalar

    # Instantiate the class and compute the PDF
    pdf_kef = PDF_KEF(coeff, L)
    pdf_vals = pdf_kef.compute_pdf(X)

    # Output results
    print("X shape:", X.shape)
    print("PDF Values shape:", pdf_vals.shape)
