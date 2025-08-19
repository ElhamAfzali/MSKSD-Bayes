import torch
from dataclasses import dataclass

@dataclass
class KernelOutput:
    """Output structure for kernel computations."""
    k: torch.Tensor  # Pairwise kernel values, shape (n, m)
    kx: torch.Tensor  # Gradients w.r.t x, shape (n, m, d)
    ky: torch.Tensor  # Gradients w.r.t y, shape (n, m, d)
    kxy: torch.Tensor  # Second derivatives, shape (n, m, d, d)


class Scaled_PIMQ:
    """Pairwise Matrix-Valued Kernel with Gradients and Second Derivatives."""

    def __init__(self, S: torch.Tensor, C: float = 1.0):
        """
        Initialize the kernel with a positive definite scale matrix.

        Args:
            S: Positive definite matrix of shape (d, d).
            C: Kernel amplitude (default 1.0).
        """
        self.S = S
        self.S_inv = torch.linalg.inv(S)  # Precompute the inverse of S
        self.C = C

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> KernelOutput:
        """
        Compute the pairwise kernel values and derivatives.

        Args:
            x: Tensor of shape (n, d) representing n samples.
            y: Tensor of shape (m, d) representing m samples.

        Returns:
            KernelOutput containing pairwise kernel values and derivatives.
        """
        n, d = x.shape
        m, _ = y.shape

        # Compute pairwise differences: shape (n, m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)

        # Compute quadratic term: shape (n, m)
        quad_term = torch.einsum('nmd,dc,nmc->nm', diff, self.S_inv, diff)

        # Compute kernel values: shape (n, m)
        k = self.C * (1 + quad_term).pow(-0.5)

        # Compute first derivatives (gradients): shape (n, m, d)
        common_factor = k.pow(3).unsqueeze(-1)  # Reshape for broadcasting
        S_inv_diff = torch.einsum('dc,nmd->nmc', self.S_inv, diff)
        kx = -common_factor * S_inv_diff
        ky = -kx  # Symmetry: dk/d(y_j) = -dk/d(x_i)

        # Compute second derivatives (Hessian): shape (n, m, d, d)
        second_factor = 3  * k.pow(5).unsqueeze(-1).unsqueeze(-1)
        kxy_term1 = -second_factor * torch.einsum('nmc,nmd->nmcd',  S_inv_diff,  S_inv_diff)
        kxy_term2 = k.pow(3).unsqueeze(-1).unsqueeze(-1) * self.S_inv.unsqueeze(0).unsqueeze(0)
        kxy = kxy_term1 + kxy_term2

        return KernelOutput(k=k, kx=kx, ky=ky, kxy=kxy)


# Example Usage
if __name__ == "__main__":
    # Example data
    n, m, d = 5, 4, 3  # 5 samples in x, 4 samples in y, 3 dimensions
    x = torch.randn(n, d)
    y = torch.randn(m, d)
    S = torch.eye(d)  # Identity matrix as a simple positive definite matrix

    # Initialize and compute kernel
    kernel = Scaled_PIMQ(S)
    output = kernel(x, y)

    # Access results
    print("Pairwise Kernel Matrix (k):", output.k.shape)  # (n, m)
    print("Gradients w.r.t x (kx):", output.kx[0])     # (n, m, d)
    print("Gradients w.r.t y (ky):", output.ky.shape)     # (n, m, d)
    print("Second Derivatives (kxy):", output.kxy.shape)  # (n, m, d, d)
