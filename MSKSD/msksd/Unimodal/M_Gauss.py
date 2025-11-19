from dataclasses import dataclass
import torch


@dataclass
class MGaussOutput:
    m: torch.Tensor
    mx: torch.Tensor


@dataclass
class MGauss:
    """
    Preconditioner matrix M(x) for use in the Gaussian location model.
    """
    robust: bool = False

    def __call__(self, x: torch.Tensor) -> MGaussOutput:
        """
        Compute preconditioner matrix and its derivatives.

        Args:
            x: Input tensor of shape (n, d)

        Returns:
            MGaussOutput containing:
                m: tensor of shape (n, d)
                mx: tensor of shape (n, d, d)
        """
        n, d = x.shape

        if not self.robust:
            m = torch.ones_like(x)  # Use ones_like for device consistency
            mx = torch.zeros(n, d, d, device=x.device)
        else:
            m = (1 + x.pow(2)).pow(-0.5)
            factor = -x * (1 + x.pow(2)).pow(-1.5)  # Numerically stable derivative

            mx = torch.zeros(n, d, d, device=x.device)
            diag_indices = torch.arange(d, device=x.device)
            mx[:, diag_indices, diag_indices] = factor

        return MGaussOutput(m, mx)


# Example usage
if __name__ == "__main__":
    x = torch.tensor([[-2.0], [0.2], [1.5]])  # Example input

    # Identity version
    m_gauss_identity = MGauss(robust=False)
    out_identity = m_gauss_identity(x)
    print("[IDENTITY] m:\n", out_identity.m)
    print("[IDENTITY] mx:\n", out_identity.mx)

    # Robust version
    m_gauss_robust = MGauss(robust=True)
    out_robust = m_gauss_robust(x)
    print("\n[ROBUST] m:\n", out_robust.m)
    print("[ROBUST] mx:\n", out_robust.mx)