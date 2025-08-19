# import torch
# from dataclasses import dataclass
#
# @dataclass
# class MKEFOutput:
#     m: torch.Tensor
#     mx: torch.Tensor
#
# class MKEF:
#     def __init__(self, robust: bool = False, C: float = 1.0):
#         self.robust = robust
#         self.C = C
#
#     def __call__(self, x: torch.Tensor) -> MKEFOutput:
#         """
#         Applies either an identity (non-robust) or robust (coordinate-wise) preconditioner
#         to x, returning:
#           - m: (n, d) tensor
#           - mx: (n, d, d) tensor (derivatives on diagonal).
#         """
#         n, d = x.shape
#
#         if not self.robust:
#             # (1) NON-ROBUST: Identity
#             m = torch.ones_like(x)                      # shape (n, d)
#             mx = torch.zeros(n, d, d, device=x.device)  # shape (n, d, d)
#         else:
#             # (2) ROBUST: M_i(x) = (1 + C*x_i^2)^(-1/2)
#             m = (1.0 + self.C * x.pow(2)).pow(-0.5)  # shape (n, d)
#
#             # derivative: d/dx_i ( (1 + C*x_i^2)^(-1/2) ) = -C*x_i*(1 + C*x_i^2)^(-3/2)
#             factor = -self.C * x * (1.0 + self.C * x.pow(2)).pow(-1.5)
#
#             # Fill the diagonal of mx with 'factor' for each sample
#             mx = torch.zeros(n, d, d, device=x.device)
#             diag_indices = torch.arange(d, device=x.device)
#             mx[:, diag_indices, diag_indices] = factor
#
#         return MKEFOutput(m, mx)

import torch
from dataclasses import dataclass

@dataclass
class MKEFOutput:
    m: torch.Tensor
    mx: torch.Tensor

class MKEF:
    def __init__(self, max_i, L, robust: bool = False, epsilon: float = 1e-8, ):
        """
        MKEF preconditioner with exponential decay.

        Args:
            robust (bool): Whether to use the robust preconditioner.
            epsilon (float): Small constant to avoid division by zero.
        """
        self.robust = robust
        self.epsilon = epsilon
        self.max_i = max_i
        self.L = L

    def __call__(self, x: torch.Tensor, comput_px) -> MKEFOutput:
        """
        Applies the preconditioner M(X) = 1 / (p(x) + epsilon).

        Args:
            x (torch.Tensor): Input data of shape (n, d).
            p_x (torch.Tensor): Probability density values of shape (n, 1).

        Returns:
            MKEFOutput: m (n, d) and mx (n, d, d).
        """
        n, d = x.shape

        if not self.robust:
            # (1) NON-ROBUST: Identity
            m = torch.ones_like(x)                      # shape (n, d)
            mx = torch.zeros(n, d, d, device=x.device)  # shape (n, d, d)
        else:
            p_x, grad_p_x = comput_px
            # (2) ROBUST: M_i(x) = 1 / (p(x) + epsilon)
            m = 1.0 / (p_x + self.epsilon)  # shape (n, 1)

            # Derivative of M_i(x): d/dx_i (1 / (p(x) + epsilon)) = -p'(x) / (p(x) + epsilon)^2
            # Assuming p_x is the probability density, we need to compute its gradient.
            # For simplicity, let's assume p_x is already the probability density and its gradient is provided.
            # In practice, you would need to compute the gradient of p(x) with respect to x.
            # Here, we assume p_x is a constant for simplicity, so the derivative is zero.
            # If you have the gradient of p(x), you can replace the following line with the actual gradient.
            p_x_grad = grad_p_x # Placeholder for gradient of p(x)

            # Compute the derivative of M_i(x)
            factor = -p_x_grad / (p_x + self.epsilon).pow(2)  # Shape: (n, d)
            # factor = -(2 * p_x) / (p_x.pow(2) + self.epsilon).pow(2)

            # Fill the diagonal of mx with 'factor' for each sample
            mx = torch.zeros(n, d, d, device=x.device)
            diag_indices = torch.arange(d, device=x.device)
            mx[:, diag_indices, diag_indices] = factor

        return MKEFOutput(m, mx)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Let's use a single sample of dimension 1 (for simple demonstration):
    x = torch.tensor([[-2.0], [0.2], [1.5]])   # shape (1, 1)

    # 1) Identity version
    m_kef_identity = MKEF(robust=False, C=1.0)
    out_identity = m_kef_identity(x)
    print("[IDENTITY] m:\n", out_identity.m)
    print("[IDENTITY] mx:\n", out_identity.mx)

    # 2) Robust version
    m_kef_robust = MKEF(robust=True, C=1.0)
    out_robust = m_kef_robust(x)
    print("\n[ROBUST] m:\n", out_robust.m)
    print("[ROBUST] mx:\n", out_robust.mx)
