# # run_kef.py
# import torch
# from dataclasses import dataclass
# from typing import Optional, Any
# from Bimodal.KEF import KEF
# from src.kernel import Scaled_PIMQ
# from Bimodal.M_KEF import MKEF
# from src.KSD_Bayes import KSD_Bayes
# from src.rscm_torch import RSCMEstimator
#
#
# @dataclass
# class KEFOutput:
#     """Output structure for KEF0 computations."""
#     An: torch.Tensor  # Matrix An from KSD
#     vn: torch.Tensor  # Vector vn from KSD
#     w: float  # Weight parameter
#     theta: torch.Tensor
#
#
# def run_KEF(
#         X: torch.Tensor,
#         p: int,
#         L: float,
#         robust: bool = False,
#         weighted: bool = False,
#         weight_factor: float = 1.0,
#         log_p: Optional[torch.Tensor] = None,
# ) -> KEFOutput:
#     """
#     Run Kernel Exponential Family computations with KSD-Bayes.
#
#     Args:
#         X: Input tensor of shape (n, d)
#         p: Number of basis functions
#         L: Width parameter for reference measure
#         robust: Whether to use robust estimation (default: False)
#         weighted: Whether to use weighted KSD (default: False)
#         weight_factor: Scaling factor for weights (default: 1.0)
#         log_p: Log probability values for weighted KSD (optional)
#
#     Returns:
#         KEFOutput containing KSD matrices and optimal weight
#     """
#     # Initialize components
#     kef = KEF(p=p, L=L)
#
#     # Compute regularized scale matrix
#     rscm = RSCMEstimator()
#     S = rscm.fit(X, approach='ell1')
#
#     # Initialize kernel and preconditioner
#     kernel = Scaled_PIMQ(S=S)
#     m_kef = MKEF(robust=robust)
#
#     # Create wrapper functions for KSD_Bayes
#     def grad_T_wrapper(x: torch.Tensor) -> torch.Tensor:
#         """Wrapper for gradient of T function."""
#         return kef.grad_T(x)
#
#     def grad_b_wrapper(x: torch.Tensor) -> torch.Tensor:
#         """Wrapper for gradient of b function."""
#         return kef.grad_b(x)
#
#     def M_wrapper(x: torch.Tensor) -> Any:
#         """Wrapper for preconditioner function."""
#         return m_kef(x)
#
#     def K_wrapper(x: torch.Tensor, y: torch.Tensor) -> Any:
#         """Wrapper for kernel function."""
#         return kernel(x, y)
#
#     # Run KSD-Bayes
#     ksd_result = KSD_Bayes(
#         X=X,
#         grad_T=grad_T_wrapper,
#         grad_b=grad_b_wrapper,
#         M=M_wrapper,
#         K=K_wrapper,
#         log_p=log_p,
#         weighted=weighted,
#         weight_factor=weight_factor
#     )
#
#     return KEFOutput(
#         An=ksd_result.An,
#         vn=ksd_result.vn,
#         w=ksd_result.w,
#         theta=ksd_result.theta
#     )

# run_kef.py
import torch
from dataclasses import dataclass
from typing import Optional, Any
from Bimodal.KEF import KEF
from src.kernel import Scaled_PIMQ
from Bimodal.M_KEF import MKEF
from src.KSD_Bayes import KSD_Bayes
from src.rscm_torch import RSCMEstimator
import math


@dataclass
class KEFOutput:
    """Output structure for KEF0 computations."""
    An: torch.Tensor  # Matrix An from KSD
    vn: torch.Tensor  # Vector vn from KSD
    w: float  # Weight parameter
    theta: torch.Tensor


def run_KEF(
        X: torch.Tensor,
        p: int,
        L: float,
        robust: bool = False,
        weighted: bool = False,
        weight_factor: float = 1.0,
        log_p: Optional[torch.Tensor] = None,
) -> KEFOutput:
    """
    Run Kernel Exponential Family computations with KSD-Bayes.

    Args:
        X: Input tensor of shape (n, d)
        p: Number of basis functions
        L: Width parameter for reference measure
        robust: Whether to use robust estimation (default: False)
        weighted: Whether to use weighted KSD (default: False)
        weight_factor: Scaling factor for weights (default: 1.0)
        log_p: Log probability values for weighted KSD (optional)

    Returns:
        KEFOutput containing KSD matrices and optimal weight
    """
    # Initialize components
    kef = KEF(p=p, L=L)

    # Compute regularized scale matrix
    rscm = RSCMEstimator()
    S = rscm.fit(X, approach='ell1')

    # Initialize kernel and preconditioner
    kernel = Scaled_PIMQ(S=S)
    m_kef = MKEF(p, L, robust=robust)

    # Create wrapper functions for KSD_Bayes
    def grad_T_wrapper(x: torch.Tensor) -> torch.Tensor:
        """Wrapper for gradient of T function."""
        return kef.grad_T(x)

    def grad_b_wrapper(x: torch.Tensor) -> torch.Tensor:
        """Wrapper for gradient of b function."""
        return kef.grad_b(x)

    def b_x(x: torch.Tensor, L: float) -> torch.Tensor:
        """Compute b(x) = -x^2 / (2 * L^2)."""
        if L == 0:
            raise ValueError("Parameter L must be non-zero")
        b = - (x ** 2) / (2 * L ** 2)
        # return torch.mean(b, dim=0)
        return b

    def phi_Tx(x: torch.Tensor, max_i: int=p) -> torch.Tensor:
        """Compute phi(x) as a vector of basis functions for x."""
        phi_values = []
        for i in range(max_i):
            factorial_i = math.sqrt(math.factorial(i))
            phi_i = (x ** i) / factorial_i * torch.exp(-x ** 2 / 2)
            phi_values.append(phi_i)
        return torch.stack(phi_values, dim=1)

    def compute_px(x: torch.Tensor,  max_i: int=p, L: float=L) -> torch.Tensor:
        """Compute log p(x) = theta * phi(x) + b(x)."""
        phi_values = phi_Tx(x, max_i)
        b_values = b_x(x, L)
        Sig0 = 100 * torch.diag(torch.tensor(
            [(i + 1) ** (-1.1) for i in range(p)]
        ))
        mu0 = torch.zeros(p, 1)
        A0 = (1 / 2) * torch.linalg.inv(Sig0)
        v0 = -2 * A0 @ mu0
        theta0 = -(1 / 2) * torch.linalg.solve(A0, v0)
        p_x = torch.exp(torch.matmul(phi_values.squeeze(-1), theta0) + b_values)
        grad_p_x = p_x * (torch.matmul(grad_T_wrapper(x).squeeze(-1), theta0)+grad_b_wrapper(x))
        return p_x, grad_p_x

    def M_wrapper(x: torch.Tensor) -> Any:
        """Wrapper for preconditioner function."""
        return m_kef(x, compute_px(x, max_i=p, L=L))

    def K_wrapper(x: torch.Tensor, y: torch.Tensor) -> Any:
        """Wrapper for kernel function."""
        return kernel(x, y)

    # Run KSD-Bayes
    ksd_result = KSD_Bayes(
        X=X,
        grad_T=grad_T_wrapper,
        grad_b=grad_b_wrapper,
        M=M_wrapper,
        K=K_wrapper,
        log_p=log_p,
        weighted=weighted,
        weight_factor=weight_factor
    )

    return KEFOutput(
        An=ksd_result.An,
        vn=ksd_result.vn,
        w=ksd_result.w,
        theta=ksd_result.theta
    )