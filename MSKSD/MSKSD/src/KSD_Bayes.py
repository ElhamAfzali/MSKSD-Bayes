# ksd_bayes.py
import torch
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class KSDOutput:
    """Output structure for KSD-Bayes computations."""
    An: torch.Tensor  # pxp matrix An
    vn: torch.Tensor  # px1 vector vn
    w: float  # Weight parameter for frequentist coverage
    theta: torch.Tensor


# def compute_weights(log_p: torch.Tensor, weight_factor: float = 1.0) -> torch.Tensor:
#     """
#     Compute weights w(x) = weight_factor / log p(x).
#     """
#     return weight_factor / (torch.abs((log_p) + 1e-8))

def compute_weights(log_p: torch.Tensor, weight_factor: float = 1.0) -> torch.Tensor:
    """
    Compute weights w(x) = weight_factor / log p(x).
    """
    return weight_factor * torch.exp(-log_p)


def KSD_Bayes(
        X: torch.Tensor,
        grad_T: Callable,
        grad_b: Callable,
        M: Callable,
        K: Callable,
        log_p: Optional[torch.Tensor] = None,
        weighted: bool = False,
        weight_factor: float = 1.0,
) -> KSDOutput:
    """
    Compute KSD-Bayes statistics with option for weighted KSD.
    """
    if weighted and log_p is None:
        raise ValueError("log_p is required for weighted KSD")

    # Get dimensions
    n, d = X.shape
    p = grad_T(X[0:1, :]).shape[-1]

    def phi(x: torch.Tensor) -> torch.Tensor:
        """Compute phi(x) = M(x).m * grad_T(x)'."""
        M_x = M(x)
        grad_T_x = grad_T(x)
        return M_x.m.unsqueeze(-1) * grad_T_x

    def compute_A(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute A term."""
        K_xy = K(x, y)
        phi_x = phi(x)
        phi_y = phi(y)
        return K_xy.k.unsqueeze(-1).unsqueeze(-1) * torch.matmul(
            phi_x.transpose(-2, -1), phi_y
        ).unsqueeze(0)

    def compute_v_perp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute v_perp term."""
        K_xy = K(x, y)
        K_yx = K(y, x)
        M_x = M(x)
        M_y = M(y)

        phi_x = phi(x).squeeze(0)
        phi_y = phi(y).squeeze(0)
        grad_b_x = grad_b(x).squeeze(0)
        grad_b_y = grad_b(y).squeeze(0)

        # Terms 1 & 2
        term1 = K_yx.k * torch.matmul(grad_b_y, M_y.m.squeeze(0).unsqueeze(-1) * phi_x)
        term2 = K_xy.k * torch.matmul(grad_b_x, M_x.m.squeeze(0).unsqueeze(-1) * phi_y)

        # Divergence terms
        def divx_MK(x, y):
            K_xy = K(x, y)
            M_x = M(x)
            div_M = torch.diagonal(M_x.mx.squeeze(0), dim1=-2, dim2=-1)  # (5,)
            kx = K_xy.kx.squeeze()  # (5,)
            return K_xy.k * div_M + M_x.m.squeeze(0) * kx

            # return M_x.m.squeeze(0) * kx

        def divy_MK(x, y):
            K_xy = K(x, y)
            M_y = M(y)
            div_M =  torch.diagonal(M_y.mx.squeeze(0), dim1=-2, dim2=-1)  # (5,)
            ky = K_xy.ky.squeeze()  # (5,)
            return K_xy.k * div_M + M_y.m.squeeze(0) * ky

            # return  M_y.m.squeeze(0) * ky

        # Terms 3 & 4: Use MATLAB's exact divergence computation
        div_x = divx_MK(x, y)  # Shape: (5,)
        div_y = divy_MK(x, y)  # Shape: (5,)

        term3 = torch.matmul(div_x, phi_y)
        term4 = torch.matmul(div_y, phi_x)

        return term1 + term2 + term3 + term4

    # Initialize matrices
    An = torch.zeros((p, p), device=X.device)
    vn = torch.zeros((p, 1), device=X.device)

    # Initialize caches
    A_cache = torch.zeros((n, p, p), device=X.device)
    v_cache = torch.zeros((n, p), device=X.device)

    # Compute weights if using weighted KSD
    weights = compute_weights(log_p, weight_factor) if weighted else torch.ones(n, device=X.device)

    # Main computation loop
    for i in range(n):
        x_i = X[i:i + 1, :]
        w_i = weights[i]

        for j in range(n):
            x_j = X[j:j + 1, :]
            w_j = weights[j]
            weight_pair = w_i * w_j if weighted else 1.0

            # Compute A and v terms
            A_ij = compute_A(x_i, x_j)
            v_ij = compute_v_perp(x_i, x_j)

            # Weight the terms if using weighted KSD
            A_ij_weighted = weight_pair * A_ij
            v_ij_weighted = weight_pair * v_ij

            # Update caches for each i
            A_cache[i] += (1 / n) * A_ij_weighted.squeeze()
            v_cache[i] += (1 / n) * v_ij_weighted.squeeze().squeeze()

            # Update final matrices
            An += (1 / n) * A_ij_weighted.squeeze()
            vn += (1 / n) * v_ij_weighted.squeeze().unsqueeze(-1)




    # Compute minimum KSD estimator
    theta = -(1 / 2) * torch.linalg.solve(An, vn)

    # Information-type matrices
    H = (2 / n) * An

    # Compute J matrix using caches
    J = torch.zeros((p, p), device=X.device)
    for k in range(n):
        tmp = 2 * torch.matmul(A_cache[k], theta) + v_cache[k].unsqueeze(-1)
        J += (1 / n) * torch.matmul(tmp, tmp.transpose(-2, -1))

    # Compute optimal weight
    J_inv_H = torch.linalg.solve(J, H)
    w = torch.trace(torch.matmul(H, J_inv_H)) / torch.trace(H)

    return KSDOutput(An=An, vn=vn, w=w.item(), theta=theta)