import torch


def nearestSPD(A: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of the nearestSPD function from MATLAB.
    Finds the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A.

    Steps:
      1) Symmetrize A into B = (A + A.T)/2
      2) Compute the symmetric polar factor of B => H
      3) A_hat = (B + H)/2
      4) Symmetrize A_hat
      5) If needed, adjust A_hat slightly so that it is strictly PD (Cholesky works).

    Args:
        A (torch.Tensor): A square matrix of shape (n, n).

    Returns:
        torch.Tensor: A symmetric positive-definite matrix closest to A in Frobenius norm.
    """

    # 1) Check A is square
    n, m = A.shape
    if n != m:
        raise ValueError("A must be a square matrix.")

    # If 1x1 and non-positive, return eps
    if n == 1 and A[0, 0] <= 0:
        return torch.tensor([[torch.finfo(A.dtype).eps]], dtype=A.dtype, device=A.device)

    # 2) Symmetrize A => B
    B = 0.5 * (A + A.transpose(-1, -2))

    # 3) Compute the symmetric polar factor of B. (Equivalent to "svd(B)" => H = V * Sigma * V^T)
    #    a. We do an eigen-decomposition of B, or an SVD of B
    #    b. The 'symmetric polar factor' is V*Sigma*V.T where B=U*Sigma*V.T in SVD sense,
    #       but for a real symmetric B, we can use eigen-decomposition.
    #
    #    However, B might not be positive-semidefinite, so we use SVD for a robust approach:
    U, S, Vt = torch.linalg.svd(B, full_matrices=False)
    # H = V * Sigma * V^T
    H = (Vt.transpose(-1, -2) * S.unsqueeze(-2)) @ Vt

    # 4) Ahat = (B+H)/2
    Ahat = 0.5 * (B + H)

    # 5) Force Ahat to be symmetric again
    Ahat = 0.5 * (Ahat + Ahat.transpose(-1, -2))

    # 6) Check if Ahat is PD by attempting a Cholesky.
    #    If it fails, adjust slightly along the diagonal.
    #    We'll do an iterative fix, similar to the MATLAB code.
    k = 0
    max_iter = 10  # safeguard
    while k < max_iter:
        try:
            _ = torch.linalg.cholesky(Ahat)
            # if Cholesky succeeds, Ahat is SPD
            return Ahat
        except RuntimeError:
            # Not SPD, so nudge eigenvalues
            k += 1
            # find smallest eigenvalue
            eigvals = torch.linalg.eigvalsh(Ahat)
            mineig = eigvals.min()
            if mineig < 0.0:
                # shift by a small amount = (-mineig * k^2 + eps(mineig)) on the diagonal
                eps_val = torch.finfo(Ahat.dtype).eps
                shift_val = -mineig * (k ** 2) + eps_val
                Ahat = Ahat + shift_val * torch.eye(n, dtype=Ahat.dtype, device=Ahat.device)
            else:
                # If no negative eigenvalue, we might just have a numerical issue
                # Add a tiny identity * epsilon
                Ahat = Ahat + torch.finfo(Ahat.dtype).eps * torch.eye(n, dtype=Ahat.dtype, device=Ahat.device)

            # Sym again
            Ahat = 0.5 * (Ahat + Ahat.transpose(-1, -2))

    # If for some reason we never returned in the loop, return Ahat
    return Ahat
