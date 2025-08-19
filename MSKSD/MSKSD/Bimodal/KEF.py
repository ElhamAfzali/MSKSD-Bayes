import torch
import math


class KEF:
    """
    A class implementing gradient functions for a 1D Gaussian-based
    kernel exponential family (KEF0), specifically designed to work with n x 1 data.

    In this setup:
      - The reference measure is Gaussian N(0, L^2).
      - b(x) ~ -x^2/(2L^2)
      - T_i(x) ~ x^i / sqrt(i!) * exp(-x^2/2),
        for i=0..(p-1), which are the 'natural sufficient statistics'
        in a Gaussian KEF0.

    We compute:
      grad_b(x) and grad_T(x),
    where x must be of shape (n, 1).

    Example Usage:
    --------------
    >>> kef = KEF0(p=4, L=1.0)
    >>> x = torch.tensor([[0.0], [1.0], [1.5], [-0.1]])
    >>> gb = kef.grad_b(x)
    >>> gT = kef.grad_T(x)
    >>> print(gb.shape)    # (4, 1)
    >>> print(gT.shape)    # (4, 4)
    """

    def __init__(self, p: int, L: float = 1.0):
        """
        Initialize a KEF0 object.

        Parameters
        ----------
        p : int
            Number of basis functions (Hermite-like expansions).
        L : float, optional
            Standard deviation of the reference Gaussian measure N(0, L^2).
            Default is 1.0.
        """
        self.p = p
        self.L = L

        # Precompute 1 / sqrt(i!) for i=0..(p-1).
        # This avoids repeated expensive math.factorial calls in grad_T().
        self.factorials_inv_sqrt = torch.tensor([
            1.0 / math.sqrt(math.factorial(i)) for i in range(self.p)
        ], dtype=torch.float32)

    def grad_b(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of b(x) for a 1D Gaussian reference.

        b(x) = - x^2 / (2 * L^2) + constant
        => grad_b(x) = - x / (L^2)

        Parameters
        ----------
        x : torch.Tensor
            Shape (n, 1). Each row is a 1-dimensional sample.

        Returns
        -------
        torch.Tensor
            Gradient tensor of shape (n, 1), with values - x_i / L^2.
        """
        if x.shape[1] != 1:
            raise ValueError("Input x must have shape (n, 1).")
        return -x / (self.L ** 2)

    def grad_T(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of T(x) for a 1D Gaussian reference.

        Parameters
        ----------
        x : torch.Tensor
            Shape (n, 1). Each row is a 1-dimensional sample.

        Returns
        -------
        torch.Tensor
            Shape (n, p) containing the derivatives for each sample and basis function.
        """
        if x.shape[1] != 1:
            raise ValueError("Input x must have shape (n, 1).")

        # Get the number of samples
        n = x.shape[0]
        x = x.squeeze(1)  # Shape: (n,)

        # Create power sequence -1 to p-2
        power_seq = torch.arange(-1, self.p - 1, dtype=x.dtype, device=x.device)

        # Compute x raised to powers from -1 to p-2
        # Use unsqueeze to make x broadcastable
        power_term = x.unsqueeze(-1).pow(power_seq)  # Shape: (n, p)

        # Create sequence 0 to p-1 for the subtraction term
        idx_seq = torch.arange(self.p, dtype=x.dtype, device=x.device)

        # Compute (i - x^2) term
        bracket_term = idx_seq.unsqueeze(0) - x.pow(2).unsqueeze(1)  # Shape: (n, p)

        # Compute exp(-x^2/2) term
        exp_term = torch.exp(-0.5 * x.pow(2)).unsqueeze(1)  # Shape: (n, 1)

        # Combine all terms
        out = (self.factorials_inv_sqrt.to(x.device) *
               power_term *
               bracket_term *
               exp_term)  # Shape: (n, p)

        return out


# Example usage
if __name__ == "__main__":
    kef = KEF(p=5, L=1.0)
    x = torch.tensor([[1.5], [-0.08]])  # (n=1, 1)

    gb = kef.grad_b(x)
    gT = kef.grad_T(x)

    print("grad_b(x):")
    print(gb)
    print("\ngrad_T(x):")
    print(gT.shape)