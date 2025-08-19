import torch

class Gauss:
    @staticmethod
    def grad_b(x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of b(x) where b(x) = (-1/2) ||x||^2 for the Gaussian location model.
        Args:
            x: Input tensor of shape (n, d), where n is the number of samples and d is the dimension.
        Returns:
            Gradient tensor of shape (n, d), representing -x for each sample.
        """
        assert x.dim() == 2, "Input tensor x must have shape (n, d)."
        return -x

    @staticmethod
    def grad_T(x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of T(x) where T(x) = x in the Gaussian location model.
        Args:
            x: Input tensor of shape (n, d), where n is the number of samples and d is the dimension.
        Returns:
            Gradient tensor of shape (n, d, d), representing the identity matrix for each sample.
        """
        n, d = x.shape
        assert d > 0, "Dimension of input x must be greater than 0."
        return torch.eye(d, device=x.device).unsqueeze(0).repeat(n, 1, 1)


