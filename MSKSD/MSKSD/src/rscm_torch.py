import torch
import torch.nn.functional as F
from torch.linalg import eigh
from sklearn.covariance import LedoitWolf


class RSCMEstimator:
    def __init__(self):
        pass

    def compute_etas(self, X, centered=False):
        """Compute eta1 = trace(S)/p and eta2 = trace(S^2)/p."""
        if not centered:
            X = X - X.mean(dim=0, keepdim=True)

        n, d = X.shape
        S = X.T @ X / (n - 1)
        eta1 = torch.trace(S) / d
        eta2 = torch.trace(S @ S) / d

        gamma = eta2 / eta1**2
        return {"eta": torch.tensor([eta1, eta2]), "S": S, "gamma": gamma}

    def estimate_kappa(self, X):
        """Estimate elliptical kurtosis."""
        n, d = X.shape
        S = X.T @ X / (n - 1)
        eta1 = torch.trace(S) / d
        m4 = torch.mean(torch.sum(X**2, dim=1)**2) / d
        kappa = (n * (m4 - (d + 2) * eta1**2 / d)) / ((n - 1) * eta1**2)
        return kappa

    def gamell1(self, X):
        """Estimate sphericity gamma using Ell1."""
        n, d = X.shape
        S = X.T @ X / (n - 1)
        eta1 = torch.trace(S) / d
        eta2 = torch.trace(S @ S) / d
        gamma = ((n - 1) * (n + 1) * eta1**2 - n * eta2) / ((n - 1) * eta1**2)
        return torch.clamp(gamma, min=1, max=d)

    def gamell2(self, X, kappa, eta):
        """Estimate sphericity gamma using Ell2."""
        n, d = X.shape
        m4 = torch.mean(torch.sum(X**2, dim=1)**2) / d
        var_m2 = (m4 - eta[1]) / n

        def objective(gamma):
            T = gamma - 1
            var_gamma = (
                (2 * T * (T + kappa * (2 * gamma + d) / n) + (gamma + d) / (n - 1)) ** 2
                / T**2
                * var_m2
            )
            return var_gamma

        gamma = torch.tensor(1.1, requires_grad=True)
        optimizer = torch.optim.LBFGS([gamma], line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = objective(gamma)
            loss.backward()
            return loss

        optimizer.step(closure)
        return gamma.item()

    def fit(self, X, approach="ell1", kappa=None, gamma=None, inverse=False, centered=False, verbose=False):
        """Fit the Regularized Sample Covariance Matrix Estimation."""
        if not centered:
            X = X - X.mean(dim=0, keepdim=True)

        n, d = X.shape
        ret = self.compute_etas(X, centered)
        eta = ret["eta"]
        S = ret["S"]

        if approach in ["ell1", "ell2"]:
            if kappa is None:
                kappa = self.estimate_kappa(X)

            if gamma is None:
                if approach == "ell1":
                    gamma = self.gamell1(X)
                elif approach == "ell2":
                    gamma = self.gamell2(X, kappa, eta)

            T = gamma - 1
            beta = T / (T + kappa * (2 * gamma + d) / n + (gamma + d) / (n - 1))
            beta = torch.clamp(beta, min=0, max=1)

            if verbose:
                print(f"Approach: {approach}")
                print(f"Elliptical Kurtosis (kappa): {kappa:.4f}")
                print(f"Sphericity (gamma): {gamma:.4f}")

        elif approach == "lw":
            lw = LedoitWolf()
            lw.fit(X.numpy())  # Convert to NumPy for Ledoit-Wolf
            S = torch.tensor(lw.covariance_, dtype=torch.float32)
            beta = lw.shrinkage_

            if verbose:
                print(f"Approach: {approach}")
                print(f"Shrinkage Parameter (beta): {beta:.4f}")

        else:
            raise ValueError("Approach must be one of ['ell1', 'ell2', 'lw'].")

        RSCM = beta * S + (1 - beta) * eta[0] * torch.eye(d)

        if inverse:
            eigvals, eigvecs = eigh(S)
            tmp = (1 - beta) * eta[0]
            invRSCM = eigvecs @ torch.diag(1 / (beta * eigvals + tmp)) @ eigvecs.T
        else:
            invRSCM = None

        # return RSCM, invRSCM, {"beta": beta, "gamma": gamma, "kappa": kappa, "eta": eta}
        return RSCM

if __name__ == "__main__":
    X = torch.tensor([
        [2.1, 3.4, 1.2],
        [4.5, 1.8, 2.3],
        [3.3, 2.5, 0.9],
        [1.1, 4.2, 3.7],
        [2.8, 3.0, 2.4]
    ], dtype=torch.float32)

    estimator = RSCMEstimator()

    # Example using Ell1 approach
    RSCM_ell1, invRSCM_ell1, stats_ell1 = estimator.fit(X, approach="lw", inverse=True, verbose=True)

    print("RSCM (Ell1):\n", RSCM_ell1)
    print("Inverse RSCM (Ell1):\n", invRSCM_ell1)
    print("Statistics:", stats_ell1)
