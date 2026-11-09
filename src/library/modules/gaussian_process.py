"""
GPIV building blocks: first-stage propensity KRR, expected treatment-state
kernel, and the exact GP used by the GPIV estimator.

  - Stage I : contextual propensity KRR on [X, Z]  ->  Pi(x,z) = [1-p, p]
  - Stage II: expected kernel  K~((x,z),(x',z')) = k_x(x,x') * Pi^T Sigma_t Pi'
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.models import ExactGP
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood


def _set_rbf_lengthscale(kernel: ScaleKernel,
                         lengthscale: Union[float, torch.Tensor]) -> None:
    """Set all ARD RBF lengthscales while respecting GPyTorch constraints."""
    base_kernel = kernel.base_kernel
    current = base_kernel.lengthscale
    value = torch.as_tensor(lengthscale, dtype=current.dtype, device=current.device)
    if value.ndim == 0:
        value = torch.full_like(current, float(value.item()))
    else:
        value = value.reshape_as(current)
    base_kernel.lengthscale = value


class PropensityKRR(nn.Module):
    """
    Kernel Ridge Regression propensity estimator.

    π̂₁ = K_xz (K_xz + ηI)⁻¹ t

    Input:  [X, Z]  (d+1 dimensional)
    Output: Π [N, 2]  propensity matrix
    """

    def __init__(self, input_dim: int,
                 init_lengthscale: Optional[Union[float, torch.Tensor]] = None):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of [X, Z] = d_covariate + 1.
        """
        super().__init__()
        # GPyTorch kernel over [X, Z] with ARD lengthscales.
        # For binary phase-I KRR, jointly learning both the kernel outputscale
        # and ridge penalty is numerically ill-conditioned: bad small-sample
        # runs tend to collapse propensities toward zero while preserving rank
        # order. We therefore keep the overall kernel scale fixed and only
        # learn the ARD lengthscales.
        self.kernel = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
        self.kernel.raw_outputscale.requires_grad_(False)
        if init_lengthscale is not None:
            self.set_lengthscale(init_lengthscale)

        # Fixed Tikhonov parameter η > 0 (softplus(-2) ≈ 0.13).
        # This stabilizes the first stage in low-data binary-IV regimes.
        self._raw_eta = nn.Parameter(torch.tensor(-2.0), requires_grad=False)

    def set_lengthscale(self, lengthscale: Union[float, torch.Tensor]) -> None:
        _set_rbf_lengthscale(self.kernel, lengthscale)

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self._raw_eta)

    def forward(self, X: torch.Tensor, Z: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        """
        Compute propensity matrix Π.

        Parameters
        ----------
        X : [N, d]  covariates
        Z : [N, 1]  binary instrument
        T : [N, 1]  binary treatment

        Returns
        -------
        Pi : [N, 2] propensity matrix  [1-π̂₁, π̂₁]
        """
        N = X.shape[0]
        device = X.device

        # 1. Input: concatenate [X, Z]
        XZ = torch.cat([X, Z], dim=-1)  # [N, d+1]

        # 2. Kernel matrix K_xz  [N, N]
        K_xz = self.kernel(XZ).evaluate()

        # 3. Differentiable KRR solve
        #    α = solve(K_xz + ηI + jitter, t)
        #    π̂₁ = K_xz α
        eta = self.eta
        jitter = 1e-4
        A = K_xz + (eta + jitter) * torch.eye(N, device=device, dtype=K_xz.dtype)
        alpha = torch.linalg.solve(A, T.squeeze(-1))  # [N]
        # 4. Predict both instrument branches for each x.
        z0 = torch.zeros_like(Z)
        z1 = torch.ones_like(Z)
        XZ0 = torch.cat([X, z0], dim=-1)
        XZ1 = torch.cat([X, z1], dim=-1)
        k_xz0 = self.kernel(XZ, XZ0).evaluate()
        k_xz1 = self.kernel(XZ, XZ1).evaluate()
        pi_0_raw = k_xz0.transpose(-1, -2) @ alpha
        pi_1_raw = k_xz1.transpose(-1, -2) @ alpha
        pi_0 = torch.clamp(pi_0_raw, min=0.0, max=1.0)
        pi_1 = torch.clamp(pi_1_raw, min=0.0, max=1.0)

        # 5. Select the factual branch for each observed (x, z)
        z_mask = (Z.squeeze(-1) > 0.5)
        pi_1_factual = torch.where(z_mask, pi_1, pi_0)

        # 6. Construct Π = [1-π̂₁, π̂₁]
        Pi = torch.stack([1.0 - pi_1_factual, pi_1_factual], dim=-1)  # [N, 2]
        return Pi


class ExpectedKernel(Kernel):
    """
    Custom GPyTorch kernel implementing:

        K̃(x_i, x_j) = k_x(x_i, x_j) · (π_i^⊤ Σ_t π_j)

    where Σ_t is a learnable 2×2 PSD treatment covariance matrix
    and π_i are the propensity vectors from Phase I.
    """

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, **kwargs):
        """
        Parameters
        ----------
        base_kernel : Kernel
            The covariate kernel k_x (e.g., ScaleKernel(RBFKernel(...))).
        """
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

        # Learnable Σ_t parameters (unconstrained)
        self._raw_sigma0_sq = nn.Parameter(torch.tensor(0.0))   # softplus → σ₀²
        self._raw_sigma1_sq = nn.Parameter(torch.tensor(0.0))   # softplus → σ₁²
        self._raw_rho = nn.Parameter(torch.tensor(0.0))         # tanh    → ρ

    @property
    def Sigma_t(self) -> torch.Tensor:
        """Construct the 2×2 PSD treatment covariance matrix."""
        sigma0 = torch.sqrt(F.softplus(self._raw_sigma0_sq))
        sigma1 = torch.sqrt(F.softplus(self._raw_sigma1_sq))
        rho = torch.tanh(self._raw_rho)

        S = torch.stack([
            torch.stack([sigma0 ** 2,        rho * sigma0 * sigma1]),
            torch.stack([rho * sigma0 * sigma1,  sigma1 ** 2       ])
        ])  # [2, 2]
        return S

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                diag: bool = False, last_dim_is_batch: bool = False,
                **params) -> torch.Tensor:
        """
        Compute K̃ = K_x ⊙ (Π₁ Σ_t Π₂^⊤) via Augmentation Trick.
        x1 and x2 are augmented inputs containing both X and Π.
        """
        # Slice X and Π from the augmented inputs
        X1, Pi1 = x1[..., :-2], x1[..., -2:]
        X2, Pi2 = x2[..., :-2], x2[..., -2:]

        # Base covariate kernel
        K_x = self.base_kernel(X1, X2, diag=diag,
                               last_dim_is_batch=last_dim_is_batch, **params)

        Sigma = self.Sigma_t  # [2, 2]

        if diag:
            # Diagonal: (Π₁ Σ_t Π₂^⊤)_diag = row-wise dot of (Π₁ Σ_t) and Π₂
            Kt_diag = torch.sum((Pi1 @ Sigma) * Pi2, dim=-1)  # [N]
            return K_x.evaluate() * Kt_diag if hasattr(K_x, 'evaluate') else K_x * Kt_diag
        else:
            # Full: Π₁ Σ_t Π₂^⊤  [N1, N2]
            Kt = Pi1 @ Sigma @ Pi2.transpose(-1, -2)  # [..., N1, N2]
            K_x_eval = K_x.evaluate() if hasattr(K_x, 'evaluate') else K_x
            return K_x_eval * Kt


class GPIVExactGP(ExactGP):
    """
    Exact GP with the Expected Kernel for Binary IV.

    forward() returns MultivariateNormal(0, K̃).
    Propensities must be set on the expected_kernel before calling forward().
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: GaussianLikelihood,
                 d_covariate: int,
                 init_lengthscale: Optional[Union[float, torch.Tensor]] = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()

        # Base covariate kernel k_x with ARD lengthscales
        base_kernel = ScaleKernel(RBFKernel(ard_num_dims=d_covariate))

        # Expected kernel wrapping the base kernel
        self.expected_kernel = ExpectedKernel(base_kernel)
        if init_lengthscale is not None:
            self.set_covariate_lengthscale(init_lengthscale)

    def set_covariate_lengthscale(self, lengthscale: Union[float, torch.Tensor]) -> None:
        _set_rbf_lengthscale(self.expected_kernel.base_kernel, lengthscale)

    def forward(self, x_aug: torch.Tensor) -> MultivariateNormal:
        # X is everything except the last 2 dimensions (which are propensities)
        X = x_aug[..., :-2]
        mean = self.mean_module(X)
        covar = self.expected_kernel(x_aug, x_aug)
        return MultivariateNormal(mean, covar)
