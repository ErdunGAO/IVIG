"""
GPIV: exact Gaussian-process instrumental-variable estimator for binary
treatment CATE, plus the posterior quantities used by the acquisition rules.

Training (GPIVTrainer):
  1. standardise Y
  2. initialise PropensityKRR (stage I), GPIVExactGP (stage II), GaussianLikelihood
  3. joint MLL optimisation with Adam
  4. wrap everything into GPIVModel for inference / acquisition scoring

Acquisition-facing API (GPIVModel):
  predict_cate, predict_cate_with_variance,
  phantom_ivig_scores      -- EB-IVIG score (Eq. 7 in the paper)
  krr_particle_ivig_scores -- S-particle first-stage Monte Carlo variant
  y_ivig_scores            -- outcome-only Y-IVIG comparator
  direct_tau_ig_scores     -- "No Stage-I" IVIG (z treated as t)
  sanitized_bald_scores    -- mu-BALD score at the plug-in propensity feature
"""

from typing import Any, Dict, Optional, Tuple, Union
import logging
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.library.datasets.data_class import TrainDataSet, TestDataSet, weighted_sqrt_pehe_np
from src.library.modules.gaussian_process import PropensityKRR, GPIVExactGP

logger = logging.getLogger(__name__)


class GPIVModel:
    """
    Trained GPIV model for CATE prediction and evaluation.

    CATE(x*) = (K̃*N^(1) - K̃*N^(0)) (K̃_train + σ²I)⁻¹ y
    """

    def __init__(self, gp: GPIVExactGP, likelihood: GaussianLikelihood,
                 propensity_module: PropensityKRR,
                 X_train: torch.Tensor, Z_train: torch.Tensor,
                 T_train: torch.Tensor, Y_train: torch.Tensor,
                 Pi_train: torch.Tensor,
                 y_mean: float, y_std: float):
        self.gp = gp
        self.likelihood = likelihood
        self.propensity_module = propensity_module
        self.X_train = X_train
        self.Z_train = Z_train
        self.T_train = T_train
        self.Y_train = Y_train
        self.Pi_train = Pi_train
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = X_train.device
        self.dtype = X_train.dtype
        self._krr_jitter = 1e-4

        self.train_x_aug = torch.cat([self.X_train, self.Pi_train], dim=-1)
        self._build_caches()

    def _build_caches(self):
        with torch.no_grad():
            self.XZ_train = torch.cat([self.X_train, self.Z_train], dim=-1)
            self.krr_train_kernel = self.propensity_module.kernel(self.XZ_train).evaluate()
            self.krr_reg = self.propensity_module.eta.detach() + self._krr_jitter
            eye = torch.eye(self.X_train.shape[0], device=self.device, dtype=self.dtype)
            self.krr_system = self.krr_train_kernel + self.krr_reg * eye
            self.krr_chol = torch.linalg.cholesky(self.krr_system)
            self.krr_alpha = torch.cholesky_solve(self.T_train, self.krr_chol).squeeze(-1)
            self.noise_var = float(self.likelihood.noise.detach().to(self.dtype).item())
            train_kernel = self._evaluate_kernel(self.train_x_aug, self.train_x_aug)
            self.gp_train_system = train_kernel + self.noise_var * eye
            self.gp_train_chol = torch.linalg.cholesky(self.gp_train_system)

    def _as_tensor(self, value: Union[np.ndarray, torch.Tensor, float, int]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        return tensor

    def _ensure_2d_covariate(self, covariate: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x_t = self._as_tensor(covariate)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        return x_t

    def _ensure_2d_instrument(self, instrument: Union[np.ndarray, torch.Tensor, float, int],
                              n_rows: int) -> torch.Tensor:
        z_t = self._as_tensor(instrument)
        if z_t.dim() == 0:
            z_t = z_t.repeat(n_rows).unsqueeze(-1)
        elif z_t.dim() == 1:
            if z_t.shape[0] == 1 and n_rows != 1:
                z_t = z_t.repeat(n_rows).unsqueeze(-1)
            else:
                z_t = z_t.reshape(-1, 1)
        if z_t.shape[0] != n_rows:
            raise ValueError(f"Instrument batch size {z_t.shape[0]} does not match covariate batch size {n_rows}.")
        return z_t

    def _build_do_augmented(self, covariate_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_rows = covariate_t.shape[0]
        pi_do1 = torch.zeros(n_rows, 2, device=self.device, dtype=self.dtype)
        pi_do1[:, 1] = 1.0
        pi_do0 = torch.zeros(n_rows, 2, device=self.device, dtype=self.dtype)
        pi_do0[:, 0] = 1.0
        return (torch.cat([covariate_t, pi_do1], dim=-1),
                torch.cat([covariate_t, pi_do0], dim=-1))

    def _evaluate_kernel(self, x1_aug: torch.Tensor, x2_aug: torch.Tensor) -> torch.Tensor:
        kernel_val = self.gp.expected_kernel(x1_aug, x2_aug)
        return kernel_val.evaluate() if hasattr(kernel_val, "evaluate") else kernel_val

    def _posterior_covariance(self, test_aug: torch.Tensor) -> torch.Tensor:
        k_tt = self._evaluate_kernel(test_aug, test_aug)
        k_tn = self._evaluate_kernel(test_aug, self.train_x_aug)
        solved = torch.cholesky_solve(k_tn.transpose(-1, -2), self.gp_train_chol)
        post_covar = k_tt - k_tn @ solved
        return 0.5 * (post_covar + post_covar.transpose(-1, -2))

    def _predict_propensity_batch(self, covariate_t: torch.Tensor, instrument_t: torch.Tensor) -> torch.Tensor:
        p0, p1 = self._predict_propensity_pair_batch(covariate_t)
        z_mask = (instrument_t.squeeze(-1) > 0.5)
        return torch.where(z_mask, p1, p0)

    def _predict_propensity_pair_batch_raw(self, covariate_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_rows = covariate_t.shape[0]
        z0 = torch.zeros(n_rows, 1, device=self.device, dtype=self.dtype)
        z1 = torch.ones(n_rows, 1, device=self.device, dtype=self.dtype)
        xz0 = torch.cat([covariate_t, z0], dim=-1)
        xz1 = torch.cat([covariate_t, z1], dim=-1)
        k_cross0 = self.propensity_module.kernel(self.XZ_train, xz0).evaluate()
        k_cross1 = self.propensity_module.kernel(self.XZ_train, xz1).evaluate()
        p0_raw = k_cross0.transpose(-1, -2) @ self.krr_alpha
        p1_raw = k_cross1.transpose(-1, -2) @ self.krr_alpha
        return p0_raw, p1_raw

    def _predict_propensity_pair_batch(self, covariate_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p0_raw, p1_raw = self._predict_propensity_pair_batch_raw(covariate_t)
        return (
            torch.clamp(p0_raw, min=0.0, max=1.0),
            torch.clamp(p1_raw, min=0.0, max=1.0),
        )

    def _phantom_query_propensity_batch(self, covariate_t: torch.Tensor, instrument_t: torch.Tensor,
                                        assumed_t: int) -> torch.Tensor:
        xz_query = torch.cat([covariate_t, instrument_t], dim=-1)
        k_cross = self.propensity_module.kernel(self.XZ_train, xz_query).evaluate()  # [N, M]
        current_raw = k_cross.transpose(-1, -2) @ self.krr_alpha
        solved = torch.cholesky_solve(k_cross, self.krr_chol)  # [N, M]
        q_kernel = self.propensity_module.kernel(xz_query, xz_query).evaluate()
        q_diag = torch.diagonal(q_kernel, dim1=-2, dim2=-1)
        innovation = q_diag - torch.sum(k_cross * solved, dim=0)
        schur = q_diag + self.krr_reg - torch.sum(k_cross * solved, dim=0)
        delta = (float(assumed_t) - current_raw) * innovation / torch.clamp(schur, min=1e-8)
        updated_raw = current_raw + delta

        p0_raw, p1_raw = self._predict_propensity_pair_batch_raw(covariate_t)
        z_mask = (instrument_t.squeeze(-1) > 0.5)
        p0_raw = torch.where(z_mask, p0_raw, updated_raw)
        p1_raw = torch.where(z_mask, updated_raw, p1_raw)
        p0 = torch.clamp(p0_raw, min=0.0, max=1.0)
        p1 = torch.clamp(p1_raw, min=0.0, max=1.0)
        return torch.where(z_mask, p1, p0)

    def _krr_query_raw_mean_var(self, covariate_t: torch.Tensor,
                                instrument_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xz_query = torch.cat([covariate_t, instrument_t], dim=-1)
        k_cross = self.propensity_module.kernel(self.XZ_train, xz_query).evaluate()
        mean = k_cross.transpose(-1, -2) @ self.krr_alpha
        solved = torch.cholesky_solve(k_cross, self.krr_chol)
        q_kernel = self.propensity_module.kernel(xz_query, xz_query).evaluate()
        q_diag = torch.diagonal(q_kernel, dim1=-2, dim2=-1)
        var = torch.clamp(q_diag - torch.sum(k_cross * solved, dim=0), min=1e-12)
        return mean, var

    def _phantom_query_propensity_raw_var_batch(self, covariate_t: torch.Tensor,
                                                instrument_t: torch.Tensor,
                                                assumed_t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xz_query = torch.cat([covariate_t, instrument_t], dim=-1)
        k_cross = self.propensity_module.kernel(self.XZ_train, xz_query).evaluate()
        current_raw = k_cross.transpose(-1, -2) @ self.krr_alpha
        solved = torch.cholesky_solve(k_cross, self.krr_chol)
        q_kernel = self.propensity_module.kernel(xz_query, xz_query).evaluate()
        q_diag = torch.diagonal(q_kernel, dim1=-2, dim2=-1)
        innovation = torch.clamp(q_diag - torch.sum(k_cross * solved, dim=0), min=1e-12)
        schur = torch.clamp(innovation + self.krr_reg, min=1e-8)
        delta = (float(assumed_t) - current_raw) * innovation / schur
        updated_raw = current_raw + delta
        updated_var = torch.clamp(innovation * self.krr_reg / schur, min=1e-12)
        return updated_raw, updated_var

    def _candidate_augmented_from_branch(self, covariate_t: torch.Tensor, instrument_t: torch.Tensor,
                                         assumed_t: int) -> torch.Tensor:
        p_branch = self._phantom_query_propensity_batch(covariate_t, instrument_t, assumed_t)
        pi_branch = torch.stack([1.0 - p_branch, p_branch], dim=-1)
        return torch.cat([covariate_t, pi_branch], dim=-1)

    def _candidate_augmented_sanitized(self, covariate_t: torch.Tensor,
                                       instrument_t: torch.Tensor) -> torch.Tensor:
        p_current = self._predict_propensity_batch(covariate_t, instrument_t)
        pi_current = torch.stack([1.0 - p_current, p_current], dim=-1)
        return torch.cat([covariate_t, pi_current], dim=-1)

    def _candidate_augmented_direct_tau(self, covariate_t: torch.Tensor,
                                        instrument_t: torch.Tensor) -> torch.Tensor:
        z_mask = (instrument_t.squeeze(-1) > 0.5)
        pi = torch.zeros(covariate_t.shape[0], 2, device=self.device, dtype=self.dtype)
        pi[:, 0] = (~z_mask).to(self.dtype)
        pi[:, 1] = z_mask.to(self.dtype)
        return torch.cat([covariate_t, pi], dim=-1)

    def _fantasy_cate_variance_from_query_aug_batch(self,
                                                    target_t: torch.Tensor,
                                                    query_aug: torch.Tensor) -> torch.Tensor:
        if query_aug.ndim != 2:
            raise ValueError("query_aug must be a 2D tensor.")
        if query_aug.shape[0] == 0:
            return torch.empty((0, target_t.shape[0]), device=self.device, dtype=self.dtype)

        aug_1, aug_0 = self._build_do_augmented(target_t)
        all_aug = torch.cat([aug_1, aug_0, query_aug], dim=0)
        cov = self._posterior_covariance(all_aug)

        n_targets = target_t.shape[0]
        q_start = 2 * n_targets
        var_1 = torch.diagonal(cov[:n_targets, :n_targets])
        var_0 = torch.diagonal(cov[n_targets:2 * n_targets, n_targets:2 * n_targets])
        cov_10 = torch.diagonal(cov[:n_targets, n_targets:2 * n_targets])
        tau_var = torch.clamp(var_1 + var_0 - 2.0 * cov_10, min=1e-12)

        query_query = cov[q_start:, q_start:]
        latent_q_var = torch.clamp(torch.diagonal(query_query), min=1e-12)
        tau_q_cov = cov[:n_targets, q_start:] - cov[n_targets:2 * n_targets, q_start:]

        updated_tau_var = tau_var.unsqueeze(-1) - (tau_q_cov ** 2) / torch.clamp(
            latent_q_var.unsqueeze(0) + self.noise_var,
            min=1e-12,
        )
        updated_tau_var = torch.clamp(updated_tau_var, min=1e-12) * (self.y_std ** 2)
        return updated_tau_var.transpose(0, 1).contiguous()

    def _tau_query_update_stats_from_query_aug_batch(
        self,
        target_t: torch.Tensor,
        query_aug: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_aug.ndim != 2:
            raise ValueError("query_aug must be a 2D tensor.")
        if query_aug.shape[0] == 0:
            n_targets = target_t.shape[0]
            return (
                torch.empty((n_targets,), device=self.device, dtype=self.dtype),
                torch.empty((n_targets,), device=self.device, dtype=self.dtype),
                torch.empty((0,), device=self.device, dtype=self.dtype),
                torch.empty((0,), device=self.device, dtype=self.dtype),
                torch.empty((0, n_targets), device=self.device, dtype=self.dtype),
                torch.empty((0, n_targets), device=self.device, dtype=self.dtype),
            )

        aug_1, aug_0 = self._build_do_augmented(target_t)
        all_aug = torch.cat([aug_1, aug_0, query_aug], dim=0)
        posterior = self.gp(all_aug)
        mean = posterior.mean
        cov = posterior.covariance_matrix

        n_targets = target_t.shape[0]
        q_start = 2 * n_targets
        mean_tau = mean[:n_targets] - mean[n_targets:2 * n_targets]

        var_1 = torch.diagonal(cov[:n_targets, :n_targets])
        var_0 = torch.diagonal(cov[n_targets:2 * n_targets, n_targets:2 * n_targets])
        cov_10 = torch.diagonal(cov[:n_targets, n_targets:2 * n_targets])
        tau_var = torch.clamp(var_1 + var_0 - 2.0 * cov_10, min=1e-12)

        query_mean = mean[q_start:]
        query_query = cov[q_start:, q_start:]
        latent_q_var = torch.clamp(torch.diagonal(query_query), min=1e-12)
        y_var = torch.clamp(latent_q_var + self.noise_var, min=1e-12)

        tau_q_cov = cov[:n_targets, q_start:] - cov[n_targets:2 * n_targets, q_start:]
        beta = tau_q_cov / torch.clamp(y_var.unsqueeze(0), min=1e-12)
        updated_tau_var = tau_var.unsqueeze(-1) - (tau_q_cov ** 2) / torch.clamp(
            y_var.unsqueeze(0),
            min=1e-12,
        )
        updated_tau_var = torch.clamp(updated_tau_var, min=1e-12)
        return (
            mean_tau,
            tau_var,
            query_mean,
            y_var,
            beta.transpose(0, 1).contiguous(),
            updated_tau_var.transpose(0, 1).contiguous(),
        )

    def _gauss_hermite_rule(self, n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes, weights = np.polynomial.hermite.hermgauss(max(int(n_nodes), 1))
        node_t = torch.as_tensor(nodes, device=self.device, dtype=self.dtype)
        weight_t = torch.as_tensor(weights / np.sqrt(np.pi), device=self.device, dtype=self.dtype)
        return node_t, weight_t

    def _normal_logpdf(self, value: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = torch.clamp(var, min=1e-12)
        return -0.5 * (torch.log(2.0 * torch.pi * var) + (value - mean) ** 2 / var)

    def _tau_posterior_stats(self, covariate_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aug_1, aug_0 = self._build_do_augmented(covariate_t)
        all_aug = torch.cat([aug_1, aug_0], dim=0)
        posterior = self.gp(all_aug)
        mean = posterior.mean
        cov = posterior.covariance_matrix
        n_pts = covariate_t.shape[0]

        mean_tau = mean[:n_pts] - mean[n_pts:]
        cov_11 = cov[:n_pts, :n_pts]
        cov_00 = cov[n_pts:, n_pts:]
        cov_10 = cov[:n_pts, n_pts:]
        cov_01 = cov[n_pts:, :n_pts]
        tau_cov = cov_11 + cov_00 - cov_10 - cov_01
        tau_cov = 0.5 * (tau_cov + tau_cov.transpose(-1, -2))
        return mean_tau, tau_cov

    def _latent_query_variance_from_aug(self, query_aug: torch.Tensor) -> np.ndarray:
        post_cov = self._posterior_covariance(query_aug)
        latent_var = torch.clamp(torch.diagonal(post_cov), min=1e-12)
        return latent_var.cpu().numpy().reshape(-1, 1)

    @torch.no_grad()
    def predict_cate(self, covariate: np.ndarray) -> np.ndarray:
        """
        Compute CATE(x*) = E[h(1,x*) - h(0,x*) | D].
        Leverages native GPyTorch `.eval()` mode inference to automatically
        apply the precision matrix to the interventional cross covariances.
        """
        X_star = self._ensure_2d_covariate(covariate)
        X_star_aug_1, X_star_aug_0 = self._build_do_augmented(X_star)

        pred1 = self.gp(X_star_aug_1).mean
        pred0 = self.gp(X_star_aug_0).mean

        # CATE prediction (in standardised Y space)
        cate_std = pred1 - pred0

        # De-standardise
        cate = cate_std.cpu().numpy() * self.y_std

        return cate

    @torch.no_grad()
    def predict_cate_with_variance(self, covariate: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        covariate_t = self._ensure_2d_covariate(covariate)
        cate_mean_std, tau_cov = self._tau_posterior_stats(covariate_t)
        cate_mean = cate_mean_std * self.y_std
        cate_var = torch.clamp(torch.diagonal(tau_cov), min=1e-12) * (self.y_std ** 2)
        return cate_mean.cpu().numpy().reshape(-1, 1), cate_var.cpu().numpy().reshape(-1, 1)

    @torch.no_grad()
    def phantom_ivig_scores(self,
                            candidate_covariate: Union[np.ndarray, torch.Tensor],
                            instrument: Union[np.ndarray, torch.Tensor, float, int],
                            target_covariate: Union[np.ndarray, torch.Tensor],
                            prior_cate_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            branch_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            variance_nugget: Union[np.ndarray, torch.Tensor, float, int] = 0.0,
                            chunk_size: int = 128) -> np.ndarray:
        """
        Empirical-Bayes IVIG score for every candidate under one instrument:
        prior target entropy minus the branch-probability-weighted posterior
        entropy after a branch-specific fantasy update, averaged over targets.
        """
        cand_t = self._ensure_2d_covariate(candidate_covariate)
        inst_t = self._ensure_2d_instrument(instrument, cand_t.shape[0])
        target_t = self._ensure_2d_covariate(target_covariate)

        if prior_cate_var is None:
            _, prior_cate_var = self.predict_cate_with_variance(target_t)
        prior_var = self._as_tensor(prior_cate_var).reshape(-1)
        nugget = torch.clamp(self._as_tensor(variance_nugget), min=0.0).reshape(-1)
        if nugget.numel() == 1:
            prior_var = prior_var + nugget
        elif nugget.shape[0] == prior_var.shape[0]:
            prior_var = prior_var + nugget
        else:
            raise ValueError("variance_nugget must be scalar or match the number of target covariates.")
        prior_entropy = 0.5 * torch.log(torch.clamp(prior_var, min=1e-12))

        if branch_prob is None:
            current_p = self._predict_propensity_batch(cand_t, inst_t)
        else:
            current_p = self._as_tensor(branch_prob).reshape(-1)
            if current_p.shape[0] != cand_t.shape[0]:
                raise ValueError("branch_prob must match the number of candidate covariates.")
        batch = max(int(chunk_size or 0), 1)
        scores = []
        for start in range(0, cand_t.shape[0], batch):
            stop = min(start + batch, cand_t.shape[0])
            query_x = cand_t[start:stop]
            query_z = inst_t[start:stop]
            chunk_p = current_p[start:stop]
            query_aug_0 = self._candidate_augmented_from_branch(query_x, query_z, assumed_t=0)
            query_aug_1 = self._candidate_augmented_from_branch(query_x, query_z, assumed_t=1)
            post_var_0 = self._fantasy_cate_variance_from_query_aug_batch(target_t, query_aug_0)
            post_var_1 = self._fantasy_cate_variance_from_query_aug_batch(target_t, query_aug_1)
            post_var_0 = post_var_0 + nugget
            post_var_1 = post_var_1 + nugget
            post_entropy_0 = 0.5 * torch.log(torch.clamp(post_var_0, min=1e-12))
            post_entropy_1 = 0.5 * torch.log(torch.clamp(post_var_1, min=1e-12))
            branch_entropy = (
                (1.0 - chunk_p).unsqueeze(-1) * post_entropy_0
                + chunk_p.unsqueeze(-1) * post_entropy_1
            )
            scores.append(torch.mean(prior_entropy.unsqueeze(0) - branch_entropy, dim=1))
        return torch.cat(scores, dim=0).cpu().numpy()

    @torch.no_grad()
    def krr_particle_ivig_scores(self,
                                 candidate_covariate: Union[np.ndarray, torch.Tensor],
                                 instrument: Union[np.ndarray, torch.Tensor, float, int],
                                 target_covariate: Union[np.ndarray, torch.Tensor],
                                 prior_cate_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                 branch_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                 variance_nugget: Union[np.ndarray, torch.Tensor, float, int] = 0.0,
                                 n_particles: int = 4,
                                 chunk_size: int = 128) -> np.ndarray:
        """
        KRR-posterior-particle IVIG diagnostic.

        For n_particles=1 this returns the existing empirical-Bayes GPIV-IVIG
        score exactly.  For n_particles>1, the second-stage GPIV posterior is
        kept fixed and the first-stage KRR working posterior is sampled at the
        candidate branch. This diagnostic averages conditional scores over
        these particles to assess acquisition rankings with the same GPIV
        estimator backbone.
        """
        n_particles = max(int(n_particles), 1)
        if n_particles <= 1:
            return self.phantom_ivig_scores(
                candidate_covariate,
                instrument,
                target_covariate,
                prior_cate_var=prior_cate_var,
                branch_prob=branch_prob,
                variance_nugget=variance_nugget,
                chunk_size=chunk_size,
            )

        cand_t = self._ensure_2d_covariate(candidate_covariate)
        inst_t = self._ensure_2d_instrument(instrument, cand_t.shape[0])
        target_t = self._ensure_2d_covariate(target_covariate)

        if prior_cate_var is None:
            _, prior_cate_var = self.predict_cate_with_variance(target_t)
        prior_var = self._as_tensor(prior_cate_var).reshape(-1)
        nugget = torch.clamp(self._as_tensor(variance_nugget), min=0.0).reshape(-1)
        if nugget.numel() == 1:
            prior_var = prior_var + nugget
        elif nugget.shape[0] == prior_var.shape[0]:
            prior_var = prior_var + nugget
        else:
            raise ValueError("variance_nugget must be scalar or match the number of target covariates.")
        prior_entropy = 0.5 * torch.log(torch.clamp(prior_var, min=1e-12))

        if branch_prob is None:
            current_mean, current_var = self._krr_query_raw_mean_var(cand_t, inst_t)
            eps = np.random.randn(n_particles, cand_t.shape[0])
            eps_t = torch.as_tensor(eps, device=self.device, dtype=self.dtype)
            current_p_particles = torch.clamp(
                current_mean.unsqueeze(0) + torch.sqrt(current_var).unsqueeze(0) * eps_t,
                min=0.0,
                max=1.0,
            )
        else:
            current_p = self._as_tensor(branch_prob).reshape(-1)
            if current_p.shape[0] != cand_t.shape[0]:
                raise ValueError("branch_prob must match the number of candidate covariates.")
            current_p_particles = current_p.unsqueeze(0).repeat(n_particles, 1)

        branch_samples = []
        for assumed_t in (0, 1):
            branch_mean, branch_var = self._phantom_query_propensity_raw_var_batch(
                cand_t,
                inst_t,
                assumed_t=assumed_t,
            )
            eps = np.random.randn(n_particles, cand_t.shape[0])
            eps_t = torch.as_tensor(eps, device=self.device, dtype=self.dtype)
            p_branch = torch.clamp(
                branch_mean.unsqueeze(0) + torch.sqrt(branch_var).unsqueeze(0) * eps_t,
                min=0.0,
                max=1.0,
            )
            branch_samples.append(p_branch)

        # Evaluate each first-stage particle and average its conditional score.
        batch = max(int(chunk_size or 0), 1)
        scores = []
        for start in range(0, cand_t.shape[0], batch):
            stop = min(start + batch, cand_t.shape[0])
            bsz = stop - start
            query_x = cand_t[start:stop]
            chunk_score = torch.zeros(bsz, device=self.device, dtype=self.dtype)
            for particle_idx in range(n_particles):
                post_entropies = []
                for p_branch in branch_samples:
                    p_chunk = p_branch[particle_idx, start:stop]
                    pi_chunk = torch.stack([1.0 - p_chunk, p_chunk], dim=-1)
                    query_aug = torch.cat([query_x, pi_chunk], dim=-1)
                    post_var = self._fantasy_cate_variance_from_query_aug_batch(target_t, query_aug)
                    post_var = post_var + nugget
                    post_entropies.append(0.5 * torch.log(torch.clamp(post_var, min=1e-12)))

                p_current = current_p_particles[particle_idx, start:stop].unsqueeze(-1)
                branch_entropy = (
                    (1.0 - p_current) * post_entropies[0]
                    + p_current * post_entropies[1]
                )
                particle_score = torch.mean(
                    prior_entropy.reshape(1, -1) - branch_entropy,
                    dim=-1,
                )
                chunk_score = chunk_score + particle_score
            scores.append(chunk_score / float(n_particles))
        return torch.cat(scores, dim=0).cpu().numpy()

    @torch.no_grad()
    def y_ivig_scores(self,
                      candidate_covariate: Union[np.ndarray, torch.Tensor],
                      instrument: Union[np.ndarray, torch.Tensor, float, int],
                      target_covariate: Union[np.ndarray, torch.Tensor],
                      branch_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      chunk_size: int = 64,
                      y_quad_nodes: int = 5,
                      tau_quad_nodes: int = 9) -> np.ndarray:
        """
        Theorem-aligned outcome-only IVIG score.

        This approximates E_{x*}[I(Y; tau(x*) | x, z, H)] when T is not
        observed. For each candidate, p(tau(x*) | y, x, z, H) is represented
        as a two-branch Gaussian mixture with responsibility
        P(T=1 | Y=y, x, z, H). The marginal mixture entropy is approximated
        by Gauss-Hermite quadrature over both Y and tau.
        """
        cand_t = self._ensure_2d_covariate(candidate_covariate)
        inst_t = self._ensure_2d_instrument(instrument, cand_t.shape[0])
        target_t = self._ensure_2d_covariate(target_covariate)

        if branch_prob is None:
            current_p = self._predict_propensity_batch(cand_t, inst_t)
        else:
            current_p = self._as_tensor(branch_prob).reshape(-1)
            if current_p.shape[0] != cand_t.shape[0]:
                raise ValueError("branch_prob must match the number of candidate covariates.")
        current_p = torch.clamp(current_p, min=1e-6, max=1.0 - 1e-6)

        batch = max(int(chunk_size or 0), 1)
        y_nodes, y_weights = self._gauss_hermite_rule(y_quad_nodes)
        tau_nodes, tau_weights = self._gauss_hermite_rule(tau_quad_nodes)
        scores = []

        for start in range(0, cand_t.shape[0], batch):
            stop = min(start + batch, cand_t.shape[0])
            query_x = cand_t[start:stop]
            query_z = inst_t[start:stop]
            p1 = current_p[start:stop]
            p0 = 1.0 - p1

            query_aug_0 = self._candidate_augmented_from_branch(query_x, query_z, assumed_t=0)
            query_aug_1 = self._candidate_augmented_from_branch(query_x, query_z, assumed_t=1)
            mean_tau, tau_var, mean_y0, var_y0, beta0, post_var0 = (
                self._tau_query_update_stats_from_query_aug_batch(target_t, query_aug_0)
            )
            _, _, mean_y1, var_y1, beta1, post_var1 = (
                self._tau_query_update_stats_from_query_aug_batch(target_t, query_aug_1)
            )

            prior_entropy = 0.5 * torch.log(2.0 * torch.pi * torch.e * torch.clamp(tau_var, min=1e-12))
            sqrt_y0 = torch.sqrt(2.0 * torch.clamp(var_y0, min=1e-12))
            sqrt_y1 = torch.sqrt(2.0 * torch.clamp(var_y1, min=1e-12))
            y0 = mean_y0.unsqueeze(-1) + sqrt_y0.unsqueeze(-1) * y_nodes.unsqueeze(0)
            y1 = mean_y1.unsqueeze(-1) + sqrt_y1.unsqueeze(-1) * y_nodes.unsqueeze(0)

            y_values = torch.cat([y0, y1], dim=1)
            y_mix_weights = torch.cat([
                p0.unsqueeze(-1) * y_weights.unsqueeze(0),
                p1.unsqueeze(-1) * y_weights.unsqueeze(0),
            ], dim=1)

            log_p0 = torch.log(torch.clamp(p0, min=1e-12)).unsqueeze(-1)
            log_p1 = torch.log(torch.clamp(p1, min=1e-12)).unsqueeze(-1)
            log_y0 = self._normal_logpdf(y_values, mean_y0.unsqueeze(-1), var_y0.unsqueeze(-1)) + log_p0
            log_y1 = self._normal_logpdf(y_values, mean_y1.unsqueeze(-1), var_y1.unsqueeze(-1)) + log_p1
            log_norm = torch.logsumexp(torch.stack([log_y0, log_y1], dim=0), dim=0)
            rho1 = torch.exp(log_y1 - log_norm)
            rho1 = torch.clamp(rho1, min=1e-6, max=1.0 - 1e-6)
            rho0 = 1.0 - rho1

            y_center0 = y_values.unsqueeze(-1) - mean_y0[:, None, None]
            y_center1 = y_values.unsqueeze(-1) - mean_y1[:, None, None]
            mu0 = mean_tau[None, None, :] + beta0[:, None, :] * y_center0
            mu1 = mean_tau[None, None, :] + beta1[:, None, :] * y_center1
            v0 = torch.clamp(post_var0[:, None, :], min=1e-12)
            v1 = torch.clamp(post_var1[:, None, :], min=1e-12)

            x0 = mu0.unsqueeze(-1) + torch.sqrt(2.0 * v0).unsqueeze(-1) * tau_nodes
            x1 = mu1.unsqueeze(-1) + torch.sqrt(2.0 * v1).unsqueeze(-1) * tau_nodes

            log_rho0 = torch.log(torch.clamp(rho0, min=1e-12)).unsqueeze(-1).unsqueeze(-1)
            log_rho1 = torch.log(torch.clamp(rho1, min=1e-12)).unsqueeze(-1).unsqueeze(-1)
            log_q_x0 = torch.logsumexp(
                torch.stack([
                    log_rho0 + self._normal_logpdf(x0, mu0.unsqueeze(-1), v0.unsqueeze(-1)),
                    log_rho1 + self._normal_logpdf(x0, mu1.unsqueeze(-1), v1.unsqueeze(-1)),
                ], dim=0),
                dim=0,
            )
            log_q_x1 = torch.logsumexp(
                torch.stack([
                    log_rho0 + self._normal_logpdf(x1, mu0.unsqueeze(-1), v0.unsqueeze(-1)),
                    log_rho1 + self._normal_logpdf(x1, mu1.unsqueeze(-1), v1.unsqueeze(-1)),
                ], dim=0),
                dim=0,
            )

            e_log_q0 = torch.sum(tau_weights * log_q_x0, dim=-1)
            e_log_q1 = torch.sum(tau_weights * log_q_x1, dim=-1)
            mixture_entropy = -(rho0.unsqueeze(-1) * e_log_q0 + rho1.unsqueeze(-1) * e_log_q1)
            expected_post_entropy = torch.sum(y_mix_weights.unsqueeze(-1) * mixture_entropy, dim=1)
            scores.append(torch.mean(prior_entropy.unsqueeze(0) - expected_post_entropy, dim=1))

        return torch.cat(scores, dim=0).cpu().numpy()

    @torch.no_grad()
    def direct_tau_ig_scores(self,
                             candidate_covariate: Union[np.ndarray, torch.Tensor],
                             instrument: Union[np.ndarray, torch.Tensor, float, int],
                             target_covariate: Union[np.ndarray, torch.Tensor],
                             prior_cate_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
                             chunk_size: int = 128) -> np.ndarray:
        """No-Stage-I diagnostic: treat the assigned instrument as the realised treatment."""
        cand_t = self._ensure_2d_covariate(candidate_covariate)
        inst_t = self._ensure_2d_instrument(instrument, cand_t.shape[0])
        target_t = self._ensure_2d_covariate(target_covariate)

        if prior_cate_var is None:
            _, prior_cate_var = self.predict_cate_with_variance(target_t)
        prior_var = self._as_tensor(prior_cate_var).reshape(-1)
        prior_entropy = 0.5 * torch.log(torch.clamp(prior_var, min=1e-12))

        batch = max(int(chunk_size or 0), 1)
        scores = []
        for start in range(0, cand_t.shape[0], batch):
            stop = min(start + batch, cand_t.shape[0])
            query_x = cand_t[start:stop]
            query_z = inst_t[start:stop]
            query_aug = self._candidate_augmented_direct_tau(query_x, query_z)
            post_var = self._fantasy_cate_variance_from_query_aug_batch(target_t, query_aug)
            post_entropy = 0.5 * torch.log(torch.clamp(post_var, min=1e-12))
            scores.append(torch.mean(prior_entropy.unsqueeze(0) - post_entropy, dim=1))
        return torch.cat(scores, dim=0).cpu().numpy()

    @torch.no_grad()
    def sanitized_bald_scores(self,
                              candidate_covariate: Union[np.ndarray, torch.Tensor],
                              instrument: Union[np.ndarray, torch.Tensor, float, int]) -> np.ndarray:
        cand_t = self._ensure_2d_covariate(candidate_covariate)
        inst_t = self._ensure_2d_instrument(instrument, cand_t.shape[0])

        query_aug = self._candidate_augmented_sanitized(cand_t, inst_t)
        latent_var = self._as_tensor(self._latent_query_variance_from_aug(query_aug)).reshape(-1)
        noise_var = torch.as_tensor(max(self.noise_var, 1e-12), device=self.device, dtype=self.dtype)
        scores = 0.5 * torch.log1p(latent_var / noise_var)
        return scores.cpu().numpy()

    def evaluate(self, test_data: TestDataSet) -> float:
        """Compute PEHE against true structural CATE."""
        pred = self.predict_cate(test_data.covariate)
        return weighted_sqrt_pehe_np(
            pred,
            test_data.structural,
            getattr(test_data, "sample_weight", None),
        )


class GPIVTrainer:
    """
    Trainer for the GPyTorch-based GPIV model.

        trainer = GPIVTrainer({"n_epoch": 200, "lr": 0.05}, gpu_flg=True)
        pehe = trainer.train(rand_seed, train_data=..., test_data=...)
        trainer.mdl  # fitted GPIVModel used by the acquisition functions
    """

    def __init__(self, train_params: Optional[Dict[str, Any]] = None, gpu_flg: bool = False):
        train_params = dict(train_params or {})
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_flg else "cpu")

        # Training hyperparameters
        self.n_epoch = train_params.get("n_epoch", 200)
        self.lr = train_params.get("lr", 0.05)
        self.lengthscale_init = train_params.get("lengthscale_init")
        self.propensity_lengthscale_init = train_params.get(
            "propensity_lengthscale_init", self.lengthscale_init
        )
        self.outcome_lengthscale_init = train_params.get(
            "outcome_lengthscale_init", self.lengthscale_init
        )

        self.mdl: Optional[GPIVModel] = None

    @staticmethod
    def _median_pairwise_distance(features: torch.Tensor,
                                  max_points: int = 512) -> float:
        """Median-distance kernel initialization for high-dimensional inputs."""
        if features.shape[0] < 2:
            return float(np.sqrt(features.shape[1]))
        sample = features[:max_points]
        distances = torch.pdist(sample)
        distances = distances[distances > 1e-8]
        if distances.numel() == 0:
            return float(np.sqrt(features.shape[1]))
        return float(torch.median(distances).detach().cpu().item())

    def _resolve_lengthscale_init(self, spec: Any, features: torch.Tensor) -> Optional[float]:
        if spec is None:
            return None
        if isinstance(spec, (int, float)):
            return float(spec)

        raw = str(spec).strip().lower()
        if raw in {"", "none", "default", "false"}:
            return None
        if raw in {"sqrt_dim", "sqrt-dim", "sqrt"}:
            return float(np.sqrt(features.shape[1]))
        if raw in {"2sqrt_dim", "2sqrt-dim", "two_sqrt_dim", "two-sqrt-dim"}:
            return float(2.0 * np.sqrt(features.shape[1]))
        if raw in {"median", "pairwise_median", "median_pairwise"}:
            return self._median_pairwise_distance(features)
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                "Unsupported GPIV lengthscale_init value "
                f"{spec!r}. Use a float, 'sqrt_dim', '2sqrt_dim', or 'median'."
            ) from exc

    def train(self, rand_seed: int = 42, verbose: int = 0,
              train_data: Optional[TrainDataSet] = None,
              test_data: Optional[TestDataSet] = None) -> float:
        """
        Train the GPIV model and return sqrt-PEHE on test data.
        """
        # ── 1. Data ──────────────────────────────────────────────────────
        if train_data is None or test_data is None:
            raise ValueError("GPIVTrainer.train requires explicit train_data and test_data.")

        dtype = torch.float64
        device = self.device

        X = torch.tensor(train_data.covariate, dtype=dtype, device=device)        # [N, d]
        Z = torch.tensor(train_data.instrumental, dtype=dtype, device=device)      # [N, 1]
        T = torch.tensor(train_data.treatment, dtype=dtype, device=device)         # [N, 1]
        Y_raw = torch.tensor(np.asarray(train_data.outcome).reshape(-1), dtype=dtype, device=device)  # [N]

        N = X.shape[0]
        if N < 2:
            raise ValueError("GPIV needs at least two acquired observations.")
        if not all(bool(torch.isfinite(v).all()) for v in (X, Z, T, Y_raw)):
            raise ValueError("Training inputs and outcomes must be finite.")
        if not (Z.shape == T.shape == (N, 1)) or Y_raw.shape != (N,):
            raise ValueError("Training arrays have incompatible shapes.")
        d_covariate = X.shape[1]

        # ── 2. Standardise Y ─────────────────────────────────────────────
        y_mean = Y_raw.mean().item()
        y_std = Y_raw.std().item()
        if y_std < 1e-6:
            y_std = 1.0
        Y = (Y_raw - y_mean) / y_std  # [N]

        # ── 3. Initialise Modules ────────────────────────────────────────
        input_dim_krr = d_covariate + 1  # [X, Z]
        XZ = torch.cat([X, Z], dim=-1)
        propensity_ls = self._resolve_lengthscale_init(
            self.propensity_lengthscale_init, XZ
        )
        outcome_ls = self._resolve_lengthscale_init(
            self.outcome_lengthscale_init, X
        )
        propensity_module = PropensityKRR(
            input_dim_krr, init_lengthscale=propensity_ls
        ).to(device).to(dtype)

        # Initial propensity computation (for GP data)
        with torch.no_grad():
            Pi_init = propensity_module(X, Z, T)  # [N, 2]

        # Likelihood
        likelihood = GaussianLikelihood().to(device).to(dtype)

        # ExactGP — takes training X_aug and Y
        X_aug_init = torch.cat([X, Pi_init], dim=-1)
        gp = GPIVExactGP(
            X_aug_init, Y, likelihood, d_covariate,
            init_lengthscale=outcome_ls,
        ).to(device).to(dtype)

        # ── 4. Joint MLL Optimisation ────────────────────────────────────
        gp.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(likelihood, gp)

        # Collect ALL learnable parameters (Phase I + Phase II + noise)
        # Use set-based dedup to avoid "appears in more than one group"
        seen_ids = set()
        all_param_list = []
        for module in [propensity_module, gp, likelihood]:
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_param_list.append(p)
        optimizer = torch.optim.Adam(all_param_list, lr=self.lr)

        if verbose >= 1:
            logger.info(f"GPIV Training: {self.n_epoch} epochs, lr={self.lr}, "
                        f"N={N}, d={d_covariate}")
            if propensity_ls is not None or outcome_ls is not None:
                logger.info(
                    "GPIV lengthscale init: "
                    f"propensity={propensity_ls}, outcome={outcome_ls}"
                )

        for epoch in range(self.n_epoch):
            optimizer.zero_grad()

            # Recompute propensities (differentiable)
            Pi = propensity_module(X, Z, T)

            # Augment inputs
            X_aug = torch.cat([X, Pi], dim=-1)

            # Update ExactGP's training inputs dynamically to bypass strict equality check
            gp.set_train_data(inputs=X_aug, targets=Y, strict=False)

            # Forward pass → MLL
            output = gp(X_aug)
            loss = -mll(output, Y)

            loss.backward()
            optimizer.step()

            if verbose >= 2 and (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    ek = gp.expected_kernel
                    sigma0_sq = torch.nn.functional.softplus(ek._raw_sigma0_sq).item()
                    sigma1_sq = torch.nn.functional.softplus(ek._raw_sigma1_sq).item()
                    rho = torch.tanh(ek._raw_rho).item()
                    eta = propensity_module.eta.item()
                    noise = likelihood.noise.item()
                logger.info(
                    f"  Epoch {epoch+1:3d} | loss={loss.item():.4f} | "
                    f"σ₀²={sigma0_sq:.3f} σ₁²={sigma1_sq:.3f} ρ={rho:.3f} | "
                    f"η={eta:.4f} noise={noise:.4f}"
                )

        # ── 5. Build GPIVModel for Inference ─────────────────────────────
        gp.eval()
        likelihood.eval()

        with torch.no_grad():
            Pi_final = propensity_module(X, Z, T)
            X_aug_final = torch.cat([X, Pi_final], dim=-1)
            
        # Update ExactGP's internal training data cache for exact inference
        gp.set_train_data(inputs=X_aug_final, targets=Y, strict=False)

        self.mdl = GPIVModel(
            gp=gp, likelihood=likelihood,
            propensity_module=propensity_module,
            X_train=X, Z_train=Z, T_train=T, Y_train=Y,
            Pi_train=Pi_final,
            y_mean=y_mean, y_std=y_std,
        )

        # ── 6. Evaluate ──────────────────────────────────────────────────
        return self.mdl.evaluate(test_data)

    def evaluate(self, test_data) -> float:
        """Evaluate sqrt-PEHE (effect RMSE) on test data."""
        return self.mdl.evaluate(test_data)

    def evaluate_cate(self, test_data) -> float:
        """Evaluate sqrt-PEHE (same as evaluate for this model)."""
        pred = self.mdl.predict_cate(test_data.covariate)
        return weighted_sqrt_pehe_np(
            pred,
            test_data.structural,
            getattr(test_data, "sample_weight", None),
        )
