"""
IndPolicy baseline: a learned contextual instrument-assignment policy adapted
from "Adaptive Instrument Design for Indirect Experiments" (Chandak et al.,
ICLR 2024).  Covariates are drawn uniformly at random from the pool and the
instrument is assigned by a policy trained on a sanitized DeepIV-style
surrogate (propensity net + arm-separable outcome net + behaviour net).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.library.datasets.data_class import TrainDataSet, TestDataSet, weighted_sqrt_pehe_np

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers = []
    dim = input_dim
    for _ in range(max(depth, 1)):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class BinaryPropensityNet(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(x_dim + 1, hidden_dim, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))


class BinaryOutcomeNet(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(x_dim + 1, hidden_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class BinaryPolicyNet(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int):
        super().__init__()
        self.net = build_mlp(x_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IndPolicyModel:
    """
    Inference wrapper for the IndPolicy surrogate.

    The estimator is a sanitized binary DeepIV-style model:
      - nuisance: p(T=1 | X, Z)
      - response: E[Y | X, T]
      - observed outcome prediction: E_Y[Y | X, Z] = mu_1(X) p + mu_0(X) (1-p)
      - CATE prediction: mu_1(X) - mu_0(X)
    """

    def __init__(self,
                 propensity_net: BinaryPropensityNet,
                 outcome_net: BinaryOutcomeNet,
                 policy_net: BinaryPolicyNet,
                 y_mean: float,
                 y_std: float,
                 device: torch.device):
        self.propensity_net = propensity_net
        self.outcome_net = outcome_net
        self.policy_net = policy_net
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.device = device

    def _to_x_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_cate(self, x: np.ndarray) -> np.ndarray:
        x_t = self._to_x_tensor(x)
        ones = torch.ones((x_t.shape[0], 1), dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(ones)
        mu1 = self.outcome_net(x_t, ones)
        mu0 = self.outcome_net(x_t, zeros)
        cate = (mu1 - mu0) * self.y_std
        return cate.cpu().numpy()

    @torch.no_grad()
    def policy_probability(self, x: np.ndarray) -> np.ndarray:
        x_t = self._to_x_tensor(x)
        prob = torch.sigmoid(self.policy_net(x_t))
        return prob.cpu().numpy()

    def assign_instruments(self, x: np.ndarray, stochastic: bool = True) -> np.ndarray:
        prob = self.policy_probability(x).reshape(-1)
        if stochastic:
            return np.random.binomial(1, prob).astype(int)
        return (prob >= 0.5).astype(int)

class IndPolicyTrainer:
    """
    Learned instrument-assignment policy (IndPolicy baseline).

      1. New covariate points are selected uniformly at random from the remaining pool.
      2. A learned contextual policy q(Z=1|X) assigns instruments to that random batch.
      3. The surrogate is a sanitized CIV/DeepIV-style model trained on the same
         acquired data as the downstream GPIV estimator.
    """

    def __init__(self, train_params: Optional[Dict[str, Any]] = None, gpu_flg: bool = False):
        train_params = dict(train_params or {})
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_flg else "cpu")

        self.hidden_dim = train_params.get("hidden_dim", 64)
        self.batch_size = train_params.get("batch_size", 64)
        self.lr = train_params.get("lr", 1e-3)
        self.weight_decay = train_params.get("weight_decay", 1e-4)
        self.propensity_epochs = train_params.get("propensity_epochs", 100)
        self.outcome_epochs = train_params.get("outcome_epochs", 150)
        self.behavior_epochs = train_params.get("behavior_epochs", 80)
        self.policy_epochs = train_params.get("policy_epochs", 120)
        self.target_jacobian_samples = train_params.get("target_jacobian_samples", 16)
        self.propensity_loss_weight = train_params.get("propensity_loss_weight", 0.5)
        self.policy_mix_alpha = train_params.get("policy_mix_alpha", 0.7)
        self.policy_entropy_weight = train_params.get("policy_entropy_weight", 0.01)
        self.stochastic_assignment = train_params.get("stochastic_assignment", True)

        self.mdl: Optional[IndPolicyModel] = None
        self.propensity_net: Optional[BinaryPropensityNet] = None
        self.outcome_net: Optional[BinaryOutcomeNet] = None
        self.policy_net: Optional[BinaryPolicyNet] = None
        self.behavior_net: Optional[BinaryPolicyNet] = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _init_modules(self, x_dim: int):
        self.propensity_net = BinaryPropensityNet(x_dim, self.hidden_dim).to(self.device)
        self.outcome_net = BinaryOutcomeNet(x_dim, self.hidden_dim).to(self.device)
        self.policy_net = BinaryPolicyNet(x_dim, self.hidden_dim).to(self.device)
        self.behavior_net = BinaryPolicyNet(x_dim, self.hidden_dim).to(self.device)

    def _utility_params(self) -> List[torch.nn.Parameter]:
        assert self.propensity_net is not None and self.outcome_net is not None
        return list(self.propensity_net.parameters()) + list(self.outcome_net.parameters())

    @staticmethod
    def _flatten_grads(grads: List[Optional[torch.Tensor]], params: List[torch.nn.Parameter]) -> torch.Tensor:
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        return torch.cat(flat)

    def _fit_propensity(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        assert self.propensity_net is not None
        dataset = TensorDataset(x, z, t)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, x.shape[0]), shuffle=True)
        opt = torch.optim.Adam(self.propensity_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.propensity_net.train()
        for _ in range(self.propensity_epochs):
            for xb, zb, tb in loader:
                opt.zero_grad()
                logits = self.propensity_net(xb, zb)
                loss = F.binary_cross_entropy_with_logits(logits, tb)
                loss.backward()
                opt.step()

    def _fit_outcome(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        assert self.propensity_net is not None and self.outcome_net is not None
        dataset = TensorDataset(x, z, y)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, x.shape[0]), shuffle=True)
        opt = torch.optim.Adam(self.outcome_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.propensity_net.eval()
        self.outcome_net.train()
        for _ in range(self.outcome_epochs):
            for xb, zb, yb in loader:
                opt.zero_grad()
                with torch.no_grad():
                    p = torch.sigmoid(self.propensity_net(xb, zb))
                ones = torch.ones_like(zb)
                zeros = torch.zeros_like(zb)
                mu1 = self.outcome_net(xb, ones)
                mu0 = self.outcome_net(xb, zeros)
                y_hat = mu1 * p + mu0 * (1.0 - p)
                loss = F.mse_loss(y_hat, yb)
                loss.backward()
                opt.step()

    def _fit_behavior(self, x: torch.Tensor, z: torch.Tensor):
        assert self.behavior_net is not None
        dataset = TensorDataset(x, z)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, x.shape[0]), shuffle=True)
        opt = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.behavior_net.train()
        for _ in range(self.behavior_epochs):
            for xb, zb in loader:
                opt.zero_grad()
                logits = self.behavior_net(xb)
                loss = F.binary_cross_entropy_with_logits(logits, zb)
                loss.backward()
                opt.step()

    def _target_signal(self, x: torch.Tensor) -> torch.Tensor:
        assert self.propensity_net is not None and self.outcome_net is not None
        n = x.shape[0]
        ones = torch.ones((n, 1), dtype=x.dtype, device=x.device)
        zeros = torch.zeros_like(ones)
        p1 = torch.sigmoid(self.propensity_net(x, ones))
        p0 = torch.sigmoid(self.propensity_net(x, zeros))
        mu11 = self.outcome_net(x, ones)
        mu10 = self.outcome_net(x, zeros)
        m1 = mu11 * p1 + mu10 * (1.0 - p1)
        m0 = mu11 * p0 + mu10 * (1.0 - p0)
        return m1 - m0

    def _compute_sample_utilities(self,
                                  x: torch.Tensor,
                                  z: torch.Tensor,
                                  t: torch.Tensor,
                                  y: torch.Tensor,
                                  target_x: torch.Tensor) -> torch.Tensor:
        assert self.propensity_net is not None and self.outcome_net is not None
        params = self._utility_params()
        self.propensity_net.eval()
        self.outcome_net.eval()

        target_count = min(self.target_jacobian_samples, target_x.shape[0])
        target_idx = torch.randperm(target_x.shape[0], device=target_x.device)[:target_count]
        target_subset = target_x[target_idx]

        target_jacs = []
        for j in range(target_subset.shape[0]):
            signal = self._target_signal(target_subset[j:j+1]).squeeze()
            grads = torch.autograd.grad(signal, params, retain_graph=False, create_graph=False, allow_unused=True)
            target_jacs.append(self._flatten_grads(list(grads), params).detach())
        J = torch.stack(target_jacs, dim=0)  # [M, P]

        utilities = []
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            zi = z[i:i+1]
            ti = t[i:i+1]
            yi = y[i:i+1]

            prop_logit = self.propensity_net(xi, zi)
            prop_prob = torch.sigmoid(prop_logit)
            mu1 = self.outcome_net(xi, torch.ones_like(zi))
            mu0 = self.outcome_net(xi, torch.zeros_like(zi))
            y_hat = mu1 * prop_prob + mu0 * (1.0 - prop_prob)
            outcome_loss = F.mse_loss(y_hat, yi)
            prop_loss = F.binary_cross_entropy_with_logits(prop_logit, ti)
            sample_loss = outcome_loss + self.propensity_loss_weight * prop_loss

            grads = torch.autograd.grad(sample_loss, params, retain_graph=False, create_graph=False, allow_unused=True)
            g = self._flatten_grads(list(grads), params).detach()
            utility = torch.mean((J @ g) ** 2)
            utilities.append(utility)

        utilities = torch.stack(utilities).reshape(-1, 1)
        utilities = utilities / (utilities.mean() + 1e-6)
        return utilities.detach()

    def _fit_policy(self, x: torch.Tensor, z: torch.Tensor, sample_utilities: torch.Tensor):
        assert self.behavior_net is not None and self.policy_net is not None
        opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.behavior_net.eval()
        with torch.no_grad():
            beta_prob = torch.sigmoid(self.behavior_net(x))

        self.policy_net.train()
        for _ in range(self.policy_epochs):
            opt.zero_grad()
            policy_prob = torch.sigmoid(self.policy_net(x))
            obs_prob_policy = torch.where(z > 0.5, policy_prob, 1.0 - policy_prob)
            obs_prob_beta = torch.where(z > 0.5, beta_prob, 1.0 - beta_prob)
            pi_eff = self.policy_mix_alpha * obs_prob_beta + (1.0 - self.policy_mix_alpha) * obs_prob_policy
            entropy = -(policy_prob * torch.log(policy_prob + 1e-6) +
                        (1.0 - policy_prob) * torch.log(1.0 - policy_prob + 1e-6))
            loss = -(sample_utilities * torch.log(pi_eff + 1e-6)).mean()
            loss = loss - self.policy_entropy_weight * entropy.mean()
            loss.backward()
            opt.step()

    def train(self, rand_seed: int = 42, verbose: int = 0,
              train_data: Optional[TrainDataSet] = None,
              test_data: Optional[TestDataSet] = None) -> float:
        if train_data is None or test_data is None:
            raise ValueError("IndPolicyTrainer requires explicit train_data and test_data.")

        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)

        x = torch.tensor(train_data.covariate, dtype=torch.float32, device=self.device)
        z = torch.tensor(train_data.instrumental, dtype=torch.float32, device=self.device)
        t = torch.tensor(train_data.treatment, dtype=torch.float32, device=self.device)
        y_raw = torch.tensor(train_data.outcome, dtype=torch.float32, device=self.device)
        target_x = torch.tensor(test_data.covariate, dtype=torch.float32, device=self.device)

        self.y_mean = float(y_raw.mean().item())
        self.y_std = float(y_raw.std().item())
        if self.y_std < 1e-6:
            self.y_std = 1.0
        y = (y_raw - self.y_mean) / self.y_std

        self._init_modules(x.shape[1])
        self._fit_propensity(x, z, t)
        self._fit_outcome(x, z, y)
        self._fit_behavior(x, z)
        utilities = self._compute_sample_utilities(x, z, t, y, target_x)
        self._fit_policy(x, z, utilities)

        assert self.propensity_net is not None
        assert self.outcome_net is not None
        assert self.policy_net is not None
        self.propensity_net.eval()
        self.outcome_net.eval()
        self.policy_net.eval()
        self.mdl = IndPolicyModel(
            propensity_net=self.propensity_net,
            outcome_net=self.outcome_net,
            policy_net=self.policy_net,
            y_mean=self.y_mean,
            y_std=self.y_std,
            device=self.device,
        )
        return self.evaluate_cate(test_data)

    def evaluate_cate(self, test_data: TestDataSet) -> float:
        assert self.mdl is not None
        pred = self.mdl.predict_cate(test_data.covariate)
        return weighted_sqrt_pehe_np(
            pred,
            test_data.structural,
            getattr(test_data, "sample_weight", None),
        )

    def assign_instruments(self, x: np.ndarray) -> np.ndarray:
        assert self.mdl is not None
        return self.mdl.assign_instruments(x, stochastic=self.stochastic_assignment)
