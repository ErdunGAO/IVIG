import os
import random

import numpy as np
import torch

from src.library.models import GPIVTrainer, IndPolicyTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Hyper-parameters used for all experiments in the paper.
GPIV_DEFAULTS = {"n_epoch": 200, "lr": 0.05, "lengthscale_init": None}

INDPOLICY_DEFAULTS = {
    "hidden_dim": 64,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "propensity_epochs": 80,
    "outcome_epochs": 120,
    "behavior_epochs": 60,
    "policy_epochs": 100,
    "target_jacobian_samples": 16,
    "propensity_loss_weight": 0.5,
    "policy_mix_alpha": 0.7,
    "policy_entropy_weight": 0.01,
    "stochastic_assignment": True,
}


def train_gpiv(train_data, test_data, config, seed):
    params = dict(GPIV_DEFAULTS)
    params.update(config.get("gpiv", {}))
    trainer = GPIVTrainer(params, gpu_flg=config.get("gpu", True))
    trainer.train(rand_seed=seed, train_data=train_data, test_data=test_data, verbose=0)
    return trainer


def train_indpolicy(train_data, test_data, config, seed):
    params = dict(INDPOLICY_DEFAULTS)
    params.update(config.get("indpolicy", {}))
    trainer = IndPolicyTrainer(params, gpu_flg=config.get("gpu", True))
    trainer.train(rand_seed=seed, train_data=train_data, test_data=test_data, verbose=0)
    return trainer
