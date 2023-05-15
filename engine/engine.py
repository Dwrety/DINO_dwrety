# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
from typing import Iterable
import itertools
from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_iters,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            eta_min = 0,
            last_epoch=-1,
    ):

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.max_iters = max_iters
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / self.max_iters)) / 2
                for base_lr in self.base_lrs
            ]


class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
            self,
            optimizer,
            max_iters,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            eta_min = 0,
            last_epoch=-1,
            patience = 5,
            verbose = False,
    ):    

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.eta_min = eta_min

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, factor=gamma, patience=patience, mode='max', min_lr=eta_min, verbose = verbose)

    def step(self, metrics=None):
        warmup_factor = 1

        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            
            if self.last_epoch >= self.warmup_iters-1:
                warmup_factor = 1.0
                
            warmup_lrs = [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]

            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group['lr'] = lr
            
            self.last_epoch += 1
        elif metrics:
            super().step(metrics)


def make_optimizer(cfg, model):
    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS_VALUE
        enable = (
                cfg.SOLVER.CLIP_GRADIENTS_ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS_TYPE == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # different lr schedule
        if "language_backbone" in key:
            lr = cfg.SOLVER.LANG_LR

        if "backbone.body" in key and "language_backbone.body" not in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_BODY_LR_FACTOR

        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if 'norm' in key or 'Norm' in key:
            weight_decay *= cfg.SOLVER.WEIGHT_DECAY_NORM_FACTOR
            print("Setting weight decay of {} to {}".format(key, weight_decay))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(params, lr, momentum=cfg.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, lr)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.MULTI_MAX_EPOCH:
        assert len(cfg.SOLVER.MULTI_MAX_EPOCH) == len(cfg.SOLVER.STEPS)
        lr_scheduler = []

        for stage_step, stage_max_epoch in zip(cfg.SOLVER.STEPS, cfg.SOLVER.MULTI_MAX_ITER):
            milestones = []
            for step in stage_step:
                milestones.append(round(step * stage_max_epoch))
            lr_scheduler.append(WarmupMultiStepLR(optimizer,
                                                  milestones,
                                                  cfg.SOLVER.GAMMA,
                                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                                                  warmup_method=cfg.SOLVER.WARMUP_METHOD, )
                                )
        return lr_scheduler

    elif cfg.SOLVER.USE_COSINE:
        max_iters = cfg.SOLVER.MAX_ITER
        return WarmupCosineAnnealingLR(
            optimizer,
            max_iters,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            eta_min=cfg.SOLVER.MIN_LR
        )

    # elif cfg.SOLVER.USE_AUTOSTEP:
    #     max_iters = cfg.SOLVER.MAX_ITER
    #     return WarmupReduceLROnPlateau(
    #         optimizer,
    #         max_iters,
    #         cfg.SOLVER.GAMMA,
    #         warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
    #         warmup_iters=cfg.SOLVER.WARMUP_ITERS,
    #         warmup_method=cfg.SOLVER.WARMUP_METHOD,
    #         eta_min=cfg.SOLVER.MIN_LR,
    #         patience=cfg.SOLVER.STEP_PATIENCE,
    #         verbose=True
    #     )

    else:
        milestones = []
        for step in cfg.SOLVER.STEPS:
            if step < 1:
                milestones.append(round(step * cfg.SOLVER.MAX_ITER))
            else:
                milestones.append(step)
        return WarmupMultiStepLR(
            optimizer,
            milestones,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
