# -*- encoding: utf-8 -*-
'''
Filename         :misc.py
Description      :
Time             :2023/07/26 21:41:45
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class KLAnnealer:
    def __init__(self,
                 epochs: int,
                 kl_start: int,
                 kl_w_start: float,
                 kl_w_end: float
                 ) -> None:
        self.i_start = kl_start  # epoch起点
        self.w_start = kl_w_start  # 权重起始值
        self.w_max = kl_w_end  # 最大权重值
        self.epochs = epochs

        self.inc = (self.w_max - self.w_start) / (self.epochs - self.i_start)

    def __call__(self,
                 i: int
                 ) -> float:
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self,
                 optimizer:Optimizer,
                 lr_n_period:int,
                 lr_n_mult:int,
                 lr_end:float) -> None:
        self.n_period = lr_n_period #
        self.n_mult = lr_n_mult #
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self) -> List:#
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self,
             epoch=None
             )->None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end
