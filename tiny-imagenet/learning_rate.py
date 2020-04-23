# Copyright 2020 Preferred Networks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math

from options import LearningRateOption


class LearningRateScheduler(object):
    """Learning rate scheduler.
    This serves as a proxy for schedule function `func`.

    Args:
        func: a schedule function. It should take 4 arguments optimizer, epoch, epoch_detail, and iteration.
    """

    def __init__(self, func, wd_rate=None):
        self.func = func
        self.wd_rate = wd_rate

    def __call__(self, optimizer, epoch, epoch_detail, iteration):
        val = self.func(epoch, epoch_detail, iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = val
            if self.wd_rate is not None and param_group['weight_decay'] > 0:
                param_group['weight_decay'] = val * self.wd_rate


def fixed_schedule_func(initial_lr, lr_rate_by_epoch):
    def f(lrs, epoch, epoch_detail, iteration):
        if epoch < 0:
            return lrs[0]
        elif epoch >= len(lrs):
            return lrs[-1]
        else:
            return lrs[epoch]

    lrs = list(map(lambda v: initial_lr * v, lr_rate_by_epoch))

    return partial(f, lrs)


def constant_schedule_func(lr):
    return fixed_schedule_func(lr, [1.0])


def real_cosine_schedule_func(initial_lr, n_total_epochs):
    def f(initial_lr, n_total_epochs, epoch, epoch_detail, iteration):
        return initial_lr * (1 + math.cos(epoch_detail / n_total_epochs * math.pi)) / 2
    return partial(f, initial_lr, n_total_epochs)


def exponential_schedule_func(initial_lr, last_lr, n_total_epochs):
    def f(initial_lr, last_lr, n_total_epochs, epoch, epoch_detail, iteration):
        return initial_lr * math.pow(last_lr / initial_lr, epoch_detail / n_total_epochs)
    return partial(f, initial_lr, last_lr, n_total_epochs)


def polynomial_schedule_func(initial_lr, power, n_total_epochs):
    def f(initial_lr, power, n_total_epochs, epoch, epoch_detail, iteration):
        return initial_lr * math.pow(1.0 - epoch_detail / n_total_epochs, power)
    return partial(f, initial_lr, power, n_total_epochs)


def with_warmup(func, warmup_epochs, warmup_start):
    def f(func, warmup_epochs, epoch, epoch_detail, iteration):
        if epoch_detail < warmup_epochs:
            return max(0.0, func(epoch, epoch_detail, iteration) - warmup_start) * epoch_detail / warmup_epochs + warmup_start
        else:
            return func(epoch, epoch_detail, iteration)

    return partial(f, func, warmup_epochs)


def with_lower_bound(func, threshold):
    def f(func, threshold, epoch, epoch_detail, iteration):
        base = func(epoch, epoch_detail, iteration)
        return max(base, threshold)

    return partial(f, func ,threshold)


def get_learning_rate_scheduler(options, epoch, effective_batchsize, initial_wd=None):
    # type: (LearningRateOption, int, int) -> LearningRateScheduler
    initial_lr = options.initial_lr * (effective_batchsize / 256.0)

    if options.cosine:
        schedule_func = real_cosine_schedule_func(initial_lr, epoch)
    elif options.exponential_rate is not None:
        schedule_func = exponential_schedule_func(initial_lr, initial_lr * options.exponential_rate, epoch)
    elif options.polynomial_power is not None:
        schedule_func = polynomial_schedule_func(initial_lr, options.polynomial_power, epoch)
    else:
        # decay
        lrs = []
        last_ratio = 1.0
        for i in range(len(options.decay_epoch)):
            lrs += [last_ratio] * options.decay_epoch[i]
            last_ratio *= options.decay_ratio[i]
        if len(lrs) < epoch:
            lrs += [last_ratio] * (epoch - len(lrs))

        schedule_func = fixed_schedule_func(initial_lr, lrs)

    if options.warmup_epoch is not None:
        schedule_func = with_warmup(schedule_func, options.warmup_epoch, options.initial_lr)
    if options.end_ratio is not None:
        schedule_func = with_lower_bound(schedule_func, options.end_ratio * initial_lr)

    if initial_wd is not None:
        wd_rate = initial_wd / options.initial_lr
    else:
        wd_rate = None
    return LearningRateScheduler(schedule_func, wd_rate)
