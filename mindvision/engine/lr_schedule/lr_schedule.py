# Copyright 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Learning rate schedule generator. TODO: replace by mindspore.nn.lr.xxx """

import ast
import math
from collections import Counter
import numpy as np
from mindspore import Tensor
from mindvision.check_param import Validator


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr, milestones, steps_per_epoch, max_epoch, gamma=0.1):
    return warmup_step_lr(lr, milestones, steps_per_epoch, 0, max_epoch, gamma=gamma)


def step_lr(lr, epoch_size, steps_per_epoch, max_epoch, gamma=0.1):
    lr_epochs = []
    for i in range(1, max_epoch):
        if i % epoch_size == 0:
            lr_epochs.append(i)
    return multi_step_lr(lr, lr_epochs, steps_per_epoch, max_epoch, gamma=gamma)


def warmup_cosine_annealing_lr_v1(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_v2(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    """Cosine annealing learning rate V2."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    last_lr = 0
    last_epoch_v1 = 0

    t_max_v2 = int(max_epoch / 3)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            if i < total_steps * (2 / 3):
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
                last_lr = lr
                last_epoch_v1 = last_epoch
            else:
                base_lr = last_lr
                last_epoch = last_epoch - last_epoch_v1
                lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max_v2)) / 2

        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_sample(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max=None, eta_min=0):
    """Warmup cosine annealing learning rate."""
    start_sample_epoch = 60
    step_sample = 2
    tobe_sampled_epoch = 60
    end_sampled_epoch = start_sample_epoch + step_sample * tobe_sampled_epoch
    max_sampled_epoch = max_epoch + tobe_sampled_epoch
    if t_max is None:
        t_max = max_sampled_epoch

    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    total_sampled_steps = int(max_sampled_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []

    for i in range(total_sampled_steps):
        last_epoch = i // steps_per_epoch
        if last_epoch in range(start_sample_epoch, end_sampled_epoch, step_sample):
            continue
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    assert total_steps == len(lr_each_step)
    return np.array(lr_each_step).astype(np.float32)


def get_lr_ssd(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def get_lr(args):
    """generate learning rate."""
    Validator.check_string(args.lr_scheduler, ['exponential',
                                               'cosine_annealing',
                                               'cosine_annealing_V2',
                                               'cosine_annealing_sample',
                                               'dynamic_lr',
                                               'multi_warmup_epochs_lr',
                                               'multistep',
                                               'lr_ssd'])
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr_v1(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_V2':
        lr = warmup_cosine_annealing_lr_v2(args.lr,
                                           args.steps_per_epoch,
                                           args.warmup_epochs,
                                           args.max_epoch,
                                           args.t_max,
                                           args.eta_min)
    elif args.lr_scheduler == 'cosine_annealing_sample':
        lr = warmup_cosine_annealing_lr_sample(args.lr,
                                               args.steps_per_epoch,
                                               args.warmup_epochs,
                                               args.max_epoch,
                                               args.t_max,
                                               args.eta_min)
    elif args.lr_scheduler == 'dynamic_lr':
        lr = dynamic_lr(args.lr,
                        args.steps_per_epoch,
                        args.warmup_steps,
                        args.warmup_ratio,
                        args.max_epoch)
    elif args.lr_scheduler == 'multi_warmup_epochs_lr':
        args.lr_end_rate = ast.literal_eval(args.lr_end_rate)
        args.lr_init = ast.literal_eval(args.lr_init)

        lr = multi_warmup_epochs_lr(args.lr_init,
                                    args.lr_end_rate * args.lr,
                                    args.lr,
                                    args.warmup_epochs,
                                    args.max_epoch,
                                    args.steps_per_epoch)
    elif args.lr_scheduler == 'multistep':
        lr = multi_step_lr(args.lr,
                           args.lr_epochs,
                           args.steps_per_epoch,
                           args.max_epoch,
                           args.lr_gamma)
    elif args.lr_scheduler == 'lr_ssd':
        lr = Tensor(get_lr_ssd(args.pre_trained_epoch_size * args.dataset_size,
                               args.lr_init,
                               args.lr_end_rate * args.lr,
                               args.lr,
                               args.warmup_epochs,
                               args.epoch_size,
                               args.dataset_size))
    else:
        raise NotImplementedError(args.lr_scheduler)
    return lr


def dynamic_lr(base_lr, steps_per_epoch, warmup_steps, warmup_ratio, epoch_size):
    """dynamic learning rate generator"""
    total_steps = steps_per_epoch * (epoch_size + 1)
    warmup_steps = int(warmup_steps)
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * warmup_ratio))
        else:
            lr.append(cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr


def multi_warmup_epochs_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, global_step=0):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps1 = steps_per_epoch * warmup_epochs[0]
    warmup_steps2 = warmup_steps1 + steps_per_epoch * warmup_epochs[1]
    warmup_steps3 = warmup_steps2 + steps_per_epoch * warmup_epochs[2]
    warmup_steps4 = warmup_steps3 + steps_per_epoch * warmup_epochs[3]
    warmup_steps5 = warmup_steps4 + steps_per_epoch * warmup_epochs[4]
    for i in range(total_steps):
        if i < warmup_steps1:
            lr = lr_init * (warmup_steps1 - i) / (warmup_steps1) + \
                 (lr_max * 1e-4) * i / (warmup_steps1 * 3)
        elif warmup_steps1 <= i < warmup_steps2:
            lr = 1e-5 * (warmup_steps2 - i) / (warmup_steps2 - warmup_steps1) + \
                 (lr_max * 1e-3) * (i - warmup_steps1) / (warmup_steps2 - warmup_steps1)
        elif warmup_steps2 <= i < warmup_steps3:
            lr = 1e-4 * (warmup_steps3 - i) / (warmup_steps3 - warmup_steps2) + \
                 (lr_max * 1e-2) * (i - warmup_steps2) / (warmup_steps3 - warmup_steps2)
        elif warmup_steps3 <= i < warmup_steps4:
            lr = 1e-3 * (warmup_steps4 - i) / (warmup_steps4 - warmup_steps3) + \
                 (lr_max * 1e-1) * (i - warmup_steps3) / (warmup_steps4 - warmup_steps3)
        elif warmup_steps4 <= i < warmup_steps5:
            lr = 1e-2 * (warmup_steps5 - i) / (warmup_steps5 - warmup_steps4) + \
                 lr_max * (i - warmup_steps4) / (warmup_steps5 - warmup_steps4)
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps5) / (total_steps - warmup_steps5))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
