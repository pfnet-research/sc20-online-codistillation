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

import argparse
import sys
from distutils.util import strtobool
import json
from functools import partial


class Options(object):
    def __init__(self):
        self.out = None

        # dataset
        self.train_dir = None
        self.val_dir = None
        self.dataloader_workers = 4
        self.tiny_imagenet = False

        # training
        self.model = 'resnet50'
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epoch = 90
        self.batchsize = 32
        self.lr = LearningRateOption()
        self.seed = 0
        self.label_smoothing = None
        self.lars_eta = None

        self.fp16 = False
        self.loss_scaling = None

        # online distillation
        self.distillation = False
        self.distillation_overlap = False
        self.equalize_data = False
        self.distillation_loss = 'kl_divergence'
        self.burnin_epoch = 0
        self.comm_model_freq = 50
        self.distillation_weight = 1.0

        # debug option
        self.start_nvprof_iter = None
        self.stop_nvprof_iter = None

        self.no_output_model = False

        # torch.distributed
        self.local_rank = None


class LearningRateOption(object):

    def __init__(self):
        self.initial_lr = 0.01
        self.warmup_epoch = None
        self.cosine = False
        self.exponential_rate = None
        self.polynomial_power = None
        self.decay_epoch = []
        self.decay_ratio = []
        self.end_ratio = None


def _set_unless_none(options, args, key, type=None):

    val = getattr(args, key)
    if val is not None:
        if type == 'bool':
            setattr(options, key, strtobool(val))
        elif type == 'json':
            subopt = getattr(options, key)
            recipe = json.loads(val)
            apply_recipe(subopt, recipe)
        else:
            setattr(options, key, val)


def apply_recipe(options, recipe):
    if not isinstance(recipe, dict):
        raise TypeError('recipe must be a dict')

    for key, val in recipe.items():
        if not hasattr(options, key):
            raise ValueError('unknown option {}'.format(key))
        if isinstance(getattr(options, key), (Options, LearningRateOption)):
            apply_recipe(getattr(options, key), val)
        else:
            setattr(options, key, val)


def parse_command_line(args_in=None):
    if args_in is None:
        args_in = sys.argv[1:]

    parser = argparse.ArgumentParser('online-distillation-tinyimagenet')

    parser.add_argument('--out', type=str)

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--dataloader_workers', type=int)
    parser.add_argument('--tiny_imagenet', type=str)  # bool-like

    parser.add_argument('--model', type=str)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--label_smoothing', type=float)
    parser.add_argument('--lars_eta', type=float)

    parser.add_argument('--fp16', type=str)  # bool-like
    parser.add_argument('--loss_scaling', type=int)

    parser.add_argument('--distillation', type=str)  # bool-like
    parser.add_argument('--distillation_overlap', type=str)  # bool-like
    parser.add_argument('--equalize_data', type=str)  # bool-like
    parser.add_argument('--distillation_loss', type=str)
    parser.add_argument('--burnin_epoch', type=int)
    parser.add_argument('--comm_model_freq', type=int)
    parser.add_argument('--distillation_weight', type=float)

    parser.add_argument('--start_nvprof_iter', type=int)
    parser.add_argument('--stop_nvprof_iter', type=int)

    parser.add_argument('--no_output_model', type=str)  # bool-like
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args(args_in)

    options = Options()

    set_unless_none = partial(_set_unless_none, options, args)

    set_unless_none('out')

    set_unless_none('train_dir')
    set_unless_none('val_dir')
    set_unless_none('dataloader_workers')
    set_unless_none('tiny_imagenet', type='bool')

    set_unless_none('model')
    set_unless_none('momentum')
    set_unless_none('weight_decay')
    set_unless_none('epoch')
    set_unless_none('batchsize')
    set_unless_none('lr', type='json')
    set_unless_none('seed')
    set_unless_none('label_smoothing')
    set_unless_none('lars_eta')

    set_unless_none('fp16', type='bool')
    set_unless_none('loss_scaling')

    set_unless_none('distillation', type='bool')
    set_unless_none('distillation_overlap', type='bool')
    set_unless_none('equalize_data', type='bool')
    set_unless_none('distillation_loss')
    set_unless_none('burnin_epoch')
    set_unless_none('comm_model_freq')
    set_unless_none('distillation_weight')

    set_unless_none('start_nvprof_iter')
    set_unless_none('stop_nvprof_iter')

    set_unless_none('no_output_model', type='bool')

    set_unless_none('local_rank')

    return options
