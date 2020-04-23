# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright 2020 Preferred Networks, Inc.

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

import os
import torch
from apex import amp
import chainerio as chio

from utils import is_main_process


class Snapshot():
    def __init__(self, args, model, another_model, optimizer, team):
        self.args = args
        self.model = model
        self.another_model = another_model
        self.optimizer = optimizer
        self.most_recent_ckpts_paths = []
        self.team = team

        self.maybe_load()

    def maybe_load(self):
        self.global_step = None
        self.f_id = None
        self.files = None
        checkpoint = None

        if chio.exists(self.args.output_dir):
            model_names = [f for f in chio.list(
                self.args.output_dir)
                if f.endswith(".pt.{}".format(self.team))]
            if len(model_names) != 0:
                self.args.resume_step = max(
                    [int(x.split(
                        '.pt.{}'.format(self.team))[0].split('_')[1].strip())
                     for x in model_names])
                self.global_step = self.args.resume_step

        if self.global_step is not None:
            print("Load from {}".format(os.path.join(self.args.output_dir,
                                                     "ckpt_{}.pt.{}".format(
                                                         self.global_step,
                                                         self.team))))
            with chio.open(os.path.join(self.args.output_dir,
                                        "ckpt_{}.pt.{}".format(
                                            self.global_step, self.team)),
                           "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            self.model.load_state_dict(checkpoint['model'],
                                       strict=False)
            self.another_model.load_state_dict(
                checkpoint['another_model'], strict=False)
            if self.args.phase2:
                self.global_step -= self.args.phase1_end_step
            if is_main_process():
                print("resume step from ", self.args.resume_step)

            if self.args.phase2:
                keys = list(checkpoint['optimizer']['state'].keys())
                # Override hyperparameters from Phase 1
                for key in keys:
                    checkpoint['optimizer']['state'][key]['step'] = \
                        self.global_step
                for iter, item in enumerate(
                        checkpoint['optimizer']['param_groups']):
                    checkpoint['optimizer']['param_groups'][iter]['t_total'] =\
                        self.args.max_steps
                    checkpoint['optimizer']['param_groups'][iter]['warmup'] = \
                        self.args.warmup_proportion
                    checkpoint['optimizer']['param_groups'][iter]['lr'] = \
                        self.args.learning_rate
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # Restore AMP master parameters
            self.f_id = checkpoint['files'][0]
            self.files = checkpoint['files'][1:]

    def save(self, step, f_id, files):
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') \
            else self.model
        another_model_to_save = self.another_model.module \
            if hasattr(self.another_model, 'module') \
            else self.another_model
        if self.args.resume_step < 0 or not self.args.phase2:
            output_save_file = os.path.join(
                self.args.output_dir,
                "ckpt_{}.pt.{}".format(step, self.team))
        else:
            output_save_file = os.path.join(
                self.args.output_dir, "ckpt_{}.pt.{}".format(
                    step + self.args.phase1_end_step, self.team)
            )
        with chio.open(output_save_file, "wb") as f:
            torch.save({
                'model': model_to_save.state_dict(),
                'another_model': another_model_to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'master params': list(amp.master_params(self.optimizer)),
                'files': [f_id] + files
            }, f)

        self.most_recent_ckpts_paths.append(output_save_file)
        if len(self.most_recent_ckpts_paths) > 3:
            ckpt_to_be_removed = self.most_recent_ckpts_paths.pop(0)
            chio.remove("{}".format(ckpt_to_be_removed))
