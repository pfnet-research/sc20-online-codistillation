# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import logging
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import chainerio as chio


logger = logging.getLogger(__name__)


def create_pretraining_dataset(container_file, input_file, max_pred_length,
                               args):
    with chio.open_as_container(container_file) as container:
        with container.open(input_file, "rb") as file:
            train_data = pretraining_dataset(
                input_file, file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler,
        batch_size=args.train_batch_size, num_workers=0,
        pin_memory=True)

    return train_dataloader, input_file


class pretraining_dataset(Dataset):

    def __init__(self, input_file_name, input_file, max_pred_length):
        self.input_file_name = input_file_name
        self.max_pred_length = max_pred_length
        import io
        data = input_file.read()
        bytesio = io.BytesIO(data)
        with h5py.File(bytesio, "r") as f:
            keys = ['input_ids', 'input_mask', 'segment_ids',
                    'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
            self.inputs = [np.asarray(f[key][:]) for key in keys]
        bytesio.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions,
         masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5
            else torch.from_numpy(
                np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]
