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

import torch
from torch.utils.data.distributed import DistributedSampler
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

import dali_dataloader


class AutoDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super(AutoDistributedSampler, self).__init__(*args, **kwargs)

    def __iter__(self):
        ret = super(AutoDistributedSampler, self).__iter__()
        self.epoch += 1
        return ret


class DALIDataLoaderWrapper(object):
    def __init__(self, base):
        self.base = base

    def __iter__(self):
        return DALIIteratorWrapper(iter(self.base))

    def reset(self):
        self.base.reset()

class DALIIteratorWrapper(object):
    def __init__(self, base):
        self.base = base

    def __next__(self):
        data = next(self.base)
        images = data[0]['data'].cuda()
        target = data[0]['label'].cuda().view(-1)
        if target.dtype != torch.long:
            target = target.long()
        return (images, target)


def get_dali_train_loader(options, size, rank, team, gpu, seed_rank):
    img_size = 64 if options.tiny_imagenet else 224
    train_pipe = dali_dataloader.TrainPipeline(gpu, img_size, size, rank, seed_rank, options)
    train_pipe.build()
    global_epoch_size = train_pipe.epoch_size('Reader')
    epoch_size = global_epoch_size // (size * options.batchsize) * options.batchsize
    dali_iter = DALIClassificationIterator(train_pipe,
                                           size=epoch_size,
                                           auto_reset=True,
                                           last_batch_padded=False,
                                           fill_last_batch=True)
    return DALIDataLoaderWrapper(dali_iter), global_epoch_size


def get_dali_val_loader(options, size, rank, gpu):
    def start_index(pipe, id):
        epoch_size = pipe.epoch_size("Reader")
        remainder = epoch_size % size
        if id < remainder:
            return epoch_size // size * id + id
        else:
            return epoch_size // size * id + remainder

    if options.tiny_imagenet:
        img_size1, img_size2 = 64, 64
    else:
        img_size1, img_size2 = 256, 224
    val_pipe = dali_dataloader.ValPipeline(gpu, rank, size, img_size1, img_size2, options)
    val_pipe.build()
    global_epoch_size = val_pipe.epoch_size("Reader")
    if rank == size - 1:
        epoch_size = global_epoch_size - start_index(val_pipe, rank)
    else:
        epoch_size = start_index(val_pipe, rank + 1) - start_index(val_pipe, rank)
    dali_iter = DALIClassificationIterator(val_pipe,
                                           dynamic_shape=True,
                                           size=epoch_size,
                                           last_batch_padded=False,
                                           fill_last_batch=False)
    return DALIDataLoaderWrapper(dali_iter), global_epoch_size
