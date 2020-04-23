# Copyright (c) 2020       Preferred Networks, Inc.
# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# NVIDIA implementation of DALI dataloders
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
#

import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


def dali_collate_fn(batch):
    jpegs, labels = tuple(zip(*batch))
    return list(jpegs), list(map(lambda x: np.array([x]), labels))


def get_mean_std(options):
    if options.tiny_imagenet:
        # computed from 100000 train images of Tiny ImageNet
        return [122.460, 114.257, 101.363], [70.491, 68.560, 71.805]
    else:
        return [0.485 * 255,0.456 * 255,0.406 * 255], [0.229 * 255,0.224 * 255,0.225 * 255]


class TrainPipeline(Pipeline):
    def __init__(self, device_id, crop, size, rank, seed_rank, options):
        super(TrainPipeline, self).__init__(options.batchsize,
                                            4,
                                            device_id,
                                            prefetch_queue_depth=3,
                                            set_affinity=True,
                                            seed=options.seed + seed_rank)
        self.input = ops.FileReader(file_root=options.train_dir,
                                    shard_id=rank,
                                    num_shards=size,
                                    random_shuffle=True)

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        random_aspect_ratio = [0.75, 4./3.]
        random_area = [0.08, 1.0]
        self.decode = ops.ImageDecoderRandomCrop(device="mixed",
                                                 output_type=types.RGB,
                                                 device_memory_padding=211025920,
                                                 host_memory_padding=140544512,
                                                 random_aspect_ratio=random_aspect_ratio,
                                                 random_area=random_area,
                                                 num_attempts=100,
                                                 seed=options.seed + seed_rank + 1641)

        self.res = ops.Resize(device="gpu", resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        dtype = types.FLOAT16 if options.fp16 else types.FLOAT
        layout = types.NCHW
        padding = False
        img_mean, img_std = get_mean_std(options)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype = dtype,
                                            output_layout = layout,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = img_mean,
                                            std = img_std,
                                            pad_output=padding,
                                            seed=options.seed + seed_rank + 1223)
        self.coin = ops.CoinFlip(probability = 0.5, seed=options.seed + seed_rank + 412)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class ValPipeline(Pipeline):
    def __init__(self, device_id,
                 shard_id, num_shards, size, crop, options):
        super(ValPipeline, self).__init__(options.batchsize,
                                          options.dataloader_workers,
                                          device_id,
                                          seed=12 + device_id)
        self.input = ops.FileReader(file_root=options.val_dir,
                                    shard_id=shard_id,
                                    num_shards=num_shards,
                                    random_shuffle=False,
                                    pad_last_batch=True)

        self.decode = ops.ImageDecoder(device="mixed", output_type = types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter = size, interp_type=types.INTERP_TRIANGULAR)
        dtype = types.FLOAT16 if options.fp16 else types.FLOAT
        layout = types.NCHW
        img_mean, img_std = get_mean_std(options)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                output_dtype = dtype,
                output_layout = layout,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = img_mean,
                std = img_std,
                pad_output=False)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
