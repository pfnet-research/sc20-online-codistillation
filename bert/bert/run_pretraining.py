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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import time
import logging
import argparse
import random
import numpy as np
import torch
from apex import amp
import chainerio as chio

from modeling import BertForPreTraining, BertConfig
from optimization import BertLAMB, BertAdam

from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C

from concurrent.futures import ThreadPoolExecutor

from dataset import create_pretraining_dataset
from snapshot import Snapshot


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def is_main_process(self):
        return self.team_rank == 0

    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument("--input_file",
                            default=None,
                            type=str,
                            required=True,
                            help="The input data file. Should be zip file "
                            "containing .hdf5 files for the task.")

        parser.add_argument("--config_file",
                            default=None,
                            type=str,
                            required=True,
                            help="The BERT model config")

        parser.add_argument("--bert_model", default="bert-large-uncased",
                            type=str,
                            help="Bert pre-trained model selected in the "
                            "list: bert-base-uncased, bert-large-uncased, "
                            "bert-base-cased, bert-base-multilingual, "
                            "bert-base-chinese.")

        parser.add_argument("--output_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The output directory where the model "
                            "checkpoints will be written.")

        # Other parameters
        parser.add_argument("--max_seq_length",
                            default=512,
                            type=int,
                            help="The maximum total input sequence length "
                            "after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, "
                            "and sequences shorter \n"
                            "than this will be padded.")
        parser.add_argument("--max_predictions_per_seq",
                            default=80,
                            type=int,
                            help="The maximum total of masked tokens in input "
                            "sequence")
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--max_steps",
                            default=1000,
                            type=float,
                            help="Total number of training steps to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.01,
                            type=float,
                            help="Proportion of training to perform linear "
                            "learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--log_freq',
                            type=float, default=50.0,
                            help='frequency of logging loss.')
        parser.add_argument('--checkpoint_activations',
                            default=False,
                            action='store_true',
                            help="Whether to use gradient checkpointing")
        parser.add_argument("--resume_from_checkpoint",
                            default=False,
                            action='store_true',
                            help="Whether to resume training from checkpoint.")
        parser.add_argument('--resume_step',
                            type=int,
                            default=-1,
                            help="Step to resume training from.")
        parser.add_argument('--num_steps_per_checkpoint',
                            type=int,
                            default=100,
                            help="Number of update steps until a model "
                            "checkpoint is saved to disk.")
        parser.add_argument('--phase2',
                            default=False,
                            action='store_true',
                            help="Whether to train with seq len 512")
        parser.add_argument('--phase1_end_step',
                            type=int,
                            default=7038,
                            help="Number of training steps in Phase1 - "
                            "seq len 128")
        parser.add_argument('--online_distillation',
                            type=str,
                            default="none",
                            choices=["none", "original", "overlap", "logit"],
                            help="Settings for online distillation")
        parser.add_argument('--burnin_steps',
                            type=int,
                            default=0)
        parser.add_argument('--distillation_weight',
                            type=float,
                            default=1)
        parser.add_argument('--distillation_loss',
                            type=str,
                            default="kl_divergence",
                            choices=["cross_entropy", "kl_divergence"])
        parser.add_argument('--distillation_steps',
                            type=int, default=50)
        parser.add_argument('--optimizer',
                            type=str,
                            default="lamb",
                            choices=["lamb", "adam"])
        self.args = parser.parse_args()

    def setup_training(self):
        assert (torch.cuda.is_available())

        torch.cuda.set_device(self.args.local_rank)
        self.device = torch.device("cuda", self.args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

        self.rank = torch.distributed.get_rank()
        self.size = torch.distributed.get_world_size()
        if self.args.online_distillation == "none":
            self.team = 0
            self.team_masters = [0]
            self.team_master = 0
            self.local_group = torch.distributed.new_group(
                ranks=list(range(0, self.size)))
            self.team_rank = torch.distributed.get_rank()
            self.team_size = torch.distributed.get_world_size()
        else:
            assert self.size % 2 == 0, \
                'with distillation, world size must be a multiple of 2'
            self.team = self.rank // (self.size // 2)
            self.team_masters = [0, (self.size // 2)]
            self.team_master = self.team_masters[self.team]
            self.is_team_master = (self.rank % (self.size // 2) == 0)
            local_group0 = torch.distributed.new_group(
                ranks=list(range(0, self.size // 2)))
            local_group1 = torch.distributed.new_group(
                ranks=list(range(self.size // 2, self.size)))
            self.local_groups = [local_group0, local_group1]
            self.local_group = self.local_groups[self.team]

            self.team_rank = self.rank % (self.size // 2)
            self.team_size = self.size // 2

            comm_model_group_rank0 = \
                [0] + list(range(self.team_size, self.team_size * 2))
            comm_model_group_rank1 = \
                [self.team_size] + list(range(0, self.team_size))
            self.comm_model_group_ranks = [comm_model_group_rank0,
                                           comm_model_group_rank1]

            if self.args.online_distillation == "logit":
                for i in range(0, self.size // 2):
                    ranks = [i, i + self.size // 2]
                    grp = torch.distributed.new_group(ranks=ranks)
                    if self.rank in ranks:
                        self.equalize_data_group = grp
                # use different seeds in different teams
                self.args.data_seed = 12345
                self.args.seed += self.team * 12345
            else:
                # use different seeds in different teams
                self.args.seed += self.team * 12345

        self.args.train_batch_size //= self.team_size

        if not self.args.resume_from_checkpoint:
            chio.makedirs(self.args.output_dir, exist_ok=True)

    def prepare_model_and_optimizer(self):
        # Prepare model
        self.config = BertConfig.from_json_file(self.args.config_file)

        # Padding for divisibility by 8
        if self.config.vocab_size % 8 != 0:
            self.config.vocab_size += 8 - (self.config.vocab_size % 8)
        self.model = BertForPreTraining(self.config)
        self.another_model = BertForPreTraining(self.config)

        self.model.to(self.device)
        self.another_model.to(self.device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

        optimizer_grouped_parameters = []
        names = []

        for n, p in param_optimizer:
            if not any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append(
                    {'params': [p], 'weight_decay': 0.01, 'name': n})
                names.append({'params': [n], 'weight_decay': 0.01})
            if any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append(
                    {'params': [p], 'weight_decay': 0.00, 'name': n})
                names.append({'params': [n], 'weight_decay': 0.00})

        if self.args.phase2:
            max_steps = self.args.max_steps
            tmp = max_steps * 10
            r = self.args.phase1_end_step / tmp
            lr = self.args.learning_rate * (1 - r)
        else:
            max_steps = int(self.args.max_steps / 9 * 10)
            lr = self.args.learning_rate
        if self.args.optimizer == "lamb":
            self.optimizer = BertLAMB(
                optimizer_grouped_parameters,
                lr=lr,
                warmup=self.args.warmup_proportion
                if not self.args.phase2 else -1,
                t_total=max_steps)
        elif self.args.optimizer == "adam":
            self.optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=lr,
                warmup=self.args.warmup_proportion
                if not self.args.phase2 else -1,
                t_total=max_steps)

    def prepare_snapshot(self):
        self.snapshot = Snapshot(self.args, self.model, self.another_model,
                                 self.optimizer, self.team)
        flat_dist_call([param.data for param in self.model.parameters()],
                       torch.distributed.broadcast, (self.team_master,
                                                     self.local_group))

    def forward(self, model, batch, calc_loss=True):
        input_ids, segment_ids, input_mask, \
            masked_lm_labels, next_sentence_labels = batch
        if calc_loss:
            return model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                masked_lm_labels=masked_lm_labels,
                next_sentence_label=next_sentence_labels,
                checkpoint_activations=self.args.checkpoint_activations)
        else:
            return model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                masked_lm_labels=None,
                next_sentence_label=None,
                checkpoint_activations=self.args.checkpoint_activations)

    def backward(self, loss):
        loss.backward()

    def comm_model(self):
        for i in range(2):
            root = self.comm_model_group_ranks[i][0]
            teams = set(range(root, root + self.team_size))
            if self.rank in teams:
                flat_dist_call(
                    [param.data for param in self.model.parameters()],
                    torch.distributed.broadcast,
                    (i * self.team_size,))
            else:
                flat_dist_call(
                    [param.data for param in self.another_model.parameters()],
                    torch.distributed.broadcast,
                    (i * self.team_size,))

    def all_reduce(self, overflow_buf, accum=1):
        scaler = amp.scaler.LossScaler(1.0)

        # 1. allocate an uninitialized buffer for flattened gradient
        master_grads = [p.grad for p in amp.master_params(
            self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float32
        flat_raw = torch.empty(
            flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            overflow_buf, [master_grads, allreduced_views],
            scaler.loss_scale() / (self.team_size * accum))
        # 3. sum gradient across ranks. Because of the predivision,
        #    this averages the gradient
        torch.distributed.all_reduce(flat_raw, group=self.local_group)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
                                 overflow_buf,
                                 [allreduced_views, master_grads],
                                 1./scaler.loss_scale())

    def take_optimizer_step(self, global_step):
        # 1. call optimizer step function
        self.optimizer.step()
        global_step += 1
        for param in self.model.parameters():
            param.grad = None

        return global_step

    def init_dataloader(self, epoch, pool, rng=None):
        rng = rng or random
        if not self.args.resume_from_checkpoint or epoch > 0 or \
                self.args.phase2:
            with chio.open_as_container(self.args.input_file) as input_file:
                files = [f for f in input_file.list() if "training" in f]
            files.sort()
            num_files = len(files)
            rng.shuffle(files)
            f_start_id = 0
        else:
            f_start_id = self.snapshot.f_id
            files = self.snapshot.files
            self.args.resume_from_checkpoint = False
            num_files = len(files)

        if torch.distributed.is_initialized() and \
                self.team_size > num_files:
            remainder = self.team_size % num_files
            data_file = files[
                (f_start_id*self.team_size +
                 self.team_rank + remainder*f_start_id
                 ) % num_files]
        else:
            data_file = files[
                (f_start_id*self.team_size + self.team_rank) % len(files)]

        return pool.submit(
            create_pretraining_dataset, self.args.input_file, data_file,
            self.args.max_predictions_per_seq,
            self.args), f_start_id, files, data_file

    def update_dataloader(self, pool, f_id, files):
        if self.team_size > len(files):
            remainder = self.team_size % len(files)
            data_file = files[
                (f_id*self.team_size +
                    self.team_rank + remainder*f_id
                 ) % len(files)]
        else:
            data_file = files[
                (f_id*self.team_size + self.team_rank) % len(files)]

        dataset_future = pool.submit(
            create_pretraining_dataset, self.args.input_file, data_file,
            self.args.max_predictions_per_seq, self.args)
        return dataset_future, data_file

    def loss(self, prediction_scores, seq_relationship_score,
             batch):
        _, _, _, masked_lm_labels, next_sentence_labels = batch
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            masked_lm_labels.view(-1))
        next_sentence_loss = loss_fct(
            seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        return masked_lm_loss + next_sentence_loss

    def compute_distillation_loss(self, output, another_output, target=None):
        c = output.shape[-1]
        output = output.view(-1, c)
        another_output = another_output.view(-1, c)
        with torch.no_grad():
            if target is None:
                mask = torch.ones(len(output), 1, device=output.device,
                                  dtype=output.dtype)
            else:
                mask = (target != -1).long().view(-1, 1)
        if self.args.distillation_loss == 'cross_entropy':
            other_distr = torch.softmax(another_output, dim=1)
            return -torch.sum(mask *
                              (torch.log_softmax(output, dim=1) * other_distr)
                              ) / sum(mask)
        elif self.args.distillation_loss == 'kl_divergence':
            return torch.sum(mask * (torch.softmax(output, dim=1) *
                                     (torch.log_softmax(output, dim=1) -
                                      torch.log_softmax(another_output, dim=1)
                                      ))) / sum(mask)
        else:
            raise ValueError('unknown distillation loss: {}'.format(
                self.args.distillation_loss))

    def train_simple(self):
        global_step = self.snapshot.global_step or 0
        if self.args.phase2:
            self.args.accum = self.args.train_batch_size // 8
            self.args.train_batch_size = 8
        else:
            self.args.accum = 1

        if self.is_main_process():
            print("SEED {}".format(self.args.seed))
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.args.train_batch_size)
            logger.info("  Accum = %d", self.args.accum)
            print("  LR = ", self.args.learning_rate)
            print("Training. . .")

        self.model.train()
        average_loss = 0.0  # averaged loss every self.args.log_freq steps
        epoch = 0

        # Note: We loop infinitely over epochs, termination is handled via
        #       iteration count
        begin = None
        with ThreadPoolExecutor(1) as pool:
            while True:
                dataset_future, f_start_id, files, data_file = \
                    self.init_dataloader(epoch, pool)
                previous_file = data_file
                train_dataloader, _ = dataset_future.result(timeout=None)

                overflow_buf = torch.cuda.IntTensor([0])

                for f_id in range(f_start_id + 1, len(files)):
                    logger.info("file no %s file %s" % (f_id, previous_file))
                    dataset_future, data_file = \
                        self.update_dataloader(pool, f_id, files)
                    previous_file = data_file

                    it = 0
                    for batch in train_dataloader:
                        if begin is None:
                            begin = time.time()
                        it += 1
                        batch = [t.to(self.device) for t in batch]
                        loss = self.forward(self.model, batch)
                        self.backward(loss)
                        average_loss += loss.item()

                        if it % self.args.accum == 0:
                            self.all_reduce(overflow_buf, self.args.accum)
                            global_step = self.take_optimizer_step(global_step)
                            it = 0

                            if global_step % self.args.log_freq == 0:
                                divisor = self.args.log_freq * self.args.accum
                                if self.is_main_process():
                                    print(
                                        "Team: {} Step:{} Average Loss = {} "
                                        .format(
                                            self.team, global_step,
                                            average_loss / divisor))
                                average_loss = 0

                            if global_step >= self.args.max_steps or \
                                (global_step %
                                 self.args.num_steps_per_checkpoint) == 0:
                                if self.team_rank == 0:
                                    # Save a trained model
                                    logger.info("** ** Saving model ** **")
                                    self.snapshot.save(
                                        global_step, f_id, files)

                            if global_step >= self.args.max_steps:
                                del train_dataloader
                                torch.distributed.barrier()
                                if torch.distributed.get_rank() == 0:
                                    print("Total time taken {}".format(
                                        time.time() - begin))
                                return self.args

                    del train_dataloader
                    # Make sure pool has finished and switch train_dataloader
                    # NOTE: Will block until complete
                    train_dataloader, data_file = dataset_future.result(
                        timeout=None)

                epoch += 1

    def train_online_distillation_original(self):
        global_step = self.snapshot.global_step or 0

        if self.is_main_process():
            print("SEED {}".format(self.args.seed))
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.args.train_batch_size)
            print("  LR = ", self.args.learning_rate)
            print("  Online Distillation")
            print("Training. . .")

        self.model.train()
        average_loss = 0.0  # averaged loss every self.args.log_freq steps
        average_dloss_0 = 0.0  # averaged loss every self.args.log_freq steps
        average_dloss_1 = 0.0
        epoch = 0
        begin = None

        # Note: We loop infinitely over epochs, termination is handled via
        #       iteration count
        with ThreadPoolExecutor(1) as pool:
            while True:
                dataset_future, f_start_id, files, data_file = \
                    self.init_dataloader(epoch, pool)
                previous_file = data_file
                train_dataloader, _ = dataset_future.result(timeout=None)

                overflow_buf = torch.cuda.IntTensor([0])

                for f_id in range(f_start_id + 1, len(files)):
                    logger.info("file no %s file %s" % (f_id, previous_file))
                    dataset_future, data_file = \
                        self.update_dataloader(pool, f_id, files)
                    previous_file = data_file

                    for batch in train_dataloader:
                        if begin is None:
                            begin = time.time()
                        step = global_step
                        if self.args.phase2:
                            step += self.args.phase1_end_step
                        if step >= self.args.burnin_steps and \
                                (step % self.args.distillation_steps) == 0:
                            self.comm_model()

                        batch = [t.to(self.device) for t in batch]
                        _, _, _, masked_lm_labels, _ = batch
                        if step < self.args.burnin_steps:
                            loss = self.forward(self.model, batch)
                            dloss0 = torch.zeros(())
                            dloss1 = torch.zeros(())
                        else:
                            out0, out1 = self.forward(self.model, batch,
                                                      calc_loss=False)
                            with torch.no_grad():
                                aout0, aout1 = self.forward(
                                    self.another_model, batch,
                                    calc_loss=False)
                            loss = self.loss(out0, out1, batch)
                            dloss0 = \
                                self.compute_distillation_loss(
                                    out0, aout0, masked_lm_labels.view(-1))
                            dloss1 = \
                                self.compute_distillation_loss(out1, aout1)
                            dloss = dloss0 + dloss1
                            loss = loss + \
                                self.args.distillation_weight * dloss
                        self.backward(loss)
                        self.all_reduce(overflow_buf)
                        global_step = self.take_optimizer_step(global_step)
                        average_loss += loss.item()
                        average_dloss_0 += dloss0.item()
                        average_dloss_1 += dloss1.item()

                        if global_step % self.args.log_freq == 0:
                            divisor = self.args.log_freq
                            if self.is_main_process():
                                print(
                                    "Team: {} Step:{} Average Loss = {} Average dLoss = {} {}"
                                    .format(
                                        self.team, global_step,
                                        average_loss / divisor,
                                        average_dloss_0 / divisor,
                                        average_dloss_1 / divisor)
                                )
                            average_loss = 0
                            average_dloss_0 = 0
                            average_dloss_1 = 0

                        if global_step >= self.args.max_steps or \
                            (global_step %
                             self.args.num_steps_per_checkpoint) == 0:
                            if self.team_rank == 0:
                                # Save a trained model
                                logger.info("** ** Saving model ** **")
                                self.snapshot.save(global_step, f_id, files)

                            if global_step >= self.args.max_steps:
                                del train_dataloader
                                torch.distributed.barrier()
                                if torch.distributed.get_rank() == 0:
                                    print("Total time taken {}".format(
                                        time.time() - begin))
                                return self.args

                    del train_dataloader
                    # Make sure pool has finished and switch train_dataloader
                    # NOTE: Will block until complete
                    train_dataloader, data_file = dataset_future.result(
                        timeout=None)

                epoch += 1

    def train_online_distillation_overlap(self):
        global_step = self.snapshot.global_step or 0

        main_stream = torch.cuda.Stream()
        another_model_fwd_stream = torch.cuda.Stream()
        all_reduce_stream = torch.cuda.Stream()
        distillation_stream = torch.cuda.Stream()

        fwd_event = torch.cuda.Event()
        bwd_event = torch.cuda.Event()
        another_model_fwd_event = torch.cuda.Event()
        all_reduce_event = torch.cuda.Event()
        distillation_event = torch.cuda.Event()

        if self.is_main_process():
            print("SEED {}".format(self.args.seed))
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.args.train_batch_size)
            print("  LR = ", self.args.learning_rate)
            print("  Online Distillation")
            print("Training. . .")

        self.model.train()
        average_loss = 0.0  # averaged loss every self.args.log_freq steps
        average_dloss_0 = 0
        average_dloss_1 = 0
        epoch = 0
        begin = None

        # Note: We loop infinitely over epochs, termination is handled via
        #       iteration count
        batch = None
        another_output = None
        with ThreadPoolExecutor(1) as pool:
            while True:
                dataset_future, f_start_id, files, data_file = \
                    self.init_dataloader(epoch, pool)
                previous_file = data_file
                train_dataloader, _ = dataset_future.result(timeout=None)

                overflow_buf = torch.cuda.IntTensor([0])

                for f_id in range(f_start_id + 1, len(files)):
                    logger.info("file no %s file %s" % (f_id, previous_file))
                    dataset_future, data_file = \
                        self.update_dataloader(pool, f_id, files)
                    previous_file = data_file

                    for next_batch in train_dataloader:
                        next_batch = [t.to(self.device) for t in next_batch]
                        if batch is None:
                            batch = next_batch
                            continue
                        if begin is None:
                            begin = time.time()

                        step = global_step
                        if self.args.phase2:
                            step += self.args.phase1_end_step

                        _, _, _, masked_lm_labels, _ = batch
                        fwd_event.record()
                        distillation_event.record()
                        if step >= self.args.burnin_steps:
                            with torch.cuda.stream(distillation_stream):
                                distillation_event.wait()
                                if (step % self.args.distillation_steps) \
                                        == 0:
                                    self.comm_model()
                                distillation_event.record()

                        with torch.cuda.stream(main_stream):
                            fwd_event.wait()
                            if another_output is None:
                                loss = self.forward(self.model, batch)
                                dloss0 = torch.zeros(())
                                dloss1 = torch.zeros(())
                            else:
                                out0, out1 = self.forward(
                                    self.model, batch, calc_loss=False)
                                aout0, aout1 = another_output
                                loss = self.loss(out0, out1, batch)
                                dloss0 = \
                                    self.compute_distillation_loss(
                                        out0, aout0,
                                        masked_lm_labels.view(-1))
                                dloss1 = \
                                    self.compute_distillation_loss(out1,
                                                                   aout1)
                                dloss = dloss0 + dloss1

                                loss = loss + \
                                    self.args.distillation_weight * dloss
                            fwd_event.record()
                        fwd_event.wait()

                        bwd_event.record()
                        with torch.cuda.stream(main_stream):
                            bwd_event.wait()
                            self.backward(loss)
                            bwd_event.record()
                        bwd_event.wait()
                        distillation_event.wait()

                        all_reduce_event.record()
                        another_model_fwd_event.record()
                        with torch.cuda.stream(all_reduce_stream):
                            all_reduce_event.wait()
                            self.all_reduce(overflow_buf)
                            all_reduce_event.record()

                        if step >= self.args.burnin_steps:
                            with torch.cuda.stream(
                                    another_model_fwd_stream):
                                another_model_fwd_event.wait()
                                with torch.no_grad():
                                    another_output = self.forward(
                                        self.another_model, next_batch,
                                        calc_loss=False)
                                another_model_fwd_event.record()
                        all_reduce_event.wait()
                        another_model_fwd_event.wait()

                        global_step = self.take_optimizer_step(global_step)

                        average_loss += loss.item()
                        average_dloss_0 += dloss0.item()
                        average_dloss_1 += dloss1.item()
                        if global_step % self.args.log_freq == 0:
                            divisor = self.args.log_freq
                            if self.is_main_process():
                                print(
                                    "Team: {} Step:{} Average Loss = {} Average dLoss = {} {}"
                                    .format(
                                        self.team, global_step,
                                        average_loss / divisor,
                                        average_dloss_0 / divisor,
                                        average_dloss_1 / divisor)
                                )
                            average_loss = 0
                            average_dloss_0 = 0
                            average_dloss_1 = 0

                        if global_step >= self.args.max_steps or \
                            (global_step %
                             self.args.num_steps_per_checkpoint) == 0:
                            if self.team_rank == 0:
                                # Save a trained model
                                logger.info("** ** Saving model ** **")
                                self.snapshot.save(global_step, f_id, files)

                        if global_step >= self.args.max_steps:
                            del train_dataloader
                            torch.distributed.barrier()
                            if torch.distributed.get_rank() == 0:
                                print("Total time taken {}".format(
                                    time.time() - begin))
                            return self.args
                        batch = next_batch

                    del train_dataloader
                    # Make sure pool has finished and switch train_dataloader
                    # NOTE: Will block until complete
                    train_dataloader, data_file = dataset_future.result(
                        timeout=None)

                epoch += 1

    def train_online_distillation_logit(self):
        global_step = self.snapshot.global_step or 0

        if self.is_main_process():
            print("SEED {}".format(self.args.seed))
            logger.info("***** Running training *****")
            # logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.args.train_batch_size)
            print("  LR = ", self.args.learning_rate)
            print("  Online Distillation")
            print("Training. . .")

        self.model.train()
        average_loss = 0.0  # averaged loss every self.args.log_freq steps
        average_dloss_0 = 0.0
        average_dloss_1 = 0.0
        epoch = 0
        begin = None

        # Note: We loop infinitely over epochs, termination is handled via
        #       iteration count
        rng = random.Random(self.args.data_seed)
        cnt = 0
        with ThreadPoolExecutor(1) as pool:
            while True:
                cnt += 1

                step = global_step
                if self.args.phase2:
                    step += self.args.phase1_end_step
                if step < self.args.burnin_steps:
                    dataset_future, f_start_id, files, data_file = \
                        self.init_dataloader(epoch, pool)
                    use_same_data = False
                else:
                    torch.manual_seed(self.args.data_seed + cnt)
                    dataset_future, f_start_id, files, data_file = \
                        self.init_dataloader(epoch, pool, rng)
                    use_same_data = True
                previous_file = data_file
                train_dataloader, _ = dataset_future.result(timeout=None)

                overflow_buf = torch.cuda.IntTensor([0])

                for f_id in range(f_start_id + 1, len(files)):
                    logger.info("file no %s file %s" % (f_id, previous_file))
                    dataset_future, data_file = \
                        self.update_dataloader(pool, f_id, files)
                    previous_file = data_file

                    for batch in train_dataloader:
                        if begin is None:
                            begin = time.time()
                        step = global_step
                        if self.args.phase2:
                            step += self.args.phase1_end_step
                        if step == self.args.burnin_steps and \
                                not use_same_data:
                            break

                        batch = [t.to(self.device) for t in batch]
                        _, _, _, masked_lm_labels, _ = batch

                        aout0 = None
                        aout1 = None
                        if step < self.args.burnin_steps:
                            loss = self.forward(self.model, batch)
                            dloss0 = torch.zeros(())
                            dloss1 = torch.zeros(())
                        else:
                            out0, out1 = self.forward(self.model, batch,
                                                      calc_loss=False)
                            mask = masked_lm_labels.view(-1)

                            c = out0.shape[-1]
                            # Send logit that are not maksed
                            dout0 = out0.view(-1, c)
                            dout0 = dout0[mask != -1]
                            with torch.no_grad():
                                aout0 = dout0.detach().clone()
                                aout1 = out1.detach().clone()
                                flat_dist_call(
                                    [aout0, aout1],
                                    torch.distributed.all_reduce,
                                    (torch.distributed.ReduceOp.SUM,
                                     self.equalize_data_group))
                                aout0 = aout0 * self.size - dout0
                                aout1 = aout1 * self.size - out1
                            loss = self.loss(out0, out1, batch)
                            dloss0 = \
                                self.compute_distillation_loss(dout0, aout0)
                            dloss1 = \
                                self.compute_distillation_loss(out1, aout1)
                            dloss = dloss0 + dloss1
                            loss = loss + \
                                self.args.distillation_weight * dloss
                        self.backward(loss)

                        self.all_reduce(overflow_buf)
                        global_step = self.take_optimizer_step(global_step)

                        average_loss += loss.item()
                        average_dloss_0 += dloss0.item()
                        average_dloss_1 += dloss1.item()
                        if global_step % self.args.log_freq == 0:
                            divisor = self.args.log_freq
                            if self.is_main_process():
                                print(
                                    "Team: {} Step:{} Average Loss = {} Average dLoss = {} {}"
                                    .format(
                                        self.team, global_step,
                                        average_loss / divisor,
                                        average_dloss_0 / divisor,
                                        average_dloss_1 / divisor)
                                )
                            average_loss = 0
                            average_dloss_0 = 0
                            average_dloss_1 = 0

                        if global_step >= self.args.max_steps or \
                            (global_step %
                             self.args.num_steps_per_checkpoint) == 0:
                            if self.team_rank == 0:
                                # Save a trained model
                                logger.info("** ** Saving model ** **")
                                self.snapshot.save(global_step, f_id, files)

                        if global_step >= self.args.max_steps:
                            del train_dataloader
                            torch.distributed.barrier()
                            if torch.distributed.get_rank() == 0:
                                print("Total time taken {}".format(
                                    time.time() - begin))
                            return self.args

                    del train_dataloader
                    # Make sure pool has finished and switch train_dataloader
                    # NOTE: Will block until complete
                    train_dataloader, data_file = dataset_future.result(
                        timeout=None)

                    if step == self.args.burnin_steps and not use_same_data:
                        break

                epoch += 1


def main():
    trainer = Trainer()
    trainer.parse_arguments()
    trainer.setup_training()
    random.seed(trainer.args.seed)
    np.random.seed(trainer.args.seed)
    torch.manual_seed(trainer.args.seed)

    trainer.prepare_model_and_optimizer()
    trainer.prepare_snapshot()

    if trainer.args.online_distillation == "none":
        trainer.train_simple()
    elif trainer.args.online_distillation == "original":
        trainer.train_online_distillation_original()
    elif trainer.args.online_distillation == "overlap":
        trainer.train_online_distillation_overlap()
    elif trainer.args.online_distillation == "logit":
        trainer.train_online_distillation_logit()


if __name__ == "__main__":
    args = main()
