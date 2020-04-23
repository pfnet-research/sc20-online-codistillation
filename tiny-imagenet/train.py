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

import os
import time

import torch
import torch.distributed as dist
import torchvision
from apex import amp
from apex.parallel.distributed import flat_dist_call

import util
import options as opt
from dataloader import get_dali_train_loader, get_dali_val_loader
from learning_rate import get_learning_rate_scheduler
import resnetv2


class Trainer(object):
    def __init__(self, options=None):
        if options is None:
            self.options = opt.Options()
        else:
            self.options = options

    def setup_distributed(self):
        dist.init_process_group(backend='nccl', init_method='env://')

        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.gpu = self.options.local_rank
        torch.cuda.set_device(self.gpu)

        if self.options.distillation:
            assert self.size % 2 == 0
            self.team_size = self.size // 2
            self.team_rank = self.rank % self.team_size
            self.team = self.rank // self.team_size
            team_groups = []
            model_comm_groups = []
            self.team_ranks = [
                [0] + list(range(self.team_size, self.team_size * 2)),
                [self.team_size] + list(range(0, self.team_size))
            ]
            for i in range(2):
                team_groups.append(dist.new_group(ranks=list(range(self.team * self.team_size,
                                                                   (self.team + 1) * self.team_size))))
                model_comm_groups.append(dist.new_group(ranks=self.team_ranks[i]))
            self.team_group = team_groups[self.team]
            self.team_leader = self.team * self.team_size
            self.model_comm_groups = model_comm_groups

            if self.options.equalize_data:
                for i in range(self.size // 2):
                    equalize_distillation_group = dist.new_group(ranks=[i, i + self.size // 2])
                    if i == self.team_rank:
                        self.equalize_distillation_group = equalize_distillation_group
        else:
            self.team_size = self.size
            self.team_rank = self.rank
            self.team = 0
            self.team_group = dist.new_group(ranks=list(range(self.size)))
            self.team_leader = 0

    def dummy_communication(self):
        # hiding irrelevant overheads (actually creating comm groups)
        dummy_tensor = torch.zeros(1).cuda()
        dist.all_reduce(dummy_tensor, group=self.team_group)
        if self.options.distillation:
            if self.options.equalize_data:
                dist.all_reduce(dummy_tensor, group=self.equalize_distillation_group)
            else:
                for i in range(2):
                    if self.rank in self.team_ranks[i]:
                        dist.all_reduce(dummy_tensor, group=self.model_comm_groups[i])
        torch.cuda.synchronize()

    def setup_cuda_primitives(self):
        self.main_stream = torch.cuda.Stream()
        self.forward_stream = torch.cuda.Stream()
        self.another_model_fwd_stream = torch.cuda.Stream()
        self.comm_model_stream = torch.cuda.Stream()
        self.all_reduce_stream = torch.cuda.Stream()
        self.distillation_stream = torch.cuda.Stream()

        self.fwd_event = torch.cuda.Event()
        self.bwd_event = torch.cuda.Event()
        self.another_model_fwd_event = torch.cuda.Event()
        self.all_reduce_event = torch.cuda.Event()
        self.step_event = torch.cuda.Event()
        self.start_distillation_event = torch.cuda.Event()
        self.comm_model_event = torch.cuda.Event()

    def build_model(self):
        num_classes = 200 if self.options.tiny_imagenet else 1000
        if self.options.model == 'resnet50':
            return torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
        elif self.options.model == 'resnet18v2':
            return resnetv2.ResNet18(num_classes=num_classes)
        else:
            raise ValueError('unknown model {}'.format(self.options.model))

    def setup_model(self):
        self.model = self.build_model()
        self.model.cuda()

        if self.options.distillation:
            self.another_model = self.build_model()
            self.another_model.cuda()
            self.another_model_updated = False

    def setup_dataset(self):
        if self.options.equalize_data:
            self.equalized_train_loader, _ = get_dali_train_loader(self.options, self.team_size, self.team_rank, self.team, self.gpu, self.team_rank)
        self.train_loader, self.train_len = get_dali_train_loader(self.options, self.team_size, self.team_rank, self.team, self.gpu, self.rank)
        self.val_loader, self.val_len = get_dali_val_loader(self.options, self.team_size, self.team_rank, self.gpu)
        self.n_iter_per_epoch = self.train_len // (self.options.batchsize * self.team_size)

    def setup_optimizer(self):
        param_groups = [
            {'params': [v for n, v in self.model.named_parameters() if 'bn' in n], 'weight_decay': 0},
            {'params': [v for n, v in self.model.named_parameters() if 'bn' not in n], 'weight_decay': self.options.weight_decay},
        ]
        optimizer = torch.optim.SGD(param_groups,
                                    self.options.lr.initial_lr,
                                    momentum=self.options.momentum,
                                    weight_decay=self.options.weight_decay)
        self.optimizer = optimizer
        self.lr_scheduler = get_learning_rate_scheduler(self.options.lr, self.options.epoch, self.team_size * self.options.batchsize)

        if self.options.fp16:
            if self.options.distillation:
                [self.model, self.another_model], self.optimizer = amp.initialize(
                    [self.model, self.another_model], self.optimizer, opt_level="O2",
                    loss_scale=self.options.loss_scaling or 1,
                    master_weights=True
                )
            else:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level="O2",
                    loss_scale=self.options.loss_scaling or 1,
                    master_weights=True
                )

    def allreduce(self):
        def dummy_all_reduce(*args, **kwargs):
            return dist.all_reduce(*args, **kwargs)

        params = [param.grad.data for param in self.model.parameters()]
        flat_dist_call(params,
                       dummy_all_reduce, (torch.distributed.ReduceOp.SUM, self.team_group))
        for param in params:
            param /= self.team_size

    def allreduce_forward2(self, next_images):
        self.all_reduce_event.record()
        self.another_model_fwd_event.record()

        with torch.cuda.stream(self.all_reduce_stream):
            self.all_reduce_event.wait()
            self.allreduce()
            self.all_reduce_event.record()
        if self.another_model_updated:
            with torch.cuda.stream(self.another_model_fwd_stream):
                self.another_model_fwd_event.wait()
                with torch.no_grad():
                    next_another_output = self.another_model(next_images).detach()
                self.another_model_fwd_event.record()
        else:
            next_another_output = None

        self.all_reduce_event.wait()
        self.another_model_fwd_event.wait()

        return next_another_output

    def allreduce_persistent(self):
        def dummy_all_reduce(*args, **kwargs):
            return dist.all_reduce(*args, **kwargs)

        all_params = list(map(lambda x: x[1], sorted(self.model.state_dict().items())))
        flat_dist_call(all_params,
                       dummy_all_reduce, (torch.distributed.ReduceOp.SUM, self.team_group))
        for param in all_params:
            param /= self.team_size

    def evaluate(self):
        model = self.model
        model.eval()
        local_match = torch.tensor(0)
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.cuda()
                target = target.cuda()

                output = model(data)
                match = accuracy_count_match(output, target)
                local_match += match
            self.val_loader.reset()
        local_match = local_match.float().cuda()
        torch.distributed.all_reduce(local_match, group=self.team_group)
        torch.cuda.synchronize()
        accuracy = local_match / self.val_len
        return accuracy

    def comm_model(self):
        for i in range(2):
            if self.rank == self.team_ranks[i][0]:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (i * self.team_size, self.model_comm_groups[i]))
            elif self.rank in self.team_ranks[i]:
                flat_dist_call([param.data for param in self.another_model.parameters()],
                               torch.distributed.broadcast, (i * self.team_size, self.model_comm_groups[i]))
        self.another_model_updated = True

    def exchange_output(self, output):
        another_output = output.detach().clone()
        dist.all_reduce(another_output, group=self.equalize_distillation_group)
        another_output = another_output - output
        return another_output

    def report(self, i):
        n_iter_per_epoch = self.n_iter_per_epoch
        if int(10 * i / n_iter_per_epoch) != int(10 * (i + 1) / n_iter_per_epoch):
            if self.team_rank == 0:
                elapsed = time.time() - self.time_start
                if self.options.distillation:
                    print('({}) {:3} | {:6} | {:.3f} loss:{:.6f} dloss:{:.6f} acc:{:.6f}'.format(
                        self.team, self.epoch, self.it, elapsed, self.loss_stats.average(),
                        self.dloss_stats.average(), self.acc_stats.average())
                    )
                else:
                    print('{:3} | {:6} | {:.3f} loss:{:.6f} acc:{:.6f}'.format(
                        self.epoch, self.it, elapsed, self.loss_stats.average(), self.acc_stats.average())
                    )
                self.loss_stats.clear()
                self.acc_stats.clear()

    def check_nvprof(self):
        if self.it == self.options.start_nvprof_iter:
            torch.cuda.profiler.cudart().cudaProfilerStart()
        if self.it == self.options.stop_nvprof_iter:
            torch.cuda.profiler.cudart().cudaProfilerStop()

    def compute_distillation_loss(self, output, another_output):
        if self.options.distillation_loss == 'cross_entropy':
            other_distr = torch.softmax(another_output, dim=1)
            return -torch.sum(torch.log_softmax(output, dim=1) * other_distr) / len(output)
        elif self.options.distillation_loss == 'kl_divergence':
            return torch.sum(torch.softmax(output, dim=1) * (torch.log_softmax(output, dim=1) - torch.log_softmax(another_output, dim=1))) / len(output)
        else:
            raise ValueError('unknown distillation loss: {}'.format(self.options.distillation_loss))

    def train_loop_plain(self):
        n_iter_per_epoch = self.n_iter_per_epoch
        self.model.train()

        for i, (images, target) in enumerate(self.train_loader):
            self.check_nvprof()
            self.lr_scheduler(self.optimizer, self.epoch, self.epoch + i / n_iter_per_epoch, self.it)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = self.model(images)
            loss = self.criterion(output, target)
            acc = accuracy_count_match(output, target).float() / float(len(target))

            self.loss_stats.add(loss.detach())
            self.acc_stats.add(acc)

            if self.options.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.allreduce()
            else:
                loss.backward()
                self.allreduce()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.report(i)
            self.it += 1

    def train_loop_distillation_original(self):
        model = self.model
        n_iter_per_epoch = self.n_iter_per_epoch

        for i, (images, target) in enumerate(self.train_loader):
            self.check_nvprof()
            self.lr_scheduler(self.optimizer, self.epoch, self.epoch + i / n_iter_per_epoch, self.it)
            model.train()

            if self.epoch >= self.options.burnin_epoch and self.it % self.options.comm_model_freq == 0:
                self.comm_model()

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = self.criterion(output, target)
            acc = accuracy_count_match(output, target).float() / float(len(target))

            self.loss_stats.add(loss.detach())
            self.acc_stats.add(acc)

            if self.another_model_updated:
                with torch.no_grad():
                    another_output = self.another_model(images).detach()
                dloss = self.compute_distillation_loss(output, another_output)
                self.dloss_stats.add(dloss.detach())
                loss += dloss * self.options.distillation_weight

            if self.options.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.allreduce()
            else:
                loss.backward()
                self.allreduce()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.report(i)
            self.it += 1

    def train_loop_distillation_equalized(self):
        model = self.model
        n_iter_per_epoch = self.n_iter_per_epoch

        if self.epoch >= self.options.burnin_epoch:
            # switch train_loader
            self.train_loader = self.equalized_train_loader

        for i, (images, target) in enumerate(self.train_loader):
            self.check_nvprof()
            self.lr_scheduler(self.optimizer, self.epoch, self.epoch + i / n_iter_per_epoch, self.it)
            model.train()

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = self.criterion(output, target)
            acc = accuracy_count_match(output, target).float() / float(len(target))

            if self.epoch >= self.options.burnin_epoch:
                another_output = self.exchange_output(output)
                dloss = self.compute_distillation_loss(output, another_output)
                self.dloss_stats.add(dloss.detach())
                loss += dloss * self.options.distillation_weight

            self.loss_stats.add(loss.detach())
            self.acc_stats.add(acc)

            if self.options.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.allreduce()
            else:
                loss.backward()
                self.allreduce()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.report(i)
            self.it += 1

    def train_loop_distillation_overlap(self):
        self.setup_cuda_primitives()

        model = self.model
        n_iter_per_epoch = self.n_iter_per_epoch

        iterator = iter(self.train_loader)
        self.epoch = 0

        next_images, next_target = next(iterator)
        next_images = next_images.cuda(non_blocking=True)
        next_target = next_target.cuda(non_blocking=True)
        next_another_output = None
        i = 0

        while True:
            self.check_nvprof()
            images, target = next_images, next_target
            while True:
                try:
                    next_images, next_target = next(iterator)
                    next_images = next_images.cuda(non_blocking=True)
                    next_target = next_target.cuda(non_blocking=True)
                    next_images = next_images.clone()
                    next_target = next_target.clone()
                    break
                except StopIteration:
                    iterator = iter(self.train_loader)

            self.lr_scheduler(self.optimizer, self.epoch, self.epoch + i / n_iter_per_epoch, self.it)
            model.train()

            # --------- forward start ---------
            self.fwd_event.record()
            self.comm_model_event.record()
            if self.epoch >= self.options.burnin_epoch and self.it % self.options.comm_model_freq == 0:
                with torch.cuda.stream(self.comm_model_stream):
                    self.comm_model_event.wait()
                    self.comm_model()
                    self.comm_model_event.record()
            with torch.cuda.stream(self.main_stream):
                self.fwd_event.wait()
                output = model(images)
                loss = self.criterion(output, target)
                acc = accuracy_count_match(output, target).float() / float(len(target))

                if next_another_output is not None:
                    another_output = next_another_output
                    dloss = self.compute_distillation_loss(output, another_output)
                    self.dloss_stats.add(dloss.detach())
                    loss += dloss * self.options.distillation_weight
                self.fwd_event.record()

            self.fwd_event.wait()
            self.comm_model_event.wait()
            # --------- forward end ---------

            # --------- backward, allreduce & forward' start ---------
            self.bwd_event.record()
            with torch.cuda.stream(self.main_stream):
                self.bwd_event.wait()

                # backward -> (allreduce | forward')
                if self.options.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                        next_another_output = self.allreduce_forward2(next_images)
                else:
                    loss.backward()
                    next_another_output = self.allreduce_forward2(next_images)

                self.bwd_event.record()
            self.bwd_event.wait()

            # --------- backward, allreduce & forward' end ---------

            self.loss_stats.add(loss.detach())
            self.acc_stats.add(acc)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.report(i)
            i += 1
            self.it += 1
            if self.it % n_iter_per_epoch == 0:
                # epoch boundary
                i = 0
                self.allreduce_persistent()
                val_acc = self.evaluate()
                if self.team_rank == 0:
                    print('accuracy epoch #{}: {}'.format(self.epoch, val_acc))
                self.epoch += 1

                if self.epoch >= self.options.epoch:
                    break

    def train(self):
        model = self.model
        self.time_start = time.time()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.loss_stats = util.Stats()
        self.acc_stats = util.Stats()
        self.dloss_stats = util.Stats()

        self.it = 0

        flat_dist_call([param.data for param in model.parameters()], dist.broadcast, (self.team_leader, self.team_group))
        self.optimizer.zero_grad()

        if self.options.distillation_overlap:
            assert self.options.distillation
            self.train_loop_distillation_overlap()
        else:
            for epoch in range(self.options.epoch):
                self.epoch = epoch

                if not self.options.distillation:
                    self.train_loop_plain()
                else:
                    if self.options.equalize_data:
                        self.train_loop_distillation_equalized()
                    else:
                        self.train_loop_distillation_original()

                self.allreduce_persistent()
                val_acc = self.evaluate()
                if self.team_rank == 0:
                    print('accuracy epoch #{}: {}'.format(epoch, val_acc))

        if self.team_rank == 0:
            elapsed = time.time() - self.time_start
            print('cost: {:.3f}'.format(elapsed))
        if self.team_rank == 0 and not self.options.no_output_model:
            if self.options.distillation:
                torch.save(model.state_dict(), os.path.join(self.options.out, 'weight.{}.pth'.format(self.team)))
            else:
                torch.save(model.state_dict(), os.path.join(self.options.out, 'weight.pth'))

    def run(self):
        self.setup_distributed()
        self.setup_model()
        self.setup_dataset()
        self.setup_optimizer()

        try:
            os.makedirs(self.options.out)
        except OSError:
            pass

        self.dummy_communication()
        self.train()


def accuracy_count_match(out, target):
    pred = out.max(1).indices
    return pred.eq(target).cpu().sum()


if __name__ == '__main__':
    opts = opt.parse_command_line()
    trainer = Trainer(opts)
    trainer.run()
