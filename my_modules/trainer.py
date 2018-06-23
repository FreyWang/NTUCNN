import datetime
import math
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader, sampler
from my_modules import utils


class Trainer:
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging.

    Interfaces:
        compute_loss(self, outputs, labels)
        setup_logger(self, epoch, mode, num_batch, ds_size)

    Methods:
        setup_optimizer(self)
            default: Adam

        setup_lr_scheduler(self)
            defalut: None

        setup_dataloader(self, mode)
            default: pytorch dataloader

        get_input_size(self, inputs)
            defalut: For torch.Tensor, return its shape. For a list, recursively
            check if the first element is torch.Tensor and return its shape.
            Otherwise return None.

        format_data(self, inputs, labels, mode)
            default: Construct a Variable for inputs and labels.

        format_output(self, outputs, input_size, mode)
            default: Directly return the outputs. 
    
        cleanup_batch(self, logger, outputs, labels, loss, batch, hz)
            default: Use logger.cleanup_batch(outputs, labels, loss, batch, hz)

        cleanup_epoch(self, logger)
            default: Use logger.cleanup_epoch()
    """

    def __init__(self, model, dataset, collate_fn, config):
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.config = config
        self.latest_state = -1
        self.optimizer = self.setup_optimizer()
        self.latest_state = utils.load_state_dict(
            self.model, self.config.PRETRAIN_PATH, self.config.STATE_DIR,
            self.config.STATE_PREFIX)

    def train(self, test=True):
        """The training process."""
        for _ in range(self.latest_state + 1, self.config.MAX_EPOCHS):
            self.train_epoch()
            if test:
                self.test_epoch()

    def train_epoch(self):
        epoch = self.latest_state + 1
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}'.format(utils.parse_time(t - s)))
        self.model.cpu() # move all parameters to cpu
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.STATE_DIR, '{}_{}.pth'.format(
                       self.config.STATE_PREFIX, epoch)))
        self.latest_state = epoch

    def test_epoch(self):
        epoch = self.latest_state
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        print('\033[1;32m{}: epoch {:3d}/{:3d}\033[0m'.format(
            mode, epoch, self.config.MAX_EPOCHS))
        # move all parameters to gpu
        self.model.cuda(self.config.DEFAULT_GPU)

        # model = ResNet(nn.Module) inherit from nn.Module
        # nn.Module.train() set module to train mode
        # nn.Module.eval() set module to test mode ,different only on dropout and BN
        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()

        # whether parallel
        if len(self.config.GPUS) > 1:  # use multi-GPU
            model = nn.DataParallel(
                self.model,
                self.config.GPUS,
                output_device=self.config.DEFAULT_GPU)
        else:
            model = self.model
        # setup dataloader , torch.dataloader as default
        data_loader = self.setup_dataloader(mode)
        num_batch = len(data_loader)
        logger = self.setup_logger(epoch, mode, num_batch,
                                   len(self.dataset[mode]))

        for i, data in enumerate(data_loader):
            # put all parameters grad to 0
            # data_loader size = (num, batchsize , 3, H, W)
            # data_input_size = (batchsize, 3, H, W)
            model.zero_grad()
            s = time.time()
            # inputs = train/test data
            inputs, labels = data
            assert (type(inputs) == torch.FloatTensor) | (type(inputs) == list), \
                'inputs type is {}'.format(type(inputs))
            if (type(inputs) == list):
                assert (type(data) == torch.FloatTensor for data in inputs)
            # type(input_size) = torch.size() ,[batch, 3, H, W]
            # label.size() = (batchsize, class)
            input_size = self.get_input_size(inputs)
            # change tensor to Variable
            inputs, labels = self.format_data(inputs, labels, mode)
            # network outputs
            outputs = self.format_output(model(inputs), input_size, mode)
            loss = self.compute_loss(outputs, labels)
            t = time.time()
            hz = outputs.size(0) / (t - s)
            self.cleanup_batch(logger, outputs, labels, loss, i, hz)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                # learning decay
                total_batch = i + (self.latest_state + 1) * num_batch
                if total_batch % self.config.LEARNING_RATE_BATCH == 0:
                    self.setup_lr_scheduler()
        self.cleanup_epoch(logger)

    def setup_optimizer(self):
        param_groups = utils.get_param_groups(self.model, self.config)
        optimizer = optim.Adam(param_groups)
        return optimizer

    def setup_lr_scheduler(self):  # learning rate decay
        for param_group in self.optimizer.param_groups:
            self.config.RATE_CURVE.append(param_group['lr'])
            param_group['lr'] *= self.config.LEARNING_RATE_DECAY

    def setup_dataloader(self, mode):
        # dataset[train] or dataset[test] ,batch_size ,sample ,dataloader
        dataset = self.dataset[mode] # VOCDataset object
        # Difficult to understand how it work
        data_loader = dataloader.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE[mode],
            sampler=sampler.RandomSampler(dataset),
            collate_fn=self.collate_fn)
        #print(data_loader)
        return data_loader

    def get_input_size(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.size()
        elif isinstance(inputs, list):
            return self.get_input_size(inputs[0])
        else:
            return None

    def format_data(self, inputs, labels, mode):
        # change gpu tensor to gpu Variable
        if type(inputs) == torch.FloatTensor:
            inputs = Variable(inputs.cuda(self.config.DEFAULT_GPU))
        elif type(inputs == list):
            inputs = [Variable(data.cuda(self.config.DEFAULT_GPU)) for data in inputs]
        labels = Variable(labels.cuda(self.config.DEFAULT_GPU))

        if mode == 'test':
            # do not calculate the grad of variable
            if type(inputs) == torch.FloatTensor:
                inputs.volatile = True
            elif type(inputs) == list:
                for data in inputs:
                    data.volatile = True
        return inputs, labels

    def format_output(self, outputs, input_size, mode):
        return outputs

    def cleanup_batch(self, logger, outputs, labels, loss, batch, hz):
        logger.cleanup_batch(outputs, labels, loss, batch, hz)
        # misc.py - VOClogger - cleanup_batch

    def cleanup_epoch(self, logger):
        logger.cleanup_epoch()

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def setup_logger(self, epoch, mode, num_batch, ds_size):
        """A logger should have the following methods.
        cleanup_batch(self, outputs, labels, loss, batch, hz)
        cleanup_epoch()
        """
        raise NotImplementedError
    def onehot_cross_entropy(self, input, target):
        """
        an implement of onehot type cross_entropy to suit the projection
        :param x: a 2D mini-batch FloatTensor after FC layer, every row stands for a sample
        :param y: a 2D binary mini-batch FloatTensor in onehot form
        :return: an scalar loss
        """
        logsoftmax = nn.LogSoftmax()  # :math:`f_i(x) = log(exp(x_i) / sum_j exp(x_j) )`
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

