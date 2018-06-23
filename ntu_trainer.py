import numpy as np
import torch.nn as nn
import torch.optim as optim

from my_modules import misc, trainer, utils


class NTUTrainer(trainer.Trainer):
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging."""

    def __init__(self, model, dataset, collate_fn, config):
        super(NTUTrainer, self).__init__(model, dataset, collate_fn, config)
        self.ml_loss = self.onehot_cross_entropy#nn.MultiLabelSoftMarginLoss(size_average=False)#

    def compute_map(self):
        self.model.cuda(self.config.DEFAULT_GPU)
        # Test mode!
        self.model.eval()

        if len(self.config.GPUS) > 1:
            model = nn.DataParallel(
                self.model,
                self.config.GPUS,
                output_device=self.config.DEFAULT_GPU)
        else:
            model = self.model
        # load test data
        data_loader = self.setup_dataloader('test')
        num_batch = len(data_loader)
        mapmeter = utils.MAP()

        for i, data in enumerate(data_loader):
            inputs, labels = data
            input_size = self.get_input_size(inputs)
            inputs, labels = self.format_data(inputs, labels, 'test')
            outputs = self.format_output(model(inputs), input_size, 'test')
            # concatenate predict and labels in every batch
            mapmeter.add(scores=outputs.data.cpu().numpy(), targets=labels.data.cpu().numpy())
            print('{:3d}/{:3d}'.format(i, num_batch))

        print('map: {}'.format(mapmeter.map()))

    def compute_loss(self, outputs, labels):
        # omit the last class because it is invalid in VOC dataset
        loss = self.ml_loss(outputs, labels)
        loss /= outputs.size(0)
        #assert not np.isnan(loss.data[0]), 'loss=NaN'
        return loss


    def setup_logger(self, epoch, mode, num_batch, ds_size):
        logger = misc.VOCLogger(epoch, mode, num_batch, ds_size, self.config)
        return logger
