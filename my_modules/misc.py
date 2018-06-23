import os
import pickle
import sys
import  random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch
import torch.nn.functional as functional

from my_modules import utils


def my_multilabel_soft_margin_loss(inputs, targets, config):
    """Inputs and targets are both Variables."""
    # ATTENTION: multilabel
    weights = torch.ones(inputs.size()).cuda(config.DEFAULT_GPU)
    n_ele = inputs.nelement()

    for i in range(targets.size(0)):
        for j in range(targets.size(1)):
            if targets.data[i, j] == 1:
                weights[i, j] = config.W_POS
            elif targets.data[i, j] == 0:
                weights[i, j] = config.W_NEG
            elif targets.data[i, j] == -1:
                weights[i, j] = 0

    return functional.binary_cross_entropy(
        torch.sigmoid(inputs).contiguous().view(n_ele),
        targets.contiguous().view(n_ele),
        weights.contiguous().view(n_ele),
        size_average=False) / inputs.size(0)


def my_precision_sum(predictions, targets):
    """Both of predictions and targets are Tensors
    Example:
        predictions = [0.2, 0.5 ,0.3]
        targets = [1, 1, 0] means that class 0 and class 1 appear in this sample ,multi-label you should know
     => true_positive = [0, 1]
        sorted_index = [1, 2, 0]
        return 1/3
    """
    assert predictions.size() == targets.size()
    # size = (batch, class)
    n_samples = predictions.size(0)
    prec_sum = 0

    for i in range(n_samples):
        # type = cpu_tensor
        p, t = predictions[i], targets[i]
        # np.where(condition, x, y)
        # if x and y == none ,return a tuple containing the coordinate of element which match the condition
        # t = [1,0,0,1] => true_positive = [0,3]
        true_positive = np.where(t.numpy() == 1)[0]

        if len(true_positive) == 0:
            continue
        # return a tuple (sorted_tensor, indices)
        _, sortind = torch.sort(p, dim=0, descending=True)

        correct = 0
        # if true label match the biggest value, correct +1
        for i in range(len(true_positive)):
            if sortind[i] in true_positive:
                correct += 1

        prec_sum += correct / max(len(true_positive), 1)  # avoid NAN

    return prec_sum


def visualize(log, train_log, config):
    """"Matplotlib version."""
    model_dir = config.MODEL_DIR
    # loss.jpg
    plt.plot(
        list(range(len(log['train']['loss']))),
        log['train']['loss'],
        'r',
        label='train_loss')
    plt.plot(
        list(range(len(log['test']['loss']))),
        log['test']['loss'],
        'g',
        label='test_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'loss.jpg'))
    plt.clf()

    # prec.jpg
    plt.plot(
        list(range(len(log['train']['prec']))),
        log['train']['prec'],
        'b',
        label='train_prec')
    plt.plot(
        list(range(len(log['test']['prec']))),
        log['test']['prec'],
        'y',
        label='test_prec')
    # limit to (0,1)
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'prec.jpg'))
    plt.clf()

    # map.jpg
    plt.plot(list(range(len(log['test']['map']))), log['test']['map'])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'map.jpg'))
    plt.clf()

    # train curve in every batch
    if not train_log:
        return
    data = np.concatenate(train_log)
    plt.plot(list(range(len(data))), data)  # every batch
    plt.savefig(os.path.join(model_dir, 'train_log.jpg'))
    plt.clf()

    # learning rate curve
    rate = config.RATE_CURVE
    param_groups_num = len(config.PARAM_GROUPS)
    x = list(range(len(rate) // param_groups_num))
    y = [[rate[i] for i in range(j, j + param_groups_num)]
         for j in range(0, len(rate), param_groups_num)]
    plt.plot(x, y)
    plt.savefig(os.path.join(model_dir, 'learning_rate.jpg'))
    plt.clf()


def email_images(loss, prec, map_value, epoch, config):
    sender_info = {
        'email': 'report1832@163.com',
        'password': 'wushuangjianji0',
        'server': 'smtp.163.com'
    }
    receiver = config.EMAIL_ADDR

    subject = '{}/{} epoch_{}'.format(
        os.path.basename(config.PROJ_ROOT_DIR),
        os.path.basename(config.__file__).rsplit('.')[0], epoch)

    text = '<h3>map: {:.3f}</h3><h3>prec: {:.3f}</h3><h3>loss: {:.3f}</h3>'.format(
        map_value if map_value is not None else -1, prec, loss)
    image_html = '<br><img src="cid:loss"><img src="cid:prec"><img src="cid:map"><br>'
    content = text + image_html

    images = [{
        'cid': 'loss',
        'path': os.path.join(config.MODEL_DIR, 'loss.jpg')
    }, {
        'cid': 'prec',
        'path': os.path.join(config.MODEL_DIR, 'prec.jpg')
    }, {
        'cid': 'map',
        'path': os.path.join(config.MODEL_DIR, 'map.jpg')
    }]
    utils.send_email(sender_info, receiver, subject, content, images)


def write_voc_result(outputs, imdb, output_dir):
    # outputs: numpy.ndarray  [batch, num_classes]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = [
        'jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
        'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking'
    ]

    for i, class_name in enumerate(classes):
        file_path = os.path.join(output_dir,
                                 'comp9_action_test_{}.txt'.format(class_name))
        with open(file_path, 'w') as f:
            for m in range(imdb['test']['names'].size):
                img_name, pid = imdb['test']['names'][m].rsplit('_', 1)
                line = '{} {} {:.6f}\n'.format(img_name, pid, outputs[m][i])
                f.write(line)
                f.flush()


class HICOLogger:
    def __init__(self, epoch, mode, num_batch, ds_size, config):
        self.config = config

        with open(os.path.join(self.config.MODEL_DIR, 'log.pickle'),
                  'rb') as f:
            log = pickle.load(f)
        with open(
                os.path.join(self.config.MODEL_DIR, 'train_log.pickle'),
                'rb') as f:
            train_log = pickle.load(f)

        self.log = log
        self.train_log = train_log
        self.epoch = epoch
        self.mode = mode
        self.num_batch = num_batch
        self.ds_size = ds_size
        self.batch_size = config.BATCH_SIZE[mode]
        self.mapmeter = utils.MAP()
        self.batch_losses = []
        self.epoch_loss = 0
        self.prec = 0

    def cleanup_batch(self, outputs, labels, loss, batch, hz):
        loss_value = loss.data[0]
        self.epoch_loss += loss_value

        if self.mode == 'train':
            self.batch_losses.append(loss_value)
        elif self.mode == 'test':
            self.mapmeter.add(
                torch.sigmoid(outputs.data).cpu().numpy(),
                labels.data.cpu().numpy())

        batch_prec = my_precision_sum(outputs.data.cpu(), labels.data.cpu())
        # first change to cpu_tensor, then can change to numpy
        self.prec += batch_prec

        # construct bar string
        num_bar = round(batch / self.num_batch * 30)
        bar_str = ' ['
        for _ in range(num_bar):
            bar_str += '='
        for _ in range(30 - num_bar):
            bar_str += ' '
        bar_str += '] '

        batch_info = '\033[1;34m ' + '{b:3d}/{n_b:3d}'.format(
            b=batch, n_b=self.num_batch) + bar_str + \
            'loss {l:.3f}, prec {p:.1f}, speed {hz:.1f} Hz'.format(
                l=loss_value, p=batch_prec / outputs.size(0), hz=hz) + \
            '\033[0m'
        sys.stdout.write('\033[K')
        print(batch_info, end='\r')

    def cleanup_epoch(self):
        sys.stdout.write('\033[K')

        if self.mode == 'train':
            self.train_log.append(self.batch_losses)
            with open(
                    os.path.join(self.config.MODEL_DIR, 'train_log.pickle'),
                    'wb') as f:
                pickle.dump(self.train_log, f)
        elif self.mode == 'test':
            map_value = self.mapmeter.map()
            print('\033[1;34m map:', map_value, '\033[0m')
            self.log['test']['map'].append(map_value)

        self.prec /= self.ds_size
        self.epoch_loss /= self.num_batch
        epoch_info = 'loss {l:.3f}, prec {p:.1f}'.format(
            l=self.epoch_loss, p=self.prec)
        print('\033[1;34m', epoch_info, '\033[0m')
        self.log[self.mode]['prec'].append(self.prec)
        self.log[self.mode]['loss'].append(self.epoch_loss)
        with open(os.path.join(self.config.MODEL_DIR, 'log.pickle'),
                  'wb') as f:
            pickle.dump(self.log, f)

        # plot
        visualize(self.log, self.train_log, self.config)

        if getattr(self.config, 'EMAIL', False) and self.mode == 'test':
            email_images(self.epoch_loss, self.prec, map_value, self.epoch,
                         self.config)


class VOCLogger:
    def __init__(self, epoch, mode, num_batch, ds_size, config):
        self.config = config

        with open(os.path.join(self.config.MODEL_DIR, 'log.pickle'),
                  'rb') as f:
            # read log information
            log = pickle.load(f)
        with open(
                os.path.join(self.config.MODEL_DIR, 'train_log.pickle'),
                'rb') as f:
            train_log = pickle.load(f)
        # loss, prec, map for train and test in every epoch
        self.log = log
        # train loss in every batch
        self.train_log = train_log
        self.epoch = epoch
        self.mode = mode
        self.num_batch = num_batch
        self.ds_size = ds_size
        self.batch_size = config.BATCH_SIZE
        self.max_epoch = config.MAX_EPOCHS
        self.mapmeter = utils.MAP()
        # append all batch loss
        self.batch_losses = []
        self.epoch_loss = 0
        self.prec = 0

    def cleanup_batch(self, outputs, labels, loss, batch, hz):
        """print progress bar"""
        loss_value = loss.data[0]
        self.epoch_loss += loss_value

        if self.mode == 'train':
            self.batch_losses.append(loss_value)

        self.mapmeter.add(
            torch.sigmoid(outputs.data[:, :]).cpu().numpy(),
            labels.data[:, :].cpu().numpy())
        batch_prec = my_precision_sum(outputs.data[:, :].cpu(),
                                      labels.data[:, :].cpu())
        self.prec += batch_prec
        # construct bar string
        num_bar = round(batch / self.num_batch * 30)
        bar_str = ' ['
        for _ in range(num_bar):
            bar_str += '='
        for _ in range(30 - num_bar):
            bar_str += ' '
        bar_str += '] '

        batch_info = '\033[1;34m ' + '{b:3d}/{n_b:3d}'.format(
            b=batch, n_b=self.num_batch) + bar_str + \
            'loss {l:.3f}, prec {p:.4f}, speed {hz:.1f} Hz'.format(
                l=loss_value, p=batch_prec / outputs.size(0), hz=hz) + \
            '\033[0m'
        sys.stdout.write('\033[K')
        print(batch_info, end='\r')

    def cleanup_epoch(self):
        """print loss and precise in every epoch ,as well as plt figure"""
        sys.stdout.write('\033[K')

        if self.mode == 'train':
            # type = list
            self.train_log.append(self.batch_losses)
            with open(
                    os.path.join(self.config.MODEL_DIR, 'train_log.pickle'),
                    'wb') as f:
                pickle.dump(self.train_log, f)

        map_value = self.mapmeter.map()

        self.prec /= self.ds_size
        self.epoch_loss /= self.num_batch
        epoch_info = 'loss {l:.3f}, prec {p:.3f}'.format(
            l=self.epoch_loss, p=self.prec)

        print('\033[1;34m map:', map_value, ',', epoch_info, '\033[0m')
        # append data in every epoch
        self.log[self.mode]['map'].append(map_value)
        self.log[self.mode]['prec'].append(self.prec)
        self.log[self.mode]['loss'].append(self.epoch_loss)
        with open(os.path.join(self.config.MODEL_DIR, 'log.pickle'),
                  'wb') as f:
            pickle.dump(self.log, f)

        # print max map and max precision in test
        if self.mode == 'test':
            self.config.MAX_MAP.append(map_value)
            self.config.MAX_PREC.append(self.prec)
            if self.epoch == (self.max_epoch - 1):
                max_map_index = self.config.MAX_MAP.index(max(self.config.MAX_MAP)) \
                            + self.max_epoch - len(self.config.MAX_MAP)
                print(self.config.CONFIG_NAME)
                print('best map in test :{},in epcho {}'.format(max(self.config.MAX_MAP),
                      max_map_index))
                max_prec_index = self.config.MAX_PREC.index(max(self.config.MAX_PREC)) \
                           + self.max_epoch - len(self.config.MAX_PREC)
                print('best precision in test :{},in epcho {}'.format(max(self.config.MAX_PREC),
                                                                     max_prec_index))
                # print hyper-parameter
                print('batch_size = {}\nparameter_group = {}'.format(self.config.BATCH_SIZE,
                                                                     self.config.PARAM_GROUPS))


        # plot
        visualize(self.log, self.train_log, self.config)

        if getattr(self.config, 'EMAIL', False) and self.mode == 'test':
            email_images(self.epoch_loss, self.prec, map_value, self.epoch,
                         self.config)
