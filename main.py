from __future__ import division
import argparse
import importlib
import os
import sys
from torch.utils.data.dataloader import default_collate
import my_modules.trainer
import my_modules.misc
import my_modules.utils
import ntu_dataset
import ntu_trainer
from torchvision import models
import data_preprocess as data
from collections import Iterable


def load_model(config):
    # model_module = models/Resnet.py
    model_module = importlib.import_module('models.{}'.format(
        config.MODEL_FILE_NAME))
    # model = resnet.ResNet(config)
    # getattr(x,y) = x.y
    # getattr(x,y) = x.y
    model = getattr(model_module, config.MODEL_NAME)(config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', type=str, choices=['all', 'train', 'test', 'map'], default='all')
    parser.add_argument('-c', type=int, default=1)
    args = parser.parse_args()

    config = importlib.import_module('myconfs.config_{}'.format(args.c))

    print('==========>> init dataset')

    # return train data and test data
    dataset = {
        'train': ntu_dataset.NTUDataset(config, 'train'),
        'test': ntu_dataset.NTUDataset(config, 'test')
    }


    print('==========>> init model')
    model = load_model(config)

    print('==========>> init trainer')
    model_trainer = ntu_trainer.NTUTrainer(model, dataset, default_collate,
                                           config)

    print('==========>> start to run model')
    if args.m == 'all':  # default setting
        model_trainer.train(test=True)
    elif args.m == 'train':
        model_trainer.train_epoch()
    elif args.m == 'test':
        model_trainer.test_epoch()
    elif args.m == 'map':
        model_trainer.compute_map()
    if isinstance(dataset['train'].index, Iterable):
        print(data.data_name[i] for i in dataset['train'].index)
    else:
        print(data.data_name[dataset['train'].index])


if __name__ == '__main__':
    main()
