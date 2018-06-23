from multiprocessing import pool
import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import dataset
from torchvision import transforms
from skimage import transform
import data_preprocess as data

NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class NTUDataset(dataset.Dataset):
    """
    All other datasets should subclass dataset.Dataset. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, config, mode):
        super(NTUDataset, self).__init__()
        self.config = config
        self.mode = mode
        with open(config.IMDB_PATH, 'rb') as f:
            self.imdb = pickle.load(f)[mode]
        self.images = {}  # {"name": image}
        self.pool = pool.Pool()
        self.index = config.DATA_INDEX
        self.y_index = 0  # contant
        self.scale = True
        # jointdiff data
        if mode == 'train':
            self.X, _ = data.input(self.index)
            y_train, _ = data.input(self.y_index)
            self.Y = one_hot(y_train)  # CrossEntropyLoss target must be longTensor
        else:
            _, self.X = data.input(self.index)
            _, y_test = data.input(self.y_index)
            self.Y = one_hot(y_test)
        self.max = [np.max(data) for data in self.X]   #/or np.max(image)? have to test
        self.imagesize = 224

    def __len__(self):
        if isinstance(self.X, list):
            return self.X[0].shape[0]
        else:
            return self.X.shape[0]

    def __getitem__(self, index):
        """Given index, return image and label"""
        # shape=(C,H,W)
        # img = []
        # for i, data in enumerate(self.X):
        #     # Do scale!
        #     #img.append(self.scale(data[index].astype('float32'), i))
        #     img.append(torch.from_numpy(data[index]).float())   # Do not scale
        img = torch.from_numpy(self.X[index]).float()
        label = torch.from_numpy(self.Y[index].astype('float32'))
        return img, label

    def scale(self, data, i):
        if (data.shape[1] != self.imagesize) | (data.shape[2] != self.imagesize):  # interpolation to 224*224
            data = data.transpose(1, 2, 0)  # H,W,C
            image = transform.resize(data, output_shape=(self.imagesize, self.imagesize),
                                 order=1, preserve_range=True)
            if np.max(image) != 0:
                image = image / np.max(image) * np.max(data)   # keep the same range with original data
            data = image.transpose(2, 0, 1)
        #  must divide global max value
        data = data / self.max[i]   # change to (0~1)
        img = torch.from_numpy(data).float()
        return img


def one_hot(y_):#input:y_ is a list
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = np.array(y_)#change list to array
    y_ = y_.reshape(len(y_))
    #print y_.shape
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def preprocess(img):
    # img is PIL Image [W, H, 3]
    # return the normalized tensor [3, H, W]
    img = transforms.ToTensor()(img)
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    img = NORM(img)

    return img
