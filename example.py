from __future__ import division
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from skimage import io,transform,data
from scipy import  interpolate
import pdb
import matplotlib.pyplot  as plt
plt.switch_backend('agg')
from scipy import interpolate

X = np.memmap('/home/xionghui/data/NTURGBD/data_32frame/X_train.dat',
                    dtype='float32', mode='r', shape=(39889, 32, 150))
X= np.random.randint(1,300,(32, 25, 3))
a = X[:,0,0]
print(np.max(a))
plt.plot(range(len(a)),a,'r')
plt.savefig('/home/xionghui/1.jpg')
plt.clf()
#X = X / np.max(X) * 255
#X2= X.astype(np.uint8)

X3 = transform.resize(X,output_shape=(224, 224),order=1,preserve_range=True)  #(4,4,3)
print(np.max(X3[:,0,0]))
#X3 = X3 / np.max(X3) * np.max(X)
plt.plot(range(len(X3[:,0,0])),X3[:,0,0],'g')
plt.savefig('/home/xionghui/2.jpg')
plt.clf()

#
# zero_image = np.zeros((224,224), dtype='float64')
#
# zero_image[:, 0:150] = X
# # copy to 3 channel
# img = []
# img.append(zero_image)
# img.append(zero_image)
# img.append(zero_image)
# img = np.array(img).astype('float32')  # (3,H,W)
