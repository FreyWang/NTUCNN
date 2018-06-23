import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class HCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 32, kernel_size=3, padding=1),
            self.maxPooling,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.5),
            self.maxPooling,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.5),
            self.maxPooling,
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.5),
            self.maxPooling,
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, config.NUM_CLASSES)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param x: list  [data1, data2...], data1.shape = (N, C, H, W)
        :return:
        """
        x1, x2 = x
        x = self.layer1(x1)
        x = x.view(x.size(0), x.size(3), x.size(2), x.size(1))
        x = self.layer2(x)
        y = self.layer1(x2)
        y = y.view(y.size(0), y.size(3), y.size(2), y.size(1))
        y = self.layer2(y)
        z = torch.cat((x, y), 1)
        z = self.layer3(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return z

if  __name__ == '__main__':
    x1 = Variable(torch.from_numpy(np.random.rand(2,3,32,25)).float())
    x2 = Variable(torch.from_numpy(np.random.rand(2,3,31,25)).float())
    x = [x1, x2]
    hcn = HCN()
    y = hcn(x)
