import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, config):#input_size, hidden_size, num_layers, num_classes):
        super(lstm, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.NUM_CLASSES)
        self._initialize_weights()

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # out.size() = (batch, T, hidden)  ht for the last layer
        # _ = (h_n, c_n), shape = (num_layers * num_directions, batch, hidden_size)  iter you know
        out = self.lstm_dropout(out)
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

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
        for name, param in self.lstm.named_parameters():
            if name.split('_')[0] == 'weight':
                nn.init.orthogonal(param.data, gain=1)
            if name.split('_')[0] == 'bias':
                # set forget bias
                param.data[self.hidden_size:self.hidden_size * 2].fill_(1)
