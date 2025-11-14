import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from sklearn.utils.class_weight import compute_class_weight

from Model2.src.hparams import Hparams


class CnnLstm(nn.Module):
    def __init__(self, args: Hparams.args_6s):
        super(CnnLstm, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(args['in_chs'], args['conv1_chs'], kernel_size=args['conv1_ker'],
                      # padding=args['padding'],
                      stride=args['stride1']),
            nn.BatchNorm2d(args['conv1_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout1_prob']),
            nn.MaxPool2d(kernel_size=args['max_pool_ker1'], stride=args['max_pool_stride1']),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(args['conv1_chs'], args['conv2_chs'], kernel_size=args['conv2_ker'],
                      # padding=args['padding'],
                      stride=args['stride2']),
            nn.BatchNorm2d(args['conv2_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout2_prob']),
            nn.MaxPool2d(kernel_size=args['max_pool_ker2'], stride=args['max_pool_stride2']),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(args['conv2_chs'], args['conv3_chs'], kernel_size=args['conv3_ker'],
                      # padding=args['padding'],
                      stride=args['stride3']),
            nn.BatchNorm2d(args['conv3_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout3_prob']),
        )

        self.lstm = nn.LSTM(
            args['lstm_chs'], args['lstm_hidden'],
            # dropout
            batch_first=True # ?
        )

        self.fc = nn.Sequential(
            nn.Linear(args['lstm_hidden'], args['fc_chs']),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(args['fc_chs'], args['out_chs']),
        )
        self.tonal_center_head = nn.Linear(args['fc_chs'], args['tonal_center_out_chs'])
        self.musical_mode_head = nn.Linear(args['fc_chs'], args['mode_out_chs'])

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        # print(0.1)

        c3 = c3.permute(0, 3, 1, 2)
        c3 = c3.flatten(2)
        # print(0.2)

        lstm1, _ = self.lstm(c3)
        # print(0.3)

        f1 = self.fc(lstm1[:, -1, :])
        out_tonal_center = self.tonal_center_head(f1)
        out_musical_mode = self.musical_mode_head(f1)
        # print(0.4)

        return (out_tonal_center, out_musical_mode)
