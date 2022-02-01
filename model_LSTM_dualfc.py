# coding=UTF-8
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import *
import os
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

class CovidNet(nn.Module):
    def __init__(self, bnd=False, bna=False, label_smoothing=False,
                 n_classes=2, hidden_size=1024, emmbedding_size=128):
        super(CovidNet, self).__init__()
        self.add_module('convct_pre1',TimeDistributedConv2d(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_pre1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_pre2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_pre2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_pre3',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_pre3', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_pre4',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_pre4', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))

        self.add_module('convpet_pre1',TimeDistributedConv2d(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_pre1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_pre2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_pre2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_pre3',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_pre3', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_pre4',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_pre4', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))

        self.add_module('conv3dct1', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dct2', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dct3', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dct4', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))

        self.add_module('conv3dpet1', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dpet2', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dpet3', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))
        self.add_module('conv3dpet4', TimeDistributed3d(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])))

        self.add_module('convct_post1_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post1_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_post1_2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post1_2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_post1_3',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post1_3', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_post2_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post2_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_post2_2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post2_2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convct_post3_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolct_post3_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))

        self.add_module('convpet_post1_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post1_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_post1_2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post1_2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_post1_3',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post1_3', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_post2_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post2_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_post2_2',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post2_2', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('convpet_post3_1',TimeDistributedConv2d(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)))
        self.add_module('poolpet_post3_1', TimeDistributedPool2d(nn.MaxPool2d(2, stride=(2,2), padding=0)))
        self.add_module('LSTMct', nn.LSTM(hidden_size=256, num_layers=1,input_size=4096))
        self.add_module('LSTMpet', nn.LSTM(hidden_size=256, num_layers=1, input_size=4096))
        self.add_module('bnct1', nn.Sequential(nn.BatchNorm1d(256), nn.ReLU()))
        self.add_module('bnpet1', nn.Sequential(nn.BatchNorm1d(256), nn.ReLU()))
        self.add_module('Linearct1', nn.Linear(256, 32))
        self.add_module('Linearpet1', nn.Linear(256, 32))
        self.add_module('bnct2', nn.Sequential(nn.BatchNorm1d(32), nn.ReLU()))
        self.add_module('bnpet2', nn.Sequential(nn.BatchNorm1d(32), nn.ReLU()))
        self.add_module('Linearct2', nn.Linear(32, 2))
        self.add_module('Linearpet2', nn.Linear(32, 2))
        self.add_module('bnct3', nn.Sequential(nn.BatchNorm1d(2), nn.ReLU()))
        self.add_module('bnpet3', nn.Sequential(nn.BatchNorm1d(2), nn.ReLU()))
        self.add_module('Sigmoidct', nn.Sigmoid())
        self.add_module('Sigmoidpet', nn.Sigmoid())


    def forward(self, ct, pet):
        convct_pre1 = self.convct_pre1(ct)
        poolct_pre1 = self.poolct_pre1(convct_pre1)
        convct_pre2 = self.convct_pre2(poolct_pre1)
        poolct_pre2 = self.poolct_pre2(convct_pre2)
        convct_pre3 = self.convct_pre3(poolct_pre2)
        poolct_pre3 = self.poolct_pre3(convct_pre3)
        convct_pre4 = self.convct_pre4(poolct_pre3)
        poolct_pre4 = self.poolct_pre4(convct_pre4)

        convpet_pre1 = self.convpet_pre1(pet)
        poolpet_pre1 = self.poolpet_pre1(convpet_pre1)
        convpet_pre2 = self.convpet_pre2(poolpet_pre1)
        poolpet_pre2 = self.poolpet_pre2(convpet_pre2)
        convpet_pre3 = self.convpet_pre3(poolpet_pre2)
        poolpet_pre3 = self.poolpet_pre3(convpet_pre3)
        convpet_pre4 = self.convpet_pre4(poolpet_pre3)
        poolpet_pre4 = self.poolpet_pre4(convpet_pre4)

        poolcat1 = torch.cat([poolct_pre1.unsqueeze(3), poolpet_pre1.unsqueeze(3)], 3)
        poolcat2 = torch.cat([poolct_pre2.unsqueeze(3), poolpet_pre2.unsqueeze(3)], 3)
        poolcat3 = torch.cat([poolct_pre3.unsqueeze(3), poolpet_pre3.unsqueeze(3)], 3)
        poolcat4 = torch.cat([poolct_pre4.unsqueeze(3), poolpet_pre4.unsqueeze(3)], 3)

        conv3dct1 = self.conv3dct1(poolcat1)
        conv3dct2 = self.conv3dct2(poolcat2)
        conv3dct3 = self.conv3dct3(poolcat3)
        conv3dct4 = self.conv3dct4(poolcat4)
        conv3dct1 = torch.multiply(conv3dct1,poolct_pre1)
        conv3dct2 = torch.multiply(conv3dct2,poolct_pre2)
        conv3dct3 = torch.multiply(conv3dct3,poolct_pre3)
        conv3dct4 = torch.multiply(conv3dct4,poolct_pre4)

        conv3dpet1 = self.conv3dpet1(poolcat1)
        conv3dpet2 = self.conv3dpet2(poolcat2)
        conv3dpet3 = self.conv3dpet3(poolcat3)
        conv3dpet4 = self.conv3dpet4(poolcat4)
        conv3dpet1 = torch.multiply(conv3dpet1,poolpet_pre1)
        conv3dpet2 = torch.multiply(conv3dpet2,poolpet_pre2)
        conv3dpet3 = torch.multiply(conv3dpet3,poolpet_pre3)
        conv3dpet4 = torch.multiply(conv3dpet4,poolpet_pre4)

        conv3dct1 = self.convct_post1_1(conv3dct1)
        conv3dct1 = self.poolct_post1_1(conv3dct1)
        conv3dct1 = self.convct_post1_2(conv3dct1)
        conv3dct1 = self.poolct_post1_2(conv3dct1)
        conv3dct1 = self.convct_post1_3(conv3dct1)
        conv3dct1 = self.poolct_post1_3(conv3dct1)
        conv3dct2 = self.convct_post2_1(conv3dct2)
        conv3dct2 = self.poolct_post2_1(conv3dct2)
        conv3dct2 = self.convct_post2_2(conv3dct2)
        conv3dct2 = self.poolct_post2_2(conv3dct2)
        conv3dct3 = self.convct_post3_1(conv3dct3)
        conv3dct3 = self.poolct_post3_1(conv3dct3)

        conv3dpet1 = self.convpet_post1_1(conv3dpet1)
        conv3dpet1 = self.poolpet_post1_1(conv3dpet1)
        conv3dpet1 = self.convpet_post1_2(conv3dpet1)
        conv3dpet1 = self.poolpet_post1_2(conv3dpet1)
        conv3dpet1 = self.convpet_post1_3(conv3dpet1)
        conv3dpet1 = self.poolpet_post1_3(conv3dpet1)
        conv3dpet2 = self.convpet_post2_1(conv3dpet2)
        conv3dpet2 = self.poolpet_post2_1(conv3dpet2)
        conv3dpet2 = self.convpet_post2_2(conv3dpet2)
        conv3dpet2 = self.poolpet_post2_2(conv3dpet2)
        conv3dpet3 = self.convpet_post3_1(conv3dpet3)
        conv3dpet3 = self.poolpet_post3_1(conv3dpet3)
        conv3dct1 = torch.reshape(conv3dct1, [4, 32, 64 * 4 * 4])
        conv3dct2 = torch.reshape(conv3dct2, [4, 32, 64 * 4 * 4])
        conv3dct3 = torch.reshape(conv3dct3, [4, 32, 64 * 4 * 4])
        conv3dct4 = torch.reshape(conv3dct4, [4, 32, 64 * 4 * 4])
        conv3dpet1 = torch.reshape(conv3dpet1, [4, 32, 64 * 4 * 4])
        conv3dpet2 = torch.reshape(conv3dpet2, [4, 32, 64 * 4 * 4])
        conv3dpet3 = torch.reshape(conv3dpet3, [4, 32, 64 * 4 * 4])
        conv3dpet4 = torch.reshape(conv3dpet4, [4, 32, 64 * 4 * 4])
        comct = torch.cat([conv3dct1,conv3dct2,conv3dct3,conv3dct4],2)
        compet = torch.cat([conv3dpet1, conv3dpet2, conv3dpet3, conv3dpet4], 2)
        LSTMct, (hn, cn) = self.LSTMct(torch.transpose(comct,1,0))
        LSTMpet, (hn, cn) = self.LSTMpet(torch.transpose(compet, 1, 0))
        LSTMct = self.bnct1(LSTMct[-1, :, :])
        LSTMpet = self.bnpet1(LSTMpet[-1, :, :])
        logitsct = self.Linearct1(LSTMct)
        logitspet = self.Linearct1(LSTMpet)

        logitsct = self.bnct2(logitsct)
        logitspet = self.bnpet2(logitspet)
        logitsct = self.Linearct2(logitsct)
        logitspet = self.Linearct2(logitspet)

        logitsct = self.bnct3(logitsct)
        logitspet = self.bnpet3(logitspet)
        logitsct = self.Sigmoidct(logitsct)
        logitspet = self.Sigmoidct(logitspet)

        return logitsct, logitspet, LSTMct, LSTMpet


class TimeDistributedConv2d(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributedConv2d, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.module(c_in)

        r_in = c_out.view(batch_size, time_steps, self.module.out_channels, int(H/self.module.stride[0]), int(W/self.module.stride[1]))
        return r_in

class TimeDistributedPool2d(nn.Module):

    def __init__(self, module, batch_first=False):
        super(TimeDistributedPool2d, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.module(c_in)

        r_in = c_out.view(batch_size, time_steps, C, int(H/self.module.stride[0]), int(W/self.module.stride[1]))
        return r_in

class TimeDistributed3d(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed3d, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        batch_size, time_steps, C, S, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, S, H, W)
        c_out = self.module(c_in)
        r_in = c_out.view(batch_size, time_steps, self.module.out_channels, int(S/2), int(H/self.module.stride[1]), int(W/self.module.stride[2]))
        return r_in.squeeze()
