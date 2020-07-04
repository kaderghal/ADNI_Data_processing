import os
import sys
import errno
import random
import pickle
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from torch import nn
from torch import optim


import matplotlib.pyplot as plt



#==============================================================================
# Network definition
#==============================================================================

class SE_HIPP_3D_Net(nn.Module):
    def __init__(self):
        super(SE_HIPP_3D_Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 32, kernel_size=4, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64*7*7, 120)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x): 
        
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.bn2(x)
        x = self.relu(x)
        # print("size", x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        # print("size", x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features