'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-05 15:59:54
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-07 21:10:13
FilePath: \Bottles-and-Cans-classifier\src\web\reg.py
Description: 
'''
# define model
from torchvision.models import resnet50,efficientnet_v2_s,regnet_y_32gf
import torch
import torch.nn as nn
import torchvision
import numpy as np

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = resnet50()
        self.resnet.fc = nn.Linear(2048, 7)

    def forward(self, x):
        return self.resnet(x)
    
class EfficientNetV2S(nn.Module):
    def __init__(self):
        super(EfficientNetV2S, self).__init__()
        self.efficientnet = efficientnet_v2_s()
        self.efficientnet.classifier = nn.Linear(1280, 2)

    def forward(self, x):
        return self.efficientnet(x)
    
class RegNetY32GF(nn.Module):
    def __init__(self):
        super(RegNetY32GF, self).__init__()
        self.regnet = regnet_y_32gf()
        self.regnet.fc = nn.Linear(3712, 4096)
        self.last_layer = nn.Linear(4096, 8)

    def forward(self, x):
        x = self.regnet(x)
        return self.last_layer(x)