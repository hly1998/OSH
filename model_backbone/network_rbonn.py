# 2023.12.14 Rbonn
# ResNet_Rbonn: Rbonn的ResNet18实现

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import math
import torch
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd
import collections
from itertools import repeat
from model_backbone.layers_rbonn import *

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, param=1e-4):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = RBOConv(inplanes, planes, stride=stride, padding=1, bias=False, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class ResNet_Rbonn(nn.Module):

    def __init__(self, hash_bit=64, block=BasicBlock, layers=[4, 4, 4, 4], zero_init_residual=False):
        super(ResNet_Rbonn, self).__init__()
        self.params=[1e-4, 1e-4, 1e-4, 1e-4]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], param=self.params[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, param=self.params[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, param=self.params[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, param=self.params[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, hash_bit)

    def _make_layer(self, block, planes, blocks, stride=1, param = 1e-4):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
   
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, param=param))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, param=param))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x