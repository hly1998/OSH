import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

class ResNet_W(nn.Module):
    def __init__(self, label_size, pretrained=True):
        super(ResNet_W, self).__init__()
        self.model_resnet = models.resnet50(pretrained=pretrained)
        layers = list(self.model_resnet.children())[:-1]
        self.model_resnet = nn.Sequential(*layers)
        self.W =  torch.nn.Parameter(torch.randn(label_size, 2048))
        self.BN = nn.BatchNorm1d(label_size, momentum=0.1)

    def forward(self, x, type='c'):
        if type=='c':
            feat = self.model_resnet(x)
            feat = feat.view(feat.size(0), -1)
            feat = F.linear(feat, self.W)
            return self.BN(feat)
        elif type=='l':
            out = x * self.W
            return out
