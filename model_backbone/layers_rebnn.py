import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import math
import torch
from torch.nn import Parameter
import collections
from torch.nn.modules.conv import _ConvNd
from itertools import repeat

threshold = [1e-5, 2e-4]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def round_pass(p):
    p = threshold[0] - F.relu(threshold[0] - p)
    p = threshold[1] + F.relu(p - threshold[1])
    return p

class Binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scaling_factor):

        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * scaling_factor
        with torch.no_grad():
            try:
                old_weight = (weight - weight.grad.clone())
            except:
                old_weight = weight

        ctx.save_for_backward(weight, scaling_factor, old_weight)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, scaling_factor, old_weight = ctx.saved_tensors
        
        C_out, C_in, K_1, K_2 = gradOutput.shape

        x0 = torch.sign(old_weight).reshape(C_out, C_in*K_1*K_2).detach()
        x = torch.sign(weight).reshape(C_out, C_in*K_1*K_2).detach()
        proportion = 1 - F.relu(x0 * x).sum(dim=1).detach() / C_in/ K_1 /K_2
        p, _ = torch.max(torch.abs(gradOutput.detach()).reshape(C_out, C_in*K_1*K_2), 1)
        p = proportion * p

        para_loss = round_pass(p).unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()

        bin = 0.02

        weight_bin = torch.sign(weight) * bin
        
        gradweight = para_loss * (weight - weight_bin * scaling_factor) + (gradOutput * scaling_factor)
        target2 = (weight - weight_bin * scaling_factor) * weight_bin
        
        grad_scale_1 = torch.sum(torch.sum(torch.sum(gradOutput * weight,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)
        
        grad_scale_2 = torch.sum(torch.sum(torch.sum(target2,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)

        gradscale = grad_scale_1 - para_loss * grad_scale_2
        return gradweight, gradscale, None

class BiConv(_ConvNd):
    '''
    Baee layer class for modulated convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BiConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

        self.generate_scaling_factor()
        self.Binarization = Binarization.apply
        self.out_channels = out_channels
        self.init_state = 0
        
    def generate_scaling_factor(self):
        self.scaling_factor = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def forward(self, x):

        scaling_factor = torch.abs(self.scaling_factor)
        new_weight = self.Binarization(self.weight, scaling_factor)

        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
                        