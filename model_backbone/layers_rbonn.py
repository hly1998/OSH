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

collections.Iterable = collections.abc.Iterable

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)

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

class Bilinear_binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale_factor):
        
        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * scale_factor

        ctx.save_for_backward(weight, scale_factor)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, scale_factor = ctx.saved_tensors
        
        para_loss = 1e-4

        bin = 0.02

        weight_bin = torch.sign(weight) * bin
        
        gradweight = para_loss * (weight - weight_bin * scale_factor) + (gradOutput * scale_factor)
        
        grad_scale_1 = torch.sum(torch.sum(torch.sum(gradOutput * weight,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)
        
        grad_scale_2 = torch.sum(torch.sum(torch.sum((weight - weight_bin * scale_factor) * weight_bin ,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True,dim=1)

        gradscale = grad_scale_1 - para_loss * grad_scale_2

        return gradweight, gradscale

class RBOConv(_ConvNd):
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
        super(RBOConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode='zeros')

        self.generate_scale_factor()
        self.Bilinear_binarization = Bilinear_binarization.apply
        self.out_channels = out_channels
        self.u = Parameter(0.2 * torch.ones(self.out_channels, 1, 1, 1))
        self.thre = 0.6
        
    def generate_scale_factor(self):
        self.scale_factor = Parameter(torch.randn(self.out_channels, 1, 1, 1))

    def recurrent_module(self, alpha, w):
        backtrack_varible = w.grad.clone()
        weight = w - self.u * self.drelu(alpha, w, backtrack_varible)
        return weight

    def drelu(self, alpha, w, backtrack_varible):
        _, idx = torch.sort(alpha, dim=0, descending=False, out=None)
        indicator = (torch.sign(idx.detach() - int(self.out_channels * (1 - self.thre)) + 0.5) - 1).detach()/ (-2)
        return backtrack_varible * indicator

    def forward(self, x):

        scale_factor = torch.abs(self.scale_factor)

        if (self.weight.grad is not None) and (self.training):
            weight = self.recurrent_module(scale_factor, self.weight)
        else:
            weight = self.weight

        new_weight = self.Bilinear_binarization(weight, scale_factor)

        return F.conv2d(x, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)