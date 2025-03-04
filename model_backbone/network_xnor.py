import torch.nn as nn
from model_backbone.layers_xnor import XNORLinear,XNORConv2d
# from network import resnet50
from torchvision import models

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = XNORConv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = XNORConv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = XNORConv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = XNORConv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = XNORConv2d(out_channels, out_channels*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.tanh(x)
        return x

class ResNet_XNOR(nn.Module):
    # 带有初始化的实现
    def __init__(self, hash_bit=64, block=BottleNeck, num_layer = [3, 4, 6, 3], input_channels=3):
        super(ResNet_XNOR, self).__init__()
        self.pretrianed_real_net = models.resnet50(pretrained=True)
        self.in_channels = 64
        self.conv1 = self.pretrianed_real_net.conv1
        self.bn1 = self.pretrianed_real_net.bn1
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.tanh = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.hash_layer = XNORLinear(block.expansion*512, hash_bit)
        self._init_pretrained_weight(num_layer)

    def _init_pretrained_weight(self, num_layer):
        layer_dict = {0:self.layer1, 1:self.layer2, 2:self.layer3, 3:self.layer4}
        p_layer_dict = {0:self.pretrianed_real_net.layer1, 1:self.pretrianed_real_net.layer2, 2:self.pretrianed_real_net.layer3, 3:self.pretrianed_real_net.layer4}
        for num in range(4): 
            layer = layer_dict[num]
            p_layer = p_layer_dict[num]
            layer[0].conv1.init(p_layer[0].conv1)
            layer[0].bn1 = p_layer[0].bn1
            layer[0].conv2.init(p_layer[0].conv2)
            layer[0].bn2 = p_layer[0].bn2
            layer[0].conv3.init(p_layer[0].conv3)
            layer[0].bn3 = p_layer[0].bn3
            layer[0].downsample[0].init(p_layer[0].downsample[0])
            layer[0].downsample[1] = p_layer[0].downsample[1]
            for i in range(1, num_layer[num]):
                layer[i].conv1.init(p_layer[i].conv1)
                layer[i].bn1 = p_layer[i].bn1
                layer[i].conv2.init(p_layer[i].conv2)
                layer[i].bn2 = p_layer[i].bn2
                layer[i].conv3.init(p_layer[i].conv3)
                layer[i].bn3 = p_layer[i].bn3

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                XNORConv2d(self.in_channels, out_channels*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        return x


class ResNet_XNOR_More_Real(nn.Module):
    def __init__(self, hash_bit=64, block=BottleNeck, num_layer = [3, 4, 6, 3], input_channels=3):
        super(ResNet_XNOR_More_Real, self).__init__()
        self.pretrianed_real_net = models.resnet50(pretrained=True)
        self.in_channels = 64
        self.conv1 = self.pretrianed_real_net.conv1
        self.bn1 = self.pretrianed_real_net.bn1
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.tanh = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.hash_layer = nn.Linear(block.expansion*512, hash_bit)
        self._init_pretrained_weight(num_layer)

    def _init_pretrained_weight(self, num_layer):
        layer_dict = {0:self.layer1, 1:self.layer2, 2:self.layer3, 3:self.layer4}
        p_layer_dict = {0:self.pretrianed_real_net.layer1, 1:self.pretrianed_real_net.layer2, 2:self.pretrianed_real_net.layer3, 3:self.pretrianed_real_net.layer4}
        for num in range(4): 
            layer = layer_dict[num]
            p_layer = p_layer_dict[num]
            layer[0].conv1.init(p_layer[0].conv1)
            layer[0].bn1 = p_layer[0].bn1
            layer[0].conv2.init(p_layer[0].conv2)
            layer[0].bn2 = p_layer[0].bn2
            layer[0].conv3.init(p_layer[0].conv3)
            layer[0].bn3 = p_layer[0].bn3
            layer[0].downsample[0] = p_layer[0].downsample[0]
            layer[0].downsample[1] = p_layer[0].downsample[1]
            for i in range(1, num_layer[num]):
                layer[i].conv1.init(p_layer[i].conv1)
                layer[i].bn1 = p_layer[i].bn1
                layer[i].conv2.init(p_layer[i].conv2)
                layer[i].bn2 = p_layer[i].bn2
                layer[i].conv3.init(p_layer[i].conv3)
                layer[i].bn3 = p_layer[i].bn3

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                XNORConv2d(self.in_channels, out_channels*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        return x
