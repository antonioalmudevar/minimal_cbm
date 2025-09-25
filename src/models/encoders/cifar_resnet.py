"""
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
from typing import List, Type

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


MODELS = [
    'resnet8',
    'resnet20', 
    'resnet32', 
    'resnet44', 
    'resnet56', 
    'resnet110', 
    'resnet1202',
    'CifarResNet'
]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class Dropout(nn.Module):
    """
    https://github.com/aredier/monte_carlo_dropout/blob/master/monte_carlo_dropout/mc_dropout.py
    """
    def __init__(self, p: float = 0.5, mc_dropout: bool = False):
        super().__init__()
        self.mc_dropout = mc_dropout
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training or self.mc_dropout)



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        stride: int=1, 
        option: str='A',
        dropout_rate: float=0.,
        mc_dropout: bool=False,
    ):
        super(BasicBlock, self).__init__()

        self.dropout1 = Dropout(dropout_rate, mc_dropout)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.dropout2 = Dropout(dropout_rate, mc_dropout)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x: Tensor) -> Tensor:

        out = self.dropout1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.dropout2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out



class CifarResNet(nn.Module):
    
    def __init__(
        self, 
        block: Type[BasicBlock], 
        num_blocks: List[int], 
        ch_in: int,
        channels: List[int]=[16, 32, 64],
        dropout_rate: float=0.,
        mc_dropout: bool=False,
        last_dropout: bool=False,
    ):
        super().__init__()
        assert len(channels)==3, "This resnet must have 3 channels"
        self.dropout_rate = dropout_rate
        self.mc_dropout = mc_dropout
        self.last_dropout = last_dropout

        self.in_planes = channels[0]

        self.conv1 = nn.Conv2d(
            ch_in, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, channels[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, channels[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, channels[2], num_blocks[2], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout = Dropout(dropout_rate, mc_dropout) if last_dropout else nn.Identity()

        self.apply(_weights_init)


    def _make_layer(
        self, 
        block: Type[BasicBlock], 
        planes: int, 
        num_blocks: int, 
        stride: int=1,
    ) -> nn.Sequential:
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        dropout_rate = 0 if self.last_dropout else self.dropout_rate
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, 'A', dropout_rate, self.mc_dropout
            ))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor, n_samples: int=1) -> Tensor:

        if n_samples>1:
            assert self.mc_dropout, "More than one sample can be used only with MC Dropout"
            if not self.last_dropout:
                x = torch.repeat_interleave(x, n_samples, dim=0)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if self.last_dropout:
            out = torch.repeat_interleave(out, n_samples, dim=0)
            out = self.fc_dropout(out)

        return out


#==========Original ResNets===============
def resnet8(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [1, 1, 1], ch_in, channels, **kwargs)


def resnet20(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [3, 3, 3], ch_in, channels, **kwargs)


def resnet32(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [5, 5, 5], ch_in, channels, **kwargs)


def resnet44(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [7, 7, 7], ch_in, channels, **kwargs)


def resnet56(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [9, 9, 9], ch_in, channels, **kwargs)


def resnet110(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [18, 18, 18], ch_in, channels, **kwargs)


def resnet1202(ch_in: int, channels: List[int] = [16, 32, 64], **kwargs):
    return CifarResNet(BasicBlock, [200, 200, 200], ch_in, channels, **kwargs)