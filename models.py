import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from einops import rearrange
import numpy as np
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        res_out = self.res(img)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out

class HyperNet_trans(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet_trans, self).__init__()

        mean = torch.ones((3,224,224)).cuda()
        std = torch.ones((3,224,224)).cuda()
        mean[0,:,:]=0.485
        mean[1,:,:]=0.456
        mean[2,:,:]=0.406
        std[0,:,:]=0.229
        std[1,:,:]=0.224
        std[2,:,:]=0.225 
        self.mean = mean
        self.std = std
        print('Using normalizing in HyperNet_trans.')

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        img = (img-self.mean)/self.std
        res_out = self.res(img)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out

class HyperNet_Depth(nn.Module):
    """
    Hyper network with depth for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet_Depth, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1_1 = nn.Conv2d(2048, 1024, 1, padding=(0, 0)) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(1024, 512, 1, padding=(0, 0)) #1024, 512
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)) #512, self.hyperInChn
        self.relu3 = nn.ReLU(inplace=True)

        print('Using Affine as affine_module') #记得把init加回来！！
        self.affine1 = Affine(1024) #1024 #512
        self.affine2 = Affine(512) #512 #224
        self.affine3 = Affine(self.hyperInChn)
        self.depth_pool = nn.AdaptiveAvgPool2d((7,7))
        
        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            # print('m_name',m_name) #res pool conv1_1 relu1 affine1
            if i > 2 and 'relu' not in m_name and 'affine' not in m_name and 'pool' not in m_name:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)
            if 'affine' in m_name:
                nn.init.kaiming_normal_(self._modules[m_name].mlp_shared[0].weight.data)
                nn.init.kaiming_normal_(self._modules[m_name].mlp_gamma.weight.data)
                nn.init.kaiming_normal_(self._modules[m_name].mlp_beta.weight.data)

    def forward(self, img, imgd):
        feature_size = self.feature_size

        res_out = self.res(img)
        depth = self.depth_pool(imgd)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1_1(res_out['hyper_in_feat']) # torch.Size([96, 1024, 7, 7])
        hyper_in_feat = self.relu1(hyper_in_feat) #hyper_in_feat:torch.Size([96, 1024, 7, 7]) depth:torch.Size([96, 1, 7, 7])
        hyper_in_feat = self.affine1(hyper_in_feat,depth)
        hyper_in_feat = self.conv1_2(hyper_in_feat)
        hyper_in_feat = self.relu2(hyper_in_feat) #torch.Size([96, 512, 7, 7])
        hyper_in_feat = self.affine2(hyper_in_feat,depth)
        hyper_in_feat = self.conv1_3(hyper_in_feat)
        hyper_in_feat = self.relu3(hyper_in_feat) #torch.Size([96, 112, 7, 7])
        hyper_in_feat = self.affine3(hyper_in_feat,depth) # torch.Size([96, 112, 7, 7])
        hyper_in_feat = hyper_in_feat.view(-1, self.hyperInChn, feature_size, feature_size)
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out
    
class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x) #torch.Size([1, 112, 1, 1])
        # q = F.dropout(q)
        # print('q.shape0',q.shape)
        q = self.l2(q) #torch.Size([1, 56, 1, 1])
        # print('q.shape1',q.shape)
        q = self.l3(q) #torch.Size([1, 28, 1, 1])
        # print('q.shape2',q.shape)
        q = self.l4(q).squeeze() #torch.Size([])
        # print('q.shape3',q.shape)
        return q
    
class TargetNet_vis(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet_vis, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q1 = self.l1(x)
        q2 = self.l2(q1) 
        q3 = self.l3(q2)
        q4 = self.l4(q3).squeeze() 
        return [q1,q2,q3,q4]


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        out = {}
        out['hyper_in_feat'] = x
        out['target_in_vec'] = vec

        return out


def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Affine(nn.Module):
    '''
    input:224*224 Dr; 56*56 feature map
    output:28*28 feature map after linear transform
    first,pooling Dr to 64*64
    second,using Dr to predect gamma and beta for feature
    NOTICE:without normalization now
    '''
    def __init__(self,out_channel):
        super().__init__()
        # self.pooling=nn.MaxPool2d(kernel_size=4,stride=4)
        nhidden = 128 #128
        print('nhidden in Affine:', nhidden)
        ks = 3
        stride = 2
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_channel, kernel_size=ks, stride=1, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, out_channel, kernel_size=ks, stride=1, padding=pw)
    
    def forward(self,feat,depth):
        # depth=self.pooling(depth)
        actv = self.mlp_shared(depth)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = feat * (1 + gamma) + beta 

        return out
    

class Affine_attention(nn.Module):
    '''
    input:224*224 Dr; 56*56 feature map
    output:28*28 feature map after linear transform
    first,pooling Dr to 64*64
    second,using Dr to predect gamma and beta for feature
    NOTICE:without normalization now
    '''
    def __init__(self, spatial_size, feat_channel, d_k, d_v, h, dropout=.1): #out_channel
        super().__init__()
        '''
        :para feat_channel: channel num of feat
        :para spatial_size: Spatial size of input feats
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        print('feat_channel in Affine_attention:',feat_channel)
        d_model = spatial_size ** 2
        self.spatial_size = spatial_size
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.pre_proc1 = nn.Conv2d(1,128,1,1)
        self.pre_proc2 = nn.Conv2d(128,feat_channel,1,1)
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self,feat,depth,attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        """
        requirement: feat:(b,c1,h,w) depth(b,c2,h,w)
        q,k: depth
        v: feat
        """
        depth = self.pre_proc1(depth)
        depth = self.pre_proc2(depth)
        queries = rearrange(depth,'b c h w -> b c (h w)')
        keys = rearrange(depth,'b c h w -> b c (h w)')
        values = rearrange(feat,'b c h w -> b c (h w)')

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = rearrange(out,'b c (h w) -> b c h w',w=self.spatial_size)
        return out