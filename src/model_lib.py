# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:15:07 2025

@author: JBC
"""


import torch
import torch.nn as nn
import torch.nn.functional as F





class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnnmodel = nn.Sequential(
            nn.Conv2d(3, 20, 5), # Convolution layer
            nn.ReLU(),          # Activation function
            nn.AvgPool2d(2, 2), # Pooling layer
            nn.Conv2d(20,50, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.fullyconn = nn.Sequential(
            nn.Linear(50* 5 * 5, 400), # Linear Layer
            nn.ReLU(),           # ReLU Activation Function
            # nn.Dropout(p=0.01),#(p=0.25),
            nn.Linear(400, 100),
            nn.ReLU(),
            # nn.Dropout(p=0.01),#(p=0.25),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.cnnmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fullyconn(x)
        return x
    
    
class ResNet9_small(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0.3):
        super(ResNet9_small, self).__init__()

        # Initial Conv Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual Block
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # Downsampling
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # More Downsampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Avg Pooling
            nn.Flatten(),
            nn.Dropout(dropout),  # Dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(x + self.res1(x))  # Residual connection
        x = self.conv2(x)
        x = nn.ReLU(x + self.res2(x))
        x = self.conv3(x)
        x = nn.ReLU(x + self.res3(x))
        x = self.classifier(x)
        return x    
    

# class BasicBlock(nn.Module):
#     expansion = 1  

#     def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.dropout(out)
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet18(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10, dropout=0.0):  # Now in_channels is flexible!
#         super(ResNet18, self).__init__()
#         self.in_channels = 64  # First layer always has 64 filters

#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)

#         self.layer1 = self._make_layer(64, 2, stride=1, dropout=dropout)
#         self.layer2 = self._make_layer(128, 2, stride=2, dropout=dropout)
#         self.layer3 = self._make_layer(256, 2, stride=2, dropout=dropout)
#         self.layer4 = self._make_layer(512, 2, stride=2, dropout=dropout)

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def _make_layer(self, out_channels, num_blocks, stride, dropout):
#         strides = [stride] + [1] * (num_blocks - 1)  
#         layers = []
#         for stride in strides:
#             layers.append(BasicBlock(self.in_channels, out_channels, stride, dropout))
#             self.in_channels = out_channels  
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avg_pool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out
    
    
    
    
    


###############################################################################
# ResNet - Weird one
###############################################################################    


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_block2(in_channels, out_channels, pool=False):
    layers = [nn.BatchNorm2d(in_channels),
              nn.ReLU(inplace=True),
              nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# Resnet model with skip connections
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, first_conv_features=64, p_drop=0.):
        super(ResNet9,self).__init__()
        
        self.conv1 = conv_block(in_channels, first_conv_features)
        self.res1 = nn.Sequential(conv_block2(first_conv_features*1, first_conv_features*1), 
                                  conv_block2(first_conv_features*1, first_conv_features*1))
        
        self.conv2 = conv_block(first_conv_features*1, first_conv_features*2, pool=True)
        self.res2 = nn.Sequential(conv_block2(first_conv_features*2, first_conv_features*2), 
                                  conv_block2(first_conv_features*2, first_conv_features*2))
        
        self.conv3 = conv_block(first_conv_features*2, first_conv_features*4, pool=True)
        self.res3 = nn.Sequential(conv_block2(first_conv_features*4, first_conv_features*4), 
                                  conv_block2(first_conv_features*4, first_conv_features*4))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(first_conv_features*4*4, num_classes))
        
    def forward(self, xb):
        x = self.conv1(xb)
        # out = self.conv2(out)
        x = self.res1(x)  + x # Turn on resnet at this level
        x = self.conv2(x)
        x = self.res2(x) + x
        x = self.conv3(x)
        x = self.res3(x) + x # Turn on resnet at this level
        x = self.classifier(x)
        return x



# Resnet model with skip connections
class ResNet9_v2(nn.Module):
    def __init__(self, in_channels, num_classes, first_conv_features=64):
        super(ResNet9_v2,self).__init__()
        
        self.conv1 = conv_block(in_channels, first_conv_features)
        self.conv2 = conv_block(first_conv_features, first_conv_features*2, pool=True)
        self.res1 = nn.Sequential(conv_block(first_conv_features*2, first_conv_features*2), conv_block(first_conv_features*2, first_conv_features*2))
        
        self.conv3 = conv_block(first_conv_features*2, first_conv_features*4, pool=True)
        self.conv4 = conv_block(first_conv_features*4, first_conv_features*8, pool=True)
        self.res2 = nn.Sequential(conv_block(first_conv_features*8, first_conv_features*8), conv_block(first_conv_features*8, first_conv_features*8))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(first_conv_features*8, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)  + out # Turn on resnet at this level
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out # Turn on resnet at this level
        out = self.classifier(out)
        return out



    
###############################################################################
# AIRBENCH94 
###############################################################################

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################
hyp = {
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
    },
}
def airbench_net():
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    # net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net