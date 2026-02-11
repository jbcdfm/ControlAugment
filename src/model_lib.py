import torch
import torch.nn as nn



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnnmodel = nn.Sequential(
            nn.Conv2d(3, 20, 5),    # Convolution layer
            nn.ReLU(),              # Activation function
            nn.AvgPool2d(2, 2),     # Pooling layer
            nn.Conv2d(20,50, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.fullyconn = nn.Sequential(
            nn.Linear(50* 5 * 5, 400),  # Linear Layer
            nn.ReLU(),                  # ReLU Activation Function
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.cnnmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fullyconn(x)
        return x
    
    
    
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