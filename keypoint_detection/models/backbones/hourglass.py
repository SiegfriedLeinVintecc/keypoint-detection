import argparse

import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if self.bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, n_channels_in, n_channels):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(n_channels_in)
        self.conv1 = Conv(n_channels_in, int(n_channels/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(n_channels/2))
        self.conv2 = Conv(int(n_channels/2), int(n_channels/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(n_channels/2))
        self.conv3 = Conv(int(n_channels/2), n_channels, 1, relu=False)
        self.skip_layer = Conv(n_channels_in, n_channels, 1, relu=False)
        if n_channels_in == n_channels:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class Hourglass(Backbone):
    def __init__(
        self, n_channels_in=3, n_channels=32, n_hourglasses=1, **kwargs
    ):
        super(Hourglass, self).__init__()
        self.n_hourglasses = n_hourglasses
        self.n_channels_in = n_channels_in
        self.n_channels = n_channels

        self.up1 = Residual(self.n_channels_in, self.n_channels_in)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(self.n_channels_in, n_channels)
        # Recursive hourglass
        if self.n_hourglasses > 1:
            self.low2 = Hourglass(n_channels_in=n_channels, n_channels=n_channels*2, n_hourglasses=n_hourglasses-1)
        else:
            self.low2 = Residual(n_channels, n_channels)
        self.low3 = Residual(n_channels, n_channels_in)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x ", x.size())
        up1  = self.up1(x)
        # print("up1 ", up1.size())
        pool1 = self.pool1(x)
        # print("pool1 ", pool1.size())
        low1 = self.low1(pool1)
        # print("low1 ",low1.size())
        low2 = self.low2(low1)
        # print("low2 ",low2.size())
        low3 = self.low3(low2)
        # print("low3 ",low3.size())
        up2  = self.up2(low3)
        # print("up2 ",up2.size())
        # print("up1+up2 ", (up1 + up2).size())
        return up1 + up2

    def get_n_channels_out(self):
        return self.n_channels_in

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("HourglassBackbone")
        parser.add_argument("--n_hourglasses", type=int, default=1)
        return parent_parser

if __name__ == "__main__":
    print(Backbone._backbone_registry)

# Params: 
# n_hourglasses = 1, Params: 8.0K
# n_hourglasses = 4, Params: 806K
# n_hourglasses = 5, Params: 3.2M

# sizes:
# n_hourglasses = 1, Params: 8.0K
# x         torch.Size([4, 3, 512, 512])
# up1       torch.Size([4, 3, 512, 512])
# pool1     torch.Size([4, 3, 256, 256])
# low1      torch.Size([4, n_channels, 256, 256])
# low2      torch.Size([4, n_channels, 256, 256])
# low3      torch.Size([4, 3, 256, 256])
# up2       torch.Size([4, 3, 512, 512])
# up1+up2   torch.Size([4, 3, 512, 512])

# --------------------------------
# n_hourglasses = 4, Params: 806K
#           torch.Size([4, 3, 512, 512])
# up1  		torch.Size([4, 3, 512, 512])
# pool1  	torch.Size([4, 3, 256, 256])
# low1  	torch.Size([4, 32, 256, 256])
# x  		torch.Size([4, 32, 256, 256])
# up1  		torch.Size([4, 32, 256, 256])
# pool1  	torch.Size([4, 32, 128, 128])
# low1  	torch.Size([4, 64, 128, 128])
# x  		torch.Size([4, 64, 128, 128])
# up1  		torch.Size([4, 64, 128, 128])
# pool1  	torch.Size([4, 64, 64, 64])
# low1  	torch.Size([4, 128, 64, 64])
# x  		torch.Size([4, 128, 64, 64])
# up1  		torch.Size([4, 128, 64, 64])
# pool1  	torch.Size([4, 128, 32, 32])
# low1  	torch.Size([4, 256, 32, 32])
# low2  	torch.Size([4, 256, 32, 32])
# low3  	torch.Size([4, 128, 32, 32])
# up2  		torch.Size([4, 128, 64, 64])
# up1+up2  	torch.Size([4, 128, 64, 64])
# low2  	torch.Size([4, 128, 64, 64])
# low3  	torch.Size([4, 64, 64, 64])
# up2  		torch.Size([4, 64, 128, 128])
# up1+up2  	torch.Size([4, 64, 128, 128])
# low2  	torch.Size([4, 64, 128, 128])
# low3  	torch.Size([4, 32, 128, 128])
# up2  		torch.Size([4, 32, 256, 256])
# up1+up2  	torch.Size([4, 32, 256, 256])
# low2  	torch.Size([4, 32, 256, 256])
# low3  	torch.Size([4, 3, 256, 256])
# up2  		torch.Size([4, 3, 512, 512])
# up1+up2  	torch.Size([4, 3, 512, 512])