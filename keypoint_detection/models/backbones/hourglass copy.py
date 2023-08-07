import argparse

import torch
import torch.nn as nn

from keypoint_detection.models.backbones.base_backbone import Backbone

"""An hourglass module first downsamples the input features by a series of
convolution and max pooling layers. It then upsamples the features back to the
original resolution by a series of upsampling and convolution layers. Since details
are lost in the max pooling layers, skip layers are added to bring back the de-
tails to the upsampled features."""

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Hourglass(Backbone):
    def __init__(
        self, n_channels_in=3, n_channels=32, n_hourglasses=1, **kwargs
    ):
        super(Hourglass, self).__init__()
        self.n_stages = 4
        self.n_channels = n_channels
        self.n_channels_in = n_channels_in

        # Down-sampling layers
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Hourglass blocks
        self.hourglass_stages = self.build_hourglass(self.n_channels)

    def build_hourglass_block(self, num_channels):
        return nn.Sequential(
            HourglassBlock(num_channels, num_channels//2),
            self.maxpool
        )

    def forward(self, x):
        # Initial convolution
        print("x ", x.size())
        x = self.conv1(x)
        print("conv1", x.size())
        x = self.relu(x)
        print("relu ", x.size())

        # Hourglass blocks
        outputs = []
        for i in range(self.n_hourglasses):
            x = self.hourglass_stages[i](x)
            print("x ", i, " ", x.size())
            outputs.append(x)
            
        print("output ", x.size())
        return x

    def get_n_channels_out(self):
        return self.n_channels

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
# n_hourglasses = 6, Params: 12.9M

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