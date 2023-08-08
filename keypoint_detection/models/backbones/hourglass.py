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
    def __init__(self, in_channels, out_channels, num_hourglass_blocks):
        super(HourglassBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Add BatchNorm after the Conv2d layer
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if num_hourglass_blocks > 1:
            self.hourglassblock = HourglassBlock(out_channels, out_channels*2, num_hourglass_blocks-1)
        else:
            self.hourglassblock = nn.Sequential()
        self.upsample = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),  # Add BatchNorm after the Conv2d layer
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        )
        self.skip_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        downsampled = self.downsample(x)
        # print("down ", downsampled.size())
        hourglassed = self.hourglassblock(downsampled)
        upsampled = self.upsample(hourglassed)
        # print("up ", upsampled.size())
        skip = self.skip_conv(x)
        output = upsampled + skip
        return output

class HourglassModule(nn.Module):
    def __init__(self, n_channels_in, n_channels, n_hg_blocks):
        super(HourglassModule, self).__init__()
        self.n_hg_blocks = n_hg_blocks
        self.initial_conv = nn.Conv2d(n_channels_in, 64, kernel_size=7, stride=2, padding=3)
        self.hourglass_modules = HourglassBlock(64, 64, n_hg_blocks)
        self.final_up = nn.Sequential(
            nn.Conv2d(64, n_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        # print("x ", x.size())
        x = self.initial_conv(x)
        # print("initial ", x.size())
        x = self.hourglass_modules(x)
        output = self.final_up(x)
        # print("output ", output.size())
        return output

class Hourglass(Backbone):
    def __init__(self, n_channels_in=3, n_channels=32, n_hg_blocks=5, n_hourglasses=2, **kwargs):
        super(Hourglass, self).__init__()
        self.n_channels = n_channels
        self.initial_hourglass = self._make_hourglass_module(n_channels_in, n_channels, n_hg_blocks)
        self.hourglasses = nn.ModuleList([
            self._make_hourglass_module(n_channels, n_channels, n_hg_blocks) for _ in range(1, n_hourglasses)
        ])
    def _make_hourglass_module(self, n_channels_in, n_channels, n_hg_blocks):
        return HourglassModule(n_channels_in=n_channels_in, n_channels=n_channels, n_hg_blocks=n_hg_blocks)
        
    def forward(self, x):
        x = self.initial_hourglass(x)
        for hourglass in self.hourglasses:
            x = hourglass(x)
        return x

    def get_n_channels_out(self):
        return self.n_channels

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("HourglassBackbone")
        parser.add_argument("--n_hourglasses", type=int, default=1)
        parser.add_argument("--n_hg_blocks", type=int, default=4)
        return parent_parser

if __name__ == "__main__":
    print(Backbone._backbone_registry)
