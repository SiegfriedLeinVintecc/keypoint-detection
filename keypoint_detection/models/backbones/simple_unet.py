from keypoint_detection.models.backbones.base_backbone import Backbone
from keypoint_detection.models.backbones.unet import UpSamplingBlock
from keypoint_detection.models.backbones.unet import MaxPoolDownSamplingBlock
from keypoint_detection.models.backbones.unet import ResNetBlock

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUnet(Backbone):
    def __init__(self, **kwargs):
        super(SimpleUnet, self).__init__()
        n_channels_in = 3
        n_downsampling_layers = 1 # 2
        n_resnet_blocks= 1 # 3 
        n_channels=16 # 32
        kernel_size=3
        self.n_channels = n_channels 
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size, padding="same")

        # create ModuleLists to ensure layers are discoverable by torch (lightning) for e.g. model summary and bringing to cuda.
        # https://pytorch.org/docs/master/generated/torch.nn.ModuleList.html#torch.nn.ModuleList
        self.downsampling_blocks = nn.ModuleList(
            [MaxPoolDownSamplingBlock(n_channels, n_channels, kernel_size) for _ in range(n_downsampling_layers)]
        )
        self.resnet_blocks = nn.ModuleList([ResNetBlock(n_channels, n_channels) for _ in range(n_resnet_blocks)])
        self.upsampling_blocks = nn.ModuleList(
            [
                UpSamplingBlock(n_channels_in=n_channels, n_channels_out=n_channels, kernel_size=kernel_size)
                for _ in range(n_downsampling_layers)
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        x = self.conv1(x)

        for block in self.downsampling_blocks:
            skips.append(x)
            x = block(x)

        for block in self.resnet_blocks:
            x = block(x)

        for block in self.upsampling_blocks:
            x_skip = skips.pop()
            x = block(x, x_skip)
        return x

    def get_n_channels_out(self):
        return 16
    
    
if __name__ == "__main__":
    print(Backbone._backbone_registry)
