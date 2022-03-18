import torch.nn as nn
import torch
from models.FE import ChannelAttn, SpatialAttn
# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.nFeat = 64
        self.ca = ChannelAttn()
        self.sa = SpatialAttn()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.ca(x)#+self.sa(x)
        # x = nn.MaxPool2d(5)(x)
        # x = x.view(x.size(0), -1)
        return x

# x=torch.randn(1,3,84,84)
# conv4=ConvNet4()
# out=conv4(x)
# print(out.shape)