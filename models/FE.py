import torch
import torch.nn as nn

class SpatialAttn(nn.Module):
    """Parameter-Free Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        input = x
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)

        out = input * z + input

        return out

class ChannelAttn(nn.Module):
    """Parameter-Free Channel Attention Layer"""
    def __init__(self):
        super(ChannelAttn, self).__init__()

    def forward(self, x):
        # global spatial averaging # e.g. 32,2048,24,8
        input = x
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = x.mean(-1, keepdim=True)  # [1,512,1]
        x = x.view(x.size(0), -1)   # [1,512]

        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),c,1,1)#[1,512,1,1]

        out = input * z + input

        return out


