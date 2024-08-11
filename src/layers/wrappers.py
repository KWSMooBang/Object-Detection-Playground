import torch

from typing import List, Optional
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Conv2d):
    """
    A wrapper around 'torch.nn.Conv2d' class 
    to be combined with normalization and activation layer
    """
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation
    
    def forward(self, x):  
        x = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

ConvTranspose2d = nn.ConvTranspose2d
BatchNorm2d = nn.BatchNorm2d
Linear = nn.Linear
