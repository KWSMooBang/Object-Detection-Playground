import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import init
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from backbone import Backbone

__all__ = [
    'ResNet'
]

class ResNet(Backbone):
    def __init__(
        self,
        model='resnet50',
        pretrained=False,
        num_classes=None,
        out_features=None,
        freeze_at=0
    ):
        super(ResNet, self).__init__()

        resnet = eval("models.{0}(weights={1})".format(model, "'DEFAULT'" if pretrained else None))

        if out_features is None:
            out_features = []
        return_nodes = {
            'layer'+str(int(f[3])-1): f
            for f in out_features
        }
        out_features.append('stem')
        return_nodes['maxpool'] = 'stem'

        if num_classes is not None:
            resnet.fc.out_features = num_classes
            return_nodes['fc'] = 'linear'
            out_features.append('linear')

        self.model = create_feature_extractor(
            resnet, return_nodes=return_nodes
        )

        self.out_features = out_features
        self.freeze(freeze_at)

    def forward(self, x):
        assert x.dim() == 4
        
        outputs = self.model(x)

        return outputs
    
    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            layers = get_graph_node_names(self.model)[0][1:5]
            for layer in layers:
                for _, param in eval('self.model.' + layer + '.named_parameters()'):
                    param.requires_grad = False
        for i in range(1, 5):
            if freeze_at > i:
                for _, param in eval('self.model.layer' + str(i) + '.named_parameters()'):
                    param.requires_grad = False
        if freeze_at == 6:
            for _, param in self.model.avgpool.named_parameters():
                param.requires_grad = False
            for _, param in self.model.fc.named_parameters():
                param.requires_grad = False

        return self
