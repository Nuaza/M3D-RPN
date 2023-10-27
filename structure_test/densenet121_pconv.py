import logging
import torch
import torch.nn as nn
from torchvision import models
from backbones.densenet121 import DenseNet
from models.PConv import PConv

logging.basicConfig(level=logging.INFO)


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


def replace_to_SiLU(layer):
    layer.relu1 = nn.SiLU(inplace=True)
    layer.relu2 = nn.SiLU(inplace=True)

if __name__ == '__main__':

    # If import the torchvision model to local codes, use this.
    # densenet121_model = DenseNet()
    # torchvision_model = models.densenet121(pretrained=True)
    # densenet121_model.load_state_dict(torchvision_model.state_dict())
    # densenet121 = densenet121_model.features

    densenet121_model = models.densenet121(weights=True)
    densenet121 = densenet121_model.features

    # dilate
    del densenet121.transition3.pool
    dilate_layer(densenet121.denseblock4.denselayer1.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer2.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer3.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer4.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer5.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer6.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer7.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer8.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer9.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer10.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer11.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer12.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer13.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer14.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer15.conv2, (2, 2))
    dilate_layer(densenet121.denseblock4.denselayer16.conv2, (2, 2))

    # Replace transition Conv2d to PConv
    densenet121.transition1.conv = PConv(256, 1, kernel_size=1)
    densenet121.transition2.conv = PConv(512, 1, kernel_size=1)
    densenet121.transition3.conv = PConv(1024, 1, kernel_size=1)

    logging.info(densenet121_model)
