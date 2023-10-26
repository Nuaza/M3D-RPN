import logging
import torch
import torch.nn as nn
from torchvision import models
from backbones.densenet121 import DenseNet

logging.basicConfig(level=logging.INFO)


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


def replace_to_SiLU(layer):
    layer.relu1 = nn.SiLU(inplace=True)
    layer.relu2 = nn.SiLU(inplace=True)


if __name__ == '__main__':
    densenet121_model = DenseNet()
    torchvision_model = models.densenet121(pretrained=True)
    densenet121_model.load_state_dict(torchvision_model.state_dict())
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

    # SiLU
    densenet121.relu0 = nn.SiLU(inplace=True)
    densenet121.transition1.relu = nn.SiLU(inplace=True)
    densenet121.transition2.relu = nn.SiLU(inplace=True)
    densenet121.transition3.relu = nn.SiLU(inplace=True)
    replace_to_SiLU(densenet121.denseblock1.denselayer1)
    replace_to_SiLU(densenet121.denseblock1.denselayer2)
    replace_to_SiLU(densenet121.denseblock1.denselayer3)
    replace_to_SiLU(densenet121.denseblock1.denselayer4)
    replace_to_SiLU(densenet121.denseblock1.denselayer5)
    replace_to_SiLU(densenet121.denseblock1.denselayer6)
    replace_to_SiLU(densenet121.denseblock2.denselayer1)
    replace_to_SiLU(densenet121.denseblock2.denselayer2)
    replace_to_SiLU(densenet121.denseblock2.denselayer3)
    replace_to_SiLU(densenet121.denseblock2.denselayer4)
    replace_to_SiLU(densenet121.denseblock2.denselayer5)
    replace_to_SiLU(densenet121.denseblock2.denselayer6)
    replace_to_SiLU(densenet121.denseblock2.denselayer7)
    replace_to_SiLU(densenet121.denseblock2.denselayer8)
    replace_to_SiLU(densenet121.denseblock2.denselayer9)
    replace_to_SiLU(densenet121.denseblock2.denselayer10)
    replace_to_SiLU(densenet121.denseblock2.denselayer11)
    replace_to_SiLU(densenet121.denseblock2.denselayer12)
    replace_to_SiLU(densenet121.denseblock3.denselayer1)
    replace_to_SiLU(densenet121.denseblock3.denselayer2)
    replace_to_SiLU(densenet121.denseblock3.denselayer3)
    replace_to_SiLU(densenet121.denseblock3.denselayer4)
    replace_to_SiLU(densenet121.denseblock3.denselayer5)
    replace_to_SiLU(densenet121.denseblock3.denselayer6)
    replace_to_SiLU(densenet121.denseblock3.denselayer7)
    replace_to_SiLU(densenet121.denseblock3.denselayer8)
    replace_to_SiLU(densenet121.denseblock3.denselayer9)
    replace_to_SiLU(densenet121.denseblock3.denselayer10)
    replace_to_SiLU(densenet121.denseblock3.denselayer11)
    replace_to_SiLU(densenet121.denseblock3.denselayer12)
    replace_to_SiLU(densenet121.denseblock3.denselayer13)
    replace_to_SiLU(densenet121.denseblock3.denselayer14)
    replace_to_SiLU(densenet121.denseblock3.denselayer15)
    replace_to_SiLU(densenet121.denseblock3.denselayer16)
    replace_to_SiLU(densenet121.denseblock3.denselayer17)
    replace_to_SiLU(densenet121.denseblock3.denselayer18)
    replace_to_SiLU(densenet121.denseblock3.denselayer19)
    replace_to_SiLU(densenet121.denseblock3.denselayer20)
    replace_to_SiLU(densenet121.denseblock3.denselayer21)
    replace_to_SiLU(densenet121.denseblock3.denselayer22)
    replace_to_SiLU(densenet121.denseblock3.denselayer23)
    replace_to_SiLU(densenet121.denseblock3.denselayer24)
    replace_to_SiLU(densenet121.denseblock4.denselayer1)
    replace_to_SiLU(densenet121.denseblock4.denselayer2)
    replace_to_SiLU(densenet121.denseblock4.denselayer3)
    replace_to_SiLU(densenet121.denseblock4.denselayer4)
    replace_to_SiLU(densenet121.denseblock4.denselayer5)
    replace_to_SiLU(densenet121.denseblock4.denselayer6)
    replace_to_SiLU(densenet121.denseblock4.denselayer7)
    replace_to_SiLU(densenet121.denseblock4.denselayer8)
    replace_to_SiLU(densenet121.denseblock4.denselayer9)
    replace_to_SiLU(densenet121.denseblock4.denselayer10)
    replace_to_SiLU(densenet121.denseblock4.denselayer11)
    replace_to_SiLU(densenet121.denseblock4.denselayer12)
    replace_to_SiLU(densenet121.denseblock4.denselayer13)
    replace_to_SiLU(densenet121.denseblock4.denselayer14)
    replace_to_SiLU(densenet121.denseblock4.denselayer15)
    replace_to_SiLU(densenet121.denseblock4.denselayer16)

    logging.info(densenet121_model)
    for param in densenet121.parameters():
        print(param)
