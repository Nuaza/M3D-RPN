import logging
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
from backbones.densenet121 import DenseNet
from models.PConv import PConv
from models.RefConv import RefConv
from thop import profile

logging.basicConfig(level=logging.INFO)


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


if __name__ == '__main__':

    densenet121_model = models.densenet121(weights=True).to("cuda")
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

    # logging.info(densenet121_model)

    input_shape = (3, 512, 1760)
    logging.info(summary(densenet121_model, input_shape, verbose=0))
