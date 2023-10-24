import logging

from backbones.densenet121 import DenseNet
from torchvision import models

logging.basicConfig(level=logging.INFO)


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


if __name__ == '__main__':
    densenet121 = models.densenet121().features
    del densenet121.transition3.pool
    # dilate_layer(densenet121.denseblock4.denselayer1.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer2.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer3.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer4.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer5.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer6.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer7.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer8.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer9.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer10.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer11.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer12.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer13.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer14.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer15.conv2, 2)
    # dilate_layer(densenet121.denseblock4.denselayer16.conv2, 2)
    # print(densenet121)

    densenet121_local = DenseNet().features
    del densenet121_local.transition3.pool
    # dilate_layer(densenet121_local.denseblock4.denselayer1.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer2.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer3.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer4.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer5.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer6.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer7.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer8.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer9.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer10.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer11.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer12.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer13.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer14.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer15.conv2, 2)
    # dilate_layer(densenet121_local.denseblock4.denselayer16.conv2, 2)
    densenet121_local.denseblock4.denselayer16.conv2.padding = 2
    logging.info(densenet121_local)
    # print(densenet121_local.denseblock4.denselayer16.conv2.padding)
