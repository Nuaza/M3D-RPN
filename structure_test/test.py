import logging

from torchvision import models

logging.basicConfig(level=logging.INFO)


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


if __name__ == '__main__':
    densenet121 = models.densenet121().features
    logging.info(densenet121)

