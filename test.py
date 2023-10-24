import logging

from backbones.densenet121 import DenseNet
from torchvision import models

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    densenet121 = models.densenet121()
    densenet121_local = DenseNet()
    logging.info(densenet121_local.features)
