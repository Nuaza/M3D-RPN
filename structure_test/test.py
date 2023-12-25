import logging
import torch_npu

from torchvision import models

CALCULATE_DEVICE = "npu:0"

def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


if __name__ == '__main__':
    torch_npu.npu.set_device(CALCULATE_DEVICE)
    densenet121 = models.densenet121()
    densenet121 = densenet121.to(CALCULATE_DEVICE)
    print(next(densenet121.parameters()).device)

