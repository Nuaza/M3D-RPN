import torch
from torchvision import models

if __name__ == '__main__':
    model = "../output/kitti_3d_multi_main/weights/exp1-4.pt"
    state_dict = torch.load(model)
    net = models.densenet121(pretrained=False)
    net.load_state_dict(state_dict, strict=False)
    print(net.features.denseblock3.denselayer1)
