# https://zhuanlan.zhihu.com/p/523211244
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os


def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)


if __name__ == '__main__':
    img_dir = "../data/kitti/testing/kitti_100"
    # img_dir = "../data/kitti/testing/image_2"
    images = os.listdir(img_dir)

    # TODO:怎么导入训练好的pt权重
    weight = "../output/kitti_3d_multi_main/weights/exp1-4.pt"
    state_dict = torch.load(weight)
    model = models.densenet121(pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    # model = deeplabv3_resnet50(weights=True, progress=False)
    # model = model.eval()

    # 定义输入图像的长宽，这里要保证每张图片都要相同
    input_H, input_W = 256, 256
    # 生成一个和输入图像大小相同的零矩阵，用于更新梯度
    heatmap = np.zeros([input_H, input_W])
    # print(model)

    layer = model.features.denseblock4.denselayer1
    # print(layer)

    # 遍历文件夹中的所有图像
    for img in images:
        read_img = os.path.join(img_dir, img)
        image = Image.open(read_img)

        # 图像预处理，将图像缩放到固定分辨率，并进行标准化
        image = image.resize((input_H, input_W))
        image = np.float32(image) / 255
        input_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)

        # 添加batch维度
        input_tensor = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()

        # 输入张量需要计算梯度
        input_tensor.requires_grad = True
        fmap_block = list()
        input_block = list()

        # 对指定层获取特征图
        layer.register_forward_hook(forward_hook)

        # 进行一次正向传播
        output = model(input_tensor)

        # 特征图的channel维度算平均值且去掉batch维度，得到二维张量
        feature_map = fmap_block[0].mean(dim=1, keepdim=False).squeeze()

        # 对二维张量中心点(标量)进行backward
        feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

        # 对输入层的梯度求绝对值
        grad = torch.abs(input_tensor.grad)

        # 梯度的channel维度算平均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
        grad = grad.mean(dim=1, keepdim=False).squeeze()

        # 累加所有图像的梯度，由于后面还要进行归一化，这里可以不用计算均值
        heatmap = heatmap + grad.cpu().numpy()

    cam = heatmap

    # 对累加的梯度进行归一化
    cam = cam / cam.max()

    # 可视化
    cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    Image.fromarray(cam).save("heatmap4_1.png")