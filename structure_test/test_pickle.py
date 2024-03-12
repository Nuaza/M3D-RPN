import torch
from torchvision import transforms

if __name__ == '__main__':
    # 取出训练好的权重文件，这是一个有序字典
    with open('../output/kitti_3d_multi_main/weights/model_30000_pkl', 'rb') as file:
        data = torch.load(file)
    # keys()查看它里面的所有键
    print(data.keys())
    print(len(data))
    # print(data['module.bbox_x3d_ble'])
    tensor = data['module.base.denseblock3.denselayer2.norm1.weight']
    print(tensor)
    unloader = transforms.ToPILImage(mode='L')
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.reshape(-1, 16)
    image = unloader(image)
    image.save('tensor.png')
    # print(data['module.base.denseblock3.denselayer2.norm1.bias'])