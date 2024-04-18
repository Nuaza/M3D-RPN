import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from thop import profile
from collections import OrderedDict
from torchsummary import summary

from models.PConv import PConv
from models.RefConv import RefConv
from models.OREPA import OREPA
from models.DEConv import DEConv
from models.LocalConv2d import LocalConv2d


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, small_inputs=False, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


def replace_refconv(net):
    net.features.denseblock1.denselayer1.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock1.denselayer2.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock1.denselayer3.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock1.denselayer4.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock1.denselayer5.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock1.denselayer6.conv2 = RefConv(128, 32, stride=1, kernel_size=3)

    net.features.denseblock2.denselayer1.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer2.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer3.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer4.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer5.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer6.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer7.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer8.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer9.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer10.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer11.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock2.denselayer12.conv2 = RefConv(128, 32, stride=1, kernel_size=3)

    net.features.denseblock3.denselayer1.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer2.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer3.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer4.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer5.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer6.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer7.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer8.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer9.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer10.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer11.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer12.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer13.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer14.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer15.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer16.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer17.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer18.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer19.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer20.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer21.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer22.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer23.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock3.denselayer24.conv2 = RefConv(128, 32, stride=1, kernel_size=3)

    net.features.denseblock4.denselayer1.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer2.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer3.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer4.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer5.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer6.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer7.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer8.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer9.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer10.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer11.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer12.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer13.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer14.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer15.conv2 = RefConv(128, 32, stride=1, kernel_size=3)
    net.features.denseblock4.denselayer16.conv2 = RefConv(128, 32, stride=1, kernel_size=3)


def dilate_layers(net):
    dilate_layer(net.features.denseblock4.denselayer1.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer2.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer3.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer4.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer5.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer6.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer7.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer8.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer9.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer10.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer11.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer12.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer13.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer14.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer15.conv2, (2, 2))
    dilate_layer(net.features.denseblock4.denselayer16.conv2, (2, 2))


def change(net):
    dilate_layers(net)
    replace_refconv(net)

    net.features.CDF = nn.Sequential(
        RefConv(1024, 1024, stride=1, kernel_size=3)
    )
    net.features.prop_feats = nn.Sequential(
        # nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
        DEConv(dim=1024),
        nn.ReLU(inplace=True),
    )
    net.features.prop_feats_loc = nn.Sequential(
        # LocalConv2d(self.num_rows, self.base[-1].num_features, 512, 3, padding=1),
        DEConv(dim=1024),
        nn.BatchNorm2d(1024),
        nn.ReLU(inplace=True),
        RefConv(1024, 1024, stride=1, kernel_size=3),
        nn.ReLU(inplace=True),
    )

if __name__ == '__main__':
    net = DenseNet()
    change(net)
    summary(net, (3, 384, 1280))
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("==========================================================================================")
    # flops, params = profile(net, (torch.randn(1, 3, 384, 1280).cuda(),))
    flops, params = profile(net, (torch.randn(1, 3, 320, 320).cuda(),))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
