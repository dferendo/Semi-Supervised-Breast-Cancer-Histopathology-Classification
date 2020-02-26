# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

# TODO: Delete if we do not use memory efficient version
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    def __init__(self, input_shape, growth_rate, bn_size, drop_rate, efficient=False,
                 use_bias=True):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.use_bias = use_bias
        self.growth_rate = growth_rate
        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()
        self.bottleneck_factor = bn_size # bottleneck size

        # build the network
        self.build_module()

    def build_module(self):
        # Assuming input shape is the pre-concatenated tensor shape
        # num_input_features should be dim 1 of the 4d tensor
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['bn_1'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1],
                                              out_channels=self.bottleneck_factor * self.growth_rate,
                                              kernel_size=1, stride=1, bias=self.use_bias)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=out.shape[1],
                                              out_channels=self.growth_rate,
                                              kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        out = self.layer_dict['conv_2'].forward(out)

        # TODO: Checkout dropout 2d
        if self.drop_rate > 0:
            self.layer_dict['dropout'] = nn.Dropout(p=self.drop_rate)
            out = self.layer_dict['dropout'](out)

        return out

    def forward(self, *prev_features):
        # concatenated features
        out = torch.cat(prev_features, 1)

        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_1'].forward(out)

        out = self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_2'].forward(out)

        if self.drop_rate > 0:
            out = self.layer_dict['dropout'](out)

        return out

    def forward_original(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


# TODO: Turn this into mlp style
class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseBlock(nn.Module):
    def __init__(self, input_shape, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, use_bias=True):
        super(DenseBlock, self).__init__()

        self.drop_rate = drop_rate
        self.efficient = efficient
        self.use_bias = use_bias
        self.growth_rate = growth_rate
        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()
        self.bottleneck_factor = bn_size # bottleneck size
        self.num_layers = num_layers

        # build the network
        self.build_module()

    def build_module(self):
        x = torch.zeros(self.input_shape)
        out = x

        # self.num_input_features + i * self.growth_rate,
        for i in range(self.num_layers):
            self.layer_dict['dense_block_%d' % (i + 1)] = DenseLayer(
                input_shape= out.shape,
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                drop_rate=self.drop_rate,
                efficient=self.efficient,
            )
            out = self.layer_dict['dense_block_%d' % (i + 1)].forward(out)

        return out

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layer_dict.items():
            new_features = layer.forward(*features)
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

    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

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
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

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
