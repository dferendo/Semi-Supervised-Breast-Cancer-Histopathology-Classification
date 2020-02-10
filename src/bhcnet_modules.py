import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze_Excite_Layer(nn.Module):
    def __init__(self, input_shape, reduction=16, use_bias=False):
        super(Squeeze_Excite_Layer, self).__init__()

        self.input_shape = input_shape
        self.reduction = reduction
        self.channels = self.input_shape[1]
        self.use_bias = use_bias
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        print("Building Squeeze_Excite_Layer with in and out shape %s and reduction factor %d" % (self.input_shape, self.reduction))

        x = torch.zeros((self.input_shape))

        self.layer_dict['se_global_avg_pool'] = nn.AdaptiveAvgPool2d(1)
        self.layer_dict['se_fc'] = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.reduction, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.reduction, self.channels, bias=self.use_bias),
            nn.Sigmoid()
        )

        w = self.layer_dict['se_global_avg_pool']\
            .forward(x)\
            .view(self.input_shape[0], self.input_shape[1])
        w = self.layer_dict['se_fc'].forward(w)\
            .view(self.input_shape[0], self.input_shape[1], 1, 1)\
            .expand_as(x)

        out = torch.mul(x, w)

        return out

    def forward(self, x):
        w = self.layer_dict['se_global_avg_pool']\
            .forward(x)\
            .view(self.input_shape[0], self.input_shape[1])
        w = self.layer_dict['se_fc'].forward(w)\
            .view(self.input_shape[0], self.input_shape[1], 1, 1)\
            .expand_as(x)

        out = torch.mul(x, w)

        return out


class Small_SE_Block(nn.Module):
    # TODO: Fix dimensionality
    def __init__(self, input_shape, dim_reduction_type, num_filters, use_bias=True,
                 reduction=16, is_first_block=False):
        super(Small_SE_Block, self).__init__()

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.use_bias = use_bias
        self.dim_reduction_type = dim_reduction_type
        self.reduction = reduction
        self.is_first_block = is_first_block
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print('Building Small_SE_Block with input shape %s reduction factor %d' % (self.input_shape, self.reduction))

        stride_red = 2 if self.is_first_block else 1

        x = torch.zeros((self.input_shape))
        out = x

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3), out_channels=self.num_filters, padding=(0,1),
                          bias=self.use_bias, stride=stride_red, dilation=1)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(3, 1), out_channels=out.shape[1], padding=(1,0),
                          bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_2'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3), out_channels=out.shape[1], padding=(0,1),
                          bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_3'].forward(out)

        self.layer_dict['bn_3'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_3'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_4'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(3, 1), out_channels=out.shape[1], padding=(1,0),
                          bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_4'].forward(out)

        self.layer_dict['se_1'] = Squeeze_Excite_Layer(out.shape, reduction=self.reduction, use_bias=False)
        out = self.layer_dict['se_1'].forward(out)

        return out

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_2'].forward(out)

        self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_3'].forward(out)

        self.layer_dict['bn_3'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_4'].forward(out)

        out = self.layer_dict['se_1'].forward(out)

        return out

class Input_Convolution_Block(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias=True):
        super(Input_Convolution_Block, self).__init__()

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.use_bias = use_bias
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building First_Convolutional_Block in BHCNet with input shape %s" % (self.input_shape,))

        x = torch.zeros((self.input_shape))
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=3, out_channels=self.num_filters, padding=1,
                          bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_0'].forward(out)

        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_0'].forward(out)

        out = F.relu(out)

        return out

    def forward(self, x):
        out = x
        out = self.layer_dict['conv_0'].forward(out)
        out = self.layer_dict['bn_0'].forward(out)
        out = F.relu(out)
        return out