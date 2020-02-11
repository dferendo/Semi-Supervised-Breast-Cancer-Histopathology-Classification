import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteLayer(nn.Module):
    def __init__(self, input_shape, reduction=16, use_bias=False):
        super(SqueezeExciteLayer, self).__init__()

        self.input_shape = input_shape
        self.reduction = reduction
        self.channels = self.input_shape[1]
        self.use_bias = use_bias
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        print("Building Squeeze_Excite_Layer with in and out shape %s and reduction factor %d" % (
        self.input_shape, self.reduction))

        x = torch.zeros(self.input_shape)

        self.layer_dict['se_global_avg_pool'] = nn.AdaptiveAvgPool2d(1)
        self.layer_dict['se_fc'] = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.reduction, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.reduction, self.channels, bias=self.use_bias),
            nn.Sigmoid()
        )

        w = self.layer_dict['se_global_avg_pool'] \
            .forward(x) \
            .view(self.input_shape[0], self.input_shape[1])
        w = self.layer_dict['se_fc'].forward(w) \
            .view(self.input_shape[0], self.input_shape[1], 1, 1) \
            .expand_as(x)

        out = torch.mul(x, w)

        return out

    def forward(self, x):
        w = self.layer_dict['se_global_avg_pool'] \
            .forward(x) \
            .view(self.input_shape[0], self.input_shape[1])
        w = self.layer_dict['se_fc'].forward(w) \
            .view(self.input_shape[0], self.input_shape[1], 1, 1) \
            .expand_as(x)

        out = torch.mul(x, w)

        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class SmallSeBlock(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias=True,
                 reduction=16, perform_downsampling=False, is_first_layer=False):
        super(SmallSeBlock, self).__init__()

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.use_bias = use_bias
        self.reduction = reduction
        self.perform_downsampling = perform_downsampling
        self.is_first_layer = is_first_layer
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print('Building Small_SE_Block with input shape %s' % (self.input_shape,))

        stride_red = 2 if self.perform_downsampling else 1

        x = torch.zeros(self.input_shape)
        out = x

        if not self.is_first_layer:
            self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
            self.layer_dict['bn_0'].forward(out)
            out = F.relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3),
                                              out_channels=self.num_filters, padding=(0, 1),
                                              bias=self.use_bias, stride=stride_red, dilation=1)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(3, 1), out_channels=out.shape[1],
                                              padding=(1, 0),
                                              bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_2'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3), out_channels=out.shape[1],
                                              padding=(0, 1),
                                              bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_3'].forward(out)

        self.layer_dict['bn_3'] = nn.BatchNorm2d(num_features=out.shape[1])
        self.layer_dict['bn_3'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_4'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(3, 1), out_channels=out.shape[1],
                                              padding=(1, 0),
                                              bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_4'].forward(out)

        self.layer_dict['se_1'] = SqueezeExciteLayer(out.shape, reduction=self.reduction, use_bias=False)
        out = self.layer_dict['se_1'].forward(out)

        if self.residual_conv_reshape_required(x, out):
            stride_width = int(round(x.shape[2] / out.shape[2]))
            stride_height = int(round(x.shape[3] / out.shape[3]))
            self.layer_dict['conv_resid_reduc'] = nn.Conv2d(in_channels=x.shape[1], kernel_size=(1, 1),
                                                            out_channels=out.shape[1],padding=0,
                                                            bias=self.use_bias, stride=(stride_height, stride_width), dilation=1)
            x = self.layer_dict['conv_resid_reduc'].forward(x)

        out = x + out

        return out

    def forward(self, x):
        out = x

        if not self.is_first_layer:
            self.layer_dict['bn_0'].forward(out)
            out = F.relu(out)

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

        if self.residual_conv_reshape_required(x, out):
            x = self.layer_dict['conv_resid_reduc'].forward(x)

        out = x + out

        return out

    def residual_conv_reshape_required(self, input, residual):
        input_shape = input.shape
        residual_shape = residual.shape
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        stride_height = int(round(input_shape[3] / residual_shape[3]))
        equal_channels = input_shape[1] == residual_shape[1]

        if stride_width > 1 or stride_height > 1 or not equal_channels:
            return True
        return False

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass


class InputConvolutionBlock(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias=True):
        super(InputConvolutionBlock, self).__init__()

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.use_bias = use_bias
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building First_Convolutional_Block in BHCNet with input shape %s" % (self.input_shape,))

        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=3, out_channels=self.num_filters
                                              , padding=1, bias=self.use_bias, stride=1, dilation=1)
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

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass
