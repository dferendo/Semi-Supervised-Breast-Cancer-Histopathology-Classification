import torch
import torch.nn as nn
import torch.nn.functional as F


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
            out = self.layer_dict['bn_0'].forward(out)
            out = F.relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3),
                                              out_channels=self.num_filters, padding=(0, 1),
                                              bias=self.use_bias, stride=stride_red, dilation=1)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(3, 1), out_channels=out.shape[1],
                                              padding=(1, 0),
                                              bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_2'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=out.shape[1], kernel_size=(1, 3), out_channels=out.shape[1],
                                              padding=(0, 1),
                                              bias=self.use_bias, stride=1, dilation=1)
        out = self.layer_dict['conv_3'].forward(out)

        self.layer_dict['bn_3'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = self.layer_dict['bn_3'].forward(out)
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

        out += x

        return out

    def forward(self, x):
        out = x

        if not self.is_first_layer:
            out = self.layer_dict['bn_0'].forward(out)
            out = F.relu(out)

        out = self.layer_dict['conv_1'].forward(out)

        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_2'].forward(out)

        out = self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_3'].forward(out)

        out = self.layer_dict['bn_3'].forward(out)
        out = F.relu(out)
        out = self.layer_dict['conv_4'].forward(out)

        out = self.layer_dict['se_1'].forward(out)

        if self.residual_conv_reshape_required(x, out):
            x = self.layer_dict['conv_resid_reduc'].forward(x)

        out = x + out

        return out