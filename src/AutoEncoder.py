from densenet import DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderTransition(nn.Sequential):
    def __init__(self, input_shape, use_bias, num_output_filters, last_block):
        super(DecoderTransition, self).__init__()

        self.num_output_filters = num_output_filters
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.last_block = last_block
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        print('Transition Layer shape', self.input_shape)
        x = torch.zeros(self.input_shape)
        out = x

        if not self.last_block:
            self.layer_dict['up1'] = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
            out = self.layer_dict['up1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_output_filters,
                                              kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_2'].forward(out)
        out = F.relu(out)

        return out

    def forward(self, inputs):
        out = inputs

        if not self.last_block:
            out = self.layer_dict['up1'].forward(out)

        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['conv_1'].forward(out)

        out = self.layer_dict['bn_2'].forward(out)
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


class DecoderLayer(nn.Module):
    def __init__(self, input_shape, use_bias=True):
        super(DecoderLayer, self).__init__()
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        # Assuming input shape is the pre-concatenated tensor shape
        # num_input_features should be dim 1 of the 4d tensor
        print('Dense Layer shape', self.input_shape)
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1],
                                              out_channels=out.shape[1],
                                              kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        out = self.layer_dict['conv_1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)

        out = F.relu(out)

        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=out.shape[1],
                                              out_channels=out.shape[1],
                                              kernel_size=3, stride=1, padding=1, bias=self.use_bias)
        out = self.layer_dict['conv_2'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_2'].forward(out)

        out = out + x

        out = F.relu(out)

        return out

    def forward(self, x):
        # concatenated features
        out = x
        out = self.layer_dict['conv_1'].forward(out)
        out = self.layer_dict['bn_1'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['conv_2'].forward(out)
        out = self.layer_dict['bn_2'].forward(out)
        out = out + x
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


class DecoderDenseBlock(nn.Module):
    def __init__(self, input_shape, number_of_layers=3, use_bias=True):
        super(DecoderDenseBlock, self).__init__()
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.number_of_layers = number_of_layers
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        # Assuming input shape is the pre-concatenated tensor shape
        # num_input_features should be dim 1 of the 4d tensor
        print('Dense Block shape', self.input_shape)
        x = torch.zeros(self.input_shape)
        out = x

        for layer in range(0, self.number_of_layers):
            self.layer_dict[f'res_block_{layer}'] = DecoderLayer(self.input_shape, self.use_bias)
            out = self.layer_dict[f'res_block_{layer}'].forward(out)

        return out

    def forward(self, x):
        out = x

        for layer in range(0, self.number_of_layers):
            out = self.layer_dict[f'res_block_{layer}'].forward(out)

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


class AutoDecoder(nn.Module):
    def __init__(self, block_config, use_bias, input_shape, growth_rate, compression, num_of_layers):
        super(AutoDecoder, self).__init__()
        self.block_config = block_config
        self.use_bias = use_bias
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.compression = compression
        self.num_of_layers = num_of_layers

        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        out = torch.zeros(self.input_shape)

        self.layer_dict['entry_bn'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['entry_bn'].forward(out)

        for i, block_layers in enumerate(self.block_config[::-1]):

            self.layer_dict[f'block_{i}'] = DecoderDenseBlock(input_shape=out.shape, number_of_layers=self.num_of_layers,
                                                              use_bias=self.use_bias)
            out = self.layer_dict[f'block_{i}'].forward(out)

            self.layer_dict[f't_block_{i}'] = DecoderTransition(input_shape=out.shape, use_bias=self.use_bias,
                                                                num_output_filters=math.ceil((out.shape[1] * self.compression)),
                                                                last_block=i == len(self.block_config) - 1)
            out = self.layer_dict[f't_block_{i}'].forward(out)

        return out

    def forward(self, input):
        out = input
        out = self.layer_dict['entry_bn'].forward(out)

        for i, block_layers in enumerate(self.block_config[::-1]):
            out = self.layer_dict[f'block_{i}'].forward(out)
            out = self.layer_dict[f't_block_{i}'].forward(out)

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


class Autoencoder(nn.Module):
    def __init__(self, densenetParameters):
        super(Autoencoder, self).__init__()

        self.densenetParameters = densenetParameters
        self.layer_dict = nn.ModuleDict()

        self.build_module()

    def build_module(self):
        x = torch.zeros(self.densenetParameters.input_shape)
        out = x

        self.layer_dict['encoder'] = DenseNet(input_shape=self.densenetParameters.input_shape,
                                              growth_rate=self.densenetParameters.growth_rate,
                                              block_config=self.densenetParameters.block_config,
                                              compression=self.densenetParameters.compression,
                                              num_init_features=self.densenetParameters.num_init_features,
                                              bottleneck_factor=self.densenetParameters.bottleneck_factor,
                                              drop_rate=self.densenetParameters.drop_rate,
                                              num_classes=self.densenetParameters.num_classes,
                                              no_classification=True,
                                              use_se=True,
                                              increasing_dilation=True,
                                              small_inputs=False)

        out = self.layer_dict['encoder'].forward(out)

        self.layer_dict['decoder'] = AutoDecoder(block_config=self.densenetParameters.block_config,
                                                 use_bias=self.densenetParameters.use_bias,
                                                 input_shape=out.shape,
                                                 growth_rate=self.densenetParameters.growth_rate,
                                                 compression=self.densenetParameters.compression,
                                                 num_of_layers=2)

        out = self.layer_dict['decoder'].forward(out)

        self.layer_dict['conv_final_a'] = nn.ConvTranspose2d(out.shape[1], out_channels=3, kernel_size=7,
                                              stride=4, padding=3, output_padding=3, bias=self.densenetParameters.use_bias)

        out = self.layer_dict['conv_final_a'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)

        self.layer_dict['conv_final_b'] = nn.ConvTranspose2d(out.shape[1],out_channels=3, kernel_size=7,
                                              stride=1, padding=3, bias=self.densenetParameters.use_bias)

        out = self.layer_dict['conv_final_b'].forward(out)

        self.layer_dict['bn_2'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_2'].forward(out)

        return out

    def forward(self, x):
        out = x
        out = self.layer_dict['encoder'].forward(out)
        out = self.layer_dict['decoder'].forward(out)

        out = self.layer_dict['conv_final_a'].forward(out)
        out = self.layer_dict['conv_final_b'].forward(out)

        out = self.layer_dict['bn_1'].forward(out)
        out = self.layer_dict['bn_2'].forward(out)

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