from densenet import DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecoderLayer(nn.Module):
    def __init__(self, input_shape):
        super(DecoderLayer, self).__init__()

        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()

        self.build_module()

    def build_module(self):
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['deconv_1'] = nn.ConvTranspose2d(in_channels=out.shape[1],
                                              out_channels=out.shape[1],
                                              kernel_size=1, stride=1, bias=True)
        out = self.layer_dict['deconv_1'].forward(out)

        self.layer_dict['bn_1'] = nn.BatchNorm2d(out.shape[1])
        out = self.layer_dict['bn_1'].forward(out)

        out = F.relu(out)

        self.layer_dict['deconv_2'] = nn.ConvTranspose2d(in_channels=out.shape[1],
                                              out_channels=3,
                                              kernel_size=3, stride=8, padding=1,output_padding=7, bias=True)
        out = self.layer_dict['deconv_2'].forward(out)

    def forward(self, input):
        out = input

        out = self.layer_dict['deconv_1'].forward(out)
        out = self.layer_dict['bn_1'].forward(out)

        out = F.relu(out)

        out = self.layer_dict['deconv_2'].forward(out)

        return out


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
                                              small_inputs=False)

        out = self.layer_dict['encoder'].forward(out)

        self.layer_dict['dtest'] = DecoderLayer(out.shape)
        out = self.layer_dict['dtest'].forward(out)
        print(out.shape)
        self.layer_dict['deconv_out'] = nn.ConvTranspose2d(out.shape[1], 3, kernel_size=7,
                                              stride=4, dilation=1,padding=3, output_padding=3, bias=True)
        out = self.layer_dict['deconv_out'].forward(out)
        print(out.shape)

        # out = torch.flatten(out, 1)

        # out = self.layer_dict['final_classifier'].forward(out)
        # out = out.view(self.densenetParameters.input_shape)
        #bottleneck?
        # out = F.relu(out)
        # out = F.adaptive_avg_pool2d(out, (12, 12

        print(out.shape)

    def forward(self, x):
        out = x
        out = self.layer_dict['encoder'].forward(out)
        out = self.layer_dict['dtest'].forward(out)

        out = self.layer_dict['deconv_out'].forward(out)

        # out = torch.flatten(out, 1)
        # out = self.layer_dict['final_classifier'].forward(out)
        # out = out.view(self.densenetParameters.input_shape)
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


