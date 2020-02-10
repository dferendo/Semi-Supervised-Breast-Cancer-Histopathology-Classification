import torch
from src.bhcnet_modules import SqueezeExciteLayer, InputConvolutionBlock, SmallSeBlock

input_shape = (2,16,224,224)
x = torch.rand(input_shape)
# model = SqueezeExciteLayer(input_shape=input_shape, reduction=2)
# model = InputConvolutionBlock(input_shape, num_filters=5, use_bias=True)
model = SmallSeBlock(input_shape, None, 10, use_bias=True,
                     reduction=2, perform_downsampling=True, is_first_layer=False)
