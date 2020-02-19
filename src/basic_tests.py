import torch
from src.bhcnet_modules import SqueezeExciteLayer, InputConvolutionBlock, SmallSeBlock
from src.model_architectures import BHCNetwork


# input_shape = (100,3,224,224)
# x = torch.rand(input_shape)
# model = BHCNetwork(input_shape, 2, 16, 2, use_bias=True)
# model = SqueezeExciteLayer(input_shape=input_shape, reduction=2)
# model = InputConvolutionBlock(input_shape, num_filters=5, use_bias=True)
# model = SmallSeBlock(input_shape, None, 10, use_bias=True,
#                      reduction=2, perform_downsampling=True, is_first_layer=False)

def sharpen(p, T):
    pt = p ** (1 / T)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    return targets_u


p_s = (1,5)
x = torch.rand(p_s)
x = x / x.sum(dim=1, keepdim=True)
print(x)
print(sharpen(x, 0.1))
print(sharpen(x, 2))
print(sharpen(x, 10))

print(x)


