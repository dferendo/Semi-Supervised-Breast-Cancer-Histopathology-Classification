import torch
from src.bhcnet_modules import SqueezeExciteLayer, InputConvolutionBlock, SmallSeBlock
from src.model_architectures import BHCNetwork
import numpy as np


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


p_s = (2,1,2,2)
x1 = torch.rand(p_s)

l = np.random.beta(0.8, 0.9, size=x1.shape[0])
# l = max(l, 1 - l)
l = np.maximum(l, 1-l)
l = torch.from_numpy(l)

print(l.shape, l)
print(x1.shape, x1)

l = l.view((l.shape[0], 1, 1, 1))

f = torch.mul(l,x1)

print(f)
# x = x1 / x1.sum(dim=1, keepdim=True)
# print(x1)
# print(x)
# print(torch.softmax(x1, dim=1))
# print(sharpen(x, 0.1))
# print(sharpen(x, 2))
# print(sharpen(x, 10))
#
# print(x)


