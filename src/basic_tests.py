import torch
from src.bhcnet_modules import Squeeze_Excite_Layer, Input_Convolution_Block

input_shape = (3,2,4,4)
x = torch.rand(input_shape)
model = Squeeze_Excite_Layer(input_shape=input_shape, reduction=2)
model = Input_Convolution_Block(input_shape, num_filters=5, use_bias=True)

