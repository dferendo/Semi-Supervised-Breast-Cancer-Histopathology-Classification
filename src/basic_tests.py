import torch
from src.bhcnet_modules import Squeeze_Excite_Layer, Input_Convolution_Block, Small_SE_Block

input_shape = (1,16,224,224)
x = torch.rand(input_shape)
# model = Squeeze_Excite_Layer(input_shape=input_shape, reduction=2)
# model = Input_Convolution_Block(input_shape, num_filters=5, use_bias=True)
model = Small_SE_Block(input_shape, None, 10, use_bias=True,
                 reduction=2, is_first_block=True)
