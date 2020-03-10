

class DenseNetParameters(object):
    growth_rate = 12
    bottleneck_factor = 4
    block_config = (6, 12, 24, 16)  # DenseNet-121
    drop_rate = 0.
    use_bias = True
    compression = 0.5

    def __init__(self, input_shape, num_init_features, num_classes):
        self.input_shape = input_shape
        self.num_init_features = num_init_features
        self.num_classes = num_classes
