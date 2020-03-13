

class DenseNetParameters(object):
    drop_rate = 0.
    use_bias = True

    def __init__(self, input_shape, num_init_features, num_classes, block_config, growth_rate,
                 bottleneck_factor, compression, use_se):
        self.input_shape = input_shape
        self.num_init_features = num_init_features
        self.num_classes = num_classes
        self.block_config = block_config
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.compression = compression
        self.use_se = use_se
