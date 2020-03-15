import argparse

'''
Argument extractor. This is a modified version of the files found in https://github.com/CSTR-Edinburgh/mlpractical.
'''


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def strToTuple(v):
    return tuple([int(x) for x in v.split(',')])


def strToArray(v):
    return [float(x) for x in v.split(',')]


def get_shared_arguments():
    """
        Returns a namedtuple with arguments extracted from the command line.
        :return: A namedtuple with arguments
        """
    parser = argparse.ArgumentParser(description='Semi-supervised Breast Cancer Classification')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--dataset_name', type=str, help='Dataset on which the system will train/eval our model')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=1,
                        help='The channel dimensionality of our image-data')
    parser.add_argument('--image_height', nargs="?", type=int, default=28, help='Height of image data')
    parser.add_argument('--image_width', nargs="?", type=int, default=28, help='Width of image data')
    parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
                        help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling]')
    parser.add_argument('--num_layers', nargs="?", type=int, default=4,
                        help='Number of convolutional layers in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--initial_num_filters', nargs="?", type=int, default=64,
                        help='Number of convolutional filters per convolutional layer in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
                        help='')
    parser.add_argument('--learn_rate_max', nargs="?", type=float, default=0.0001,
                        help='Starting learning rate. Max learning rate in a lr scheduler')
    parser.add_argument('--learn_rate_min', nargs="?", type=float, default=0.00001,
                        help='Min learning rate in a lr scheduler')
    parser.add_argument('--optim_type', nargs="?", type=str, default='Adam',
                        help='Which optimiser to use (Adam, SGD)')
    parser.add_argument('--momentum', nargs="?", type=float, default=0.9,
                        help='Which optimiser to use (Adam, SGD)')
    parser.add_argument('--nesterov', nargs="?", type=str2bool, default=True,
                        help='Whether to use Nesterov Momentum')
    parser.add_argument('--sched_type', nargs="?", type=str, default=None,
                        help='Which learn rate scheduler to use (ERF)')
    parser.add_argument('--erf_sched_alpha', nargs="?", type=int, default=None,
                        help='ERF alpha hyperparam')
    parser.add_argument('--erf_sched_beta', nargs="?", type=int, default=None,
                        help='ERF beta hyperparam')
    parser.add_argument('--magnification', nargs="?", type=str, default=None,
                        help='The type of magnification to consider (40X, 100X, 200X, 400X)')
    parser.add_argument('--dataset_location', nargs="?", type=str, default=None,
                        help='The location of the dataset')
    parser.add_argument('--unlabelled_split', nargs="?", type=float, default=None,
                        help='The amount of the training set to be set as unlabelled (0 to 1)')
    parser.add_argument('--labelled_images_amount', nargs="?", type=int, default=None,
                        help='The amount of labelled images per subclass in the training set')
    parser.add_argument('--use_mix_match', nargs="?", type=str2bool, default=False,
                        help='Whether to use MixMatch or not')
    parser.add_argument('--use_fix_match', nargs="?", type=str2bool, default=False,
                        help='Whether to use FixMatch or not')
    parser.add_argument('--multi_class', nargs="?", type=str2bool, default=False,
                        help='Whether to use Multi class or Binary class')
    parser.add_argument('--drop_rate', nargs="?", type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_se', nargs="?", type=str2bool, default=False,
                        help='Whether to include Squeeze Excite')
    parser.add_argument('--se_reduction', nargs="?", type=int, default=16,
                        help='Squeeze Excitation reduction')
    parser.add_argument('--loss_lambda_u', nargs="?", type=int, default=1,
                        help='Mixmatch lambda_u unlabelled loss')
    parser.add_argument('--n_raug', nargs="?", type=int, default=None,
                        help='N randaugment parameter')
    parser.add_argument('--m_raug', nargs="?", type=int, default=None,
                        help='M randaugment parameter')
    parser.add_argument('--unlabelled_factor', nargs="?", type=int, default=1,
                        help='Factor of batch size of unlabelled data')
    parser.add_argument('--fm_conf_threshold', nargs="?", type=float, default=0.95,
                        help='The fixmatch threshold')
    parser.add_argument('--pretrained_weights_locations', nargs="?", type=str, default=None,
                        help='Pre-trained weights locations for Densenet')
    parser.add_argument('--block_config', nargs="?", type=strToTuple, default=None,
                        help='Block config for Densenet')
    parser.add_argument('--growth_rate', nargs="?", type=int, default=None,
                        help='Densenet Growth Rate')
    parser.add_argument('--compression', nargs="?", type=float, default=None,
                        help='Densenet compression')
    parser.add_argument('--bottleneck_factor', nargs="?", type=int, default=None,
                        help='Densenet Bottleneck')
    parser.add_argument('--transformation_labeled_parameters', nargs="?", type=strToArray, default=None,
                        help='Which transformation to use and and the parameters of the transformation')
    parser.add_argument('--transformation_unlabeled_parameters', nargs="?", type=strToArray, default=None,
                        help='Which transformation to use and and the parameters of the transformation')
    parser.add_argument('--transformation_unlabeled_strong_parameters', nargs="?", type=strToArray, default=None,
                        help='Which transformation to use and and the parameters of the transformation')
    parser.add_argument('--fine_tune', nargs="?", type=str2bool, default=None,
                        help='Perform fine-tuning on last block and fc')

    args = parser.parse_args()
    print('Printing arguments: ', [(str(key), str(value)) for (key, value) in vars(args).items()])

    return args
