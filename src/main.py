import os

import data_providers as data_providers
from data_providers import DataParameters
from arg_extractor import get_shared_arguments
from util import get_processing_device
from experiment_builder import ExperimentBuilder
from ExperimentBuilderMixMatch import ExperimentBuilderMixMatch
from ExperimentBuilderFixMatch import ExperimentBuilderFixMatch
import random

import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from RandAugment import RandAugment

from densenet import DenseNet


def get_image_normalization(magnification):
    """
    Get the normalization mean and variance according to the magnification given
    :param magnification:
    :return:
    """
    if magnification == '40X':
        normalization_mean = (0.7966, 0.6515, 0.7687)
        normalization_var = (0.0773, 0.1053, 0.0742)
    elif magnification == '100X':
        normalization_mean = (0.7869, 0.6278, 0.7647)
        normalization_var = (0.0981, 0.1291, 0.0835)
    elif magnification == '200X':
        normalization_mean = (0.7842, 0.6172, 0.7655)
        normalization_var = (0.1035, 0.1331, 0.0790)
    elif magnification == '400X':
        normalization_mean = (0.7516, 0.5758, 0.7414)
        normalization_var = (0.1192, 0.1552, 0.0849)
    else:
        normalization_mean = (0.7818, 0.6171, 0.7621)
        normalization_var = (0.0996, 0.1314, 0.0812)

    return normalization_mean, normalization_var


def get_transformations(normalization_mean, normalization_var, image_height, image_width, transformations_labeled):
    transformations = transforms.Compose([])

    if transformations_labeled is not None:

        while len(transformations_labeled) != 0:
            transformation_to_choice = transformations_labeled.pop(0)

            if transformation_to_choice == 0:
                prob_parameter = transformations_labeled.pop(0)
                print('Transformation Added: Horizontal Flip ', prob_parameter)

                transformations.transforms.append(transforms.RandomHorizontalFlip(prob_parameter))
            elif transformation_to_choice == 1:
                prob_parameter = transformations_labeled.pop(0)
                print('Transformation Added: Vertical Flip ', prob_parameter)

                transformations.transforms.append(transforms.RandomVerticalFlip(prob_parameter))

    transformations.transforms.append(transforms.Resize((image_height, image_width), interpolation=Image.BILINEAR))
    transformations.transforms.append(transforms.ToTensor())
    transformations.transforms.append(transforms.Normalize(normalization_mean, normalization_var))

    transformations_test = transforms.Compose([
        transforms.Resize((image_height, image_width), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    return transformations, transformations_test


def get_unlabeled_transformations(normalization_mean, normalization_var, image_height, image_width, m_raug=None, n_raug=None,
                                  transformations_unlabeled=None, transformation_unlabeled_strong=None):
    transformations_1 = transforms.Compose([])

    if transformations_unlabeled is not None:

        while len(transformations_unlabeled) != 0:
            transformation_to_choice = transformations_unlabeled.pop(0)

            if transformation_to_choice == 0:
                prob_parameter = transformations_unlabeled.pop(0)
                print('Unlabeled Transformation Added: Horizontal Flip ', prob_parameter)

                transformations_1.transforms.append(transforms.RandomHorizontalFlip(prob_parameter))
            elif transformation_to_choice == 1:
                prob_parameter = transformations_unlabeled.pop(0)
                print('Unlabeled Transformation Added: Vertical Flip ', prob_parameter)

                transformations_1.transforms.append(transforms.RandomVerticalFlip(prob_parameter))

    transformations_1.transforms.append(transforms.Resize((image_height, image_width), interpolation=Image.BILINEAR))
    transformations_1.transforms.append(transforms.ToTensor())
    transformations_1.transforms.append(transforms.Normalize(normalization_mean, normalization_var))

    transformations_2 = transforms.Compose([])

    if transformation_unlabeled_strong is not None:

        while len(transformation_unlabeled_strong) != 0:
            transformation_to_choice = transformation_unlabeled_strong.pop(0)

            if transformation_to_choice == 0:
                prob_parameter = transformation_unlabeled_strong.pop(0)
                print('Unlabeled Strong Transformation Added: Horizontal Flip ', prob_parameter)

                transformations_2.transforms.append(transforms.RandomHorizontalFlip(prob_parameter))
            elif transformation_to_choice == 1:
                prob_parameter = transformation_unlabeled_strong.pop(0)
                print('Unlabeled Strong Transformation Added: Vertical Flip ', prob_parameter)

                transformations_2.transforms.append(transforms.RandomVerticalFlip(prob_parameter))

    transformations_2.transforms.append(transforms.Resize((image_height, image_width), interpolation=Image.BILINEAR))
    transformations_2.transforms.append(transforms.ToTensor())
    transformations_2.transforms.append(transforms.Normalize(normalization_mean, normalization_var))

    if n_raug is not None and m_raug is not None:
        print('-Doing RandAugment-')
        transformations_2.transforms.insert(1, RandAugment(n=n_raug, m=m_raug))

    return [transformations_1, transformations_2]


args = get_shared_arguments()
device = get_processing_device()

# Seeds
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.manual_seed(args.seed)


# Data Loading
normalization_mean, normalization_var = get_image_normalization(args.magnification)
transformations, transformations_test = get_transformations(normalization_mean, normalization_var, args.image_height,
                                                            args.image_height, args.transformation_labeled_parameters)
unlabeled_transformations = get_unlabeled_transformations(normalization_mean, normalization_var, args.image_height,
                                                          args.image_height, m_raug=args.m_raug, n_raug=args.n_raug,
                                                          transformations_unlabeled=args.transformation_unlabeled_parameters,
                                                          transformation_unlabeled_strong=args.transformation_unlabeled_strong_parameters)

data_location = os.path.abspath(args.dataset_location)

data_parameters = DataParameters(data_location, args.batch_size, transformations, transformations_test,
                                 args.multi_class)

data_parameters.magnification = args.magnification
data_parameters.unlabeled_split = args.unlabelled_split
data_parameters.labelled_images_amount = args.labelled_images_amount
data_parameters.unlabeled_transformations = unlabeled_transformations
data_parameters.unlabelled_factor = args.unlabelled_factor
data_parameters.seed = args.seed

train_loader, train_unlabeled_loader, val_loader, test_loader = data_providers.get_datasets(data_parameters)

if args.multi_class:
    print('Multi-class')
    num_output_classes = 8
else:
    print('Binary-class')
    num_output_classes = 2

model = DenseNet(input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
                 growth_rate=args.growth_rate, block_config=args.block_config,
                 compression=args.compression,
                 num_init_features=args.initial_num_filters,
                 bottleneck_factor=args.bottleneck_factor,
                 drop_rate=args.drop_rate,
                 num_classes=num_output_classes, small_inputs=False, efficient=False,
                 use_bias=True,
                 use_se=args.use_se, se_reduction=args.se_reduction, increasing_dilation=True)
#
# from torchvision import models
# import torch.nn as nn

# model = models.densenet121(pretrained=True, memory_efficient=True)

# model.classifier = nn.Linear(in_features=model.classifier.in_features,
#                              out_features=num_output_classes,
#                              bias=True)

# model = temp.DenseNet(growth_rate=12, block_config=(6, 12, 24, 16), compression=0.5,
#                      num_init_features=args.num_filters, bn_size=4, drop_rate=0,
#                      num_classes=num_output_classes, small_inputs=False, efficient=False)

# # Build the BHC Network
# bch_network = BHCNetwork(input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
#                          num_filters=args.num_filters, num_layers=args.num_layers,
#                          use_bias=True,
#                          num_output_classes=num_output_classes)
#
# # Parameters for BCH Network
optimizer_params = {'weight_decay': args.weight_decay_coefficient,
                    'momentum': args.momentum,
                    'nesterov': args.nesterov}
scheduler_params = {'lr_max': args.learn_rate_max,
                    'lr_min': args.learn_rate_min,
                    'erf_alpha': args.erf_sched_alpha,
                    'erf_beta': args.erf_sched_beta}

if not args.use_mix_match and not args.use_fix_match:
    print('No Mix Match')
    bhc_experiment = ExperimentBuilder(network_model=model,
                                       use_gpu=args.use_gpu,
                                       experiment_name=args.experiment_name,
                                       num_epochs=args.num_epochs,
                                       continue_from_epoch=args.continue_from_epoch,
                                       train_data=train_loader,
                                       val_data=val_loader,
                                       test_data=test_loader,
                                       optimiser=args.optim_type,
                                       optim_params=optimizer_params,
                                       scheduler=args.sched_type,
                                       sched_params=scheduler_params)
elif args.use_mix_match:
    print('Mix Match')
    bhc_experiment = ExperimentBuilderMixMatch(network_model=model,
                                               use_gpu=args.use_gpu,
                                               experiment_name=args.experiment_name,
                                               num_epochs=args.num_epochs,
                                               continue_from_epoch=args.continue_from_epoch,
                                               train_data=train_loader,
                                               val_data=val_loader,
                                               test_data=test_loader,
                                               optimiser=args.optim_type,
                                               optim_params=optimizer_params,
                                               scheduler=args.sched_type,
                                               sched_params=scheduler_params,
                                               train_data_unlabeled=train_unlabeled_loader,
                                               lambda_u=args.loss_lambda_u)
elif args.use_fix_match:
    print('Fix Match')
    bhc_experiment = ExperimentBuilderFixMatch(network_model=model,
                                               use_gpu=args.use_gpu,
                                               experiment_name=args.experiment_name,
                                               num_epochs=args.num_epochs,
                                               continue_from_epoch=args.continue_from_epoch,
                                               train_data=train_loader,
                                               val_data=val_loader,
                                               test_data=test_loader,
                                               optimiser=args.optim_type,
                                               optim_params=optimizer_params,
                                               scheduler=args.sched_type,
                                               sched_params=scheduler_params,
                                               train_data_unlabeled=train_unlabeled_loader,
                                               lambda_u=args.loss_lambda_u,
                                               threshold=args.fm_conf_threshold,
                                               pretrained_weights_locations=args.pretrained_weights_locations)

experiment_metrics, test_metrics = bhc_experiment.run_experiment()
