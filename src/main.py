import os

import data_providers as data_providers
from data_providers import DataParameters
from arg_extractor import get_shared_arguments
from util import get_processing_device
from model_architectures import BHCNetwork
from experiment_builder import ExperimentBuilder
from ExperimentBuilderMixMatch import ExperimentBuilderMixMatch

import numpy as np
from torchvision import transforms
from PIL import Image
import torch


def get_image_normalization(magnification):
    """
    Get the normalization mean and variance according to the magnification given
    :param magnification:
    :return:
    """
    if magnification == '40X':
        normalization_mean = (0.8035, 0.6524, 0.7726)
        normalization_var = (0.0782, 0.1073, 0.0765)
    elif magnification == '100X':
        normalization_mean = (0.7956, 0.6360, 0.7709)
        normalization_var = (0.0973, 0.1314, 0.0853)
    elif magnification == '200X':
        normalization_mean = (0.7886, 0.6228, 0.7681)
        normalization_var = (0.0996, 0.1317, 0.0782)
    elif magnification == '400X':
        normalization_mean = (0.7565, 0.5902, 0.7428)
        normalization_var = (0.1163, 0.1556, 0.0857)
    else:
        normalization_mean = (0.7868, 0.6263, 0.7642)
        normalization_var = (0.0974, 0.1310, 0.0814)

    return normalization_mean, normalization_var


def get_transformations(normalization_mean, normalization_var):
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    return transformations, transformations_test


def get_unlabeled_transformations(normalization_mean, normalization_var):
    transformations_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    transformations_2 = transforms.Compose([
        transforms.RandomCrop(0.1),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    return [transformations_1, transformations_2]


args = get_shared_arguments()
device = get_processing_device()

# Seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# Data Loading
normalization_mean, normalization_var = get_image_normalization(args.magnification)
transformations, transformations_test = get_transformations(normalization_mean, normalization_var)
unlabeled_transformations = get_unlabeled_transformations(normalization_mean, normalization_var)

data_location = os.path.abspath(args.dataset_location)

data_parameters = DataParameters(data_location, args.batch_size, transformations, transformations_test)

data_parameters.magnification = args.magnification
data_parameters.unlabeled_split = args.unlabelled_split
data_parameters.unlabeled_transformations = unlabeled_transformations

train_loader, train_unlabeled_loader, val_loader, test_loader = data_providers.get_datasets(data_parameters)

# Build the BHC Network
bch_network = BHCNetwork(input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
                         num_filters=args.num_filters, num_layers=args.num_layers,
                         use_bias=True,
                         num_output_classes=2)

# Parameters for BCH Network
optimizer_params = {'weight_decay': args.weight_decay_coefficient,
                    'momentum': args.momentum,
                    'nesterov': args.nesterov}
scheduler_params = {'lr_max': args.learn_rate_max,
                    'lr_min': args.learn_rate_min,
                    'erf_alpha': args.erf_sched_alpha,
                    'erf_beta': args.erf_sched_beta}

if not args.use_mix_match:
    bhc_experiment = ExperimentBuilder(network_model=bch_network,
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
else:
    bhc_experiment = ExperimentBuilderMixMatch(network_model=bch_network,
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
                                               lambda_u=0.5)

experiment_metrics, test_metrics = bhc_experiment.run_experiment()
