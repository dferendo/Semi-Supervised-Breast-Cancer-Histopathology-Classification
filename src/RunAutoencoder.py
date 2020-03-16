import os

from arg_extractor import get_shared_arguments
from util import get_processing_device

import numpy as np
from torchvision import transforms
from PIL import Image
import torch

from DenseNetParameters import DenseNetParameters
from AutoEncoder import Autoencoder
from ExperimentBuilderAE import ExperimentBuilder
from DataLoading import get_datasets, DataParameters
import matplotlib.pyplot as plt


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


def get_transformations(normalization_mean, normalization_var, image_height, image_width):
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((image_height, image_width), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_var)
    ])

    noising = None
    # noising = transforms.Compose([
    #     transforms.RandomErasing(1, scale=(0.02, 0.20))
    # ])

    return transformations, noising


args = get_shared_arguments()
device = get_processing_device()

# Seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# Data Loading
normalization_mean, normalization_var = get_image_normalization(args.magnification)
transformations, noising = get_transformations(normalization_mean, normalization_var, args.image_height,
                                               args.image_height)

data_location = os.path.abspath(args.dataset_location)

data_parameters = DataParameters(data_location, args.batch_size, transformations, None, args.multi_class)

data_parameters.magnification = args.magnification
data_parameters.unlabeled_split = args.unlabelled_split
data_parameters.labelled_images_amount = args.labelled_images_amount

train_loader = get_datasets(data_parameters, noising)

if args.multi_class:
    print('Multi-class')
    num_output_classes = 8
else:
    print('Binary-class')
    num_output_classes = 2

densenetParameters = DenseNetParameters(input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
                                        num_init_features=args.initial_num_filters, num_classes=num_output_classes,
                                        block_config=args.block_config,
                                        growth_rate=args.growth_rate,
                                        compression=args.compression,
                                        bottleneck_factor=args.bottleneck_factor,
                                        use_se=args.use_se)

model = Autoencoder(densenetParameters)

# Parameters for BCH Network
optimizer_params = {'weight_decay': args.weight_decay_coefficient,
                    'momentum': args.momentum,
                    'nesterov': args.nesterov}
scheduler_params = {'lr_max': args.learn_rate_max,
                    'lr_min': args.learn_rate_min,
                    'erf_alpha': args.erf_sched_alpha,
                    'erf_beta': args.erf_sched_beta}

bhc_experiment = ExperimentBuilder(network_model=model,
                                   use_gpu=args.use_gpu,
                                   experiment_name=args.experiment_name,
                                   num_epochs=args.num_epochs,
                                   continue_from_epoch=args.continue_from_epoch,
                                   train_data=train_loader,
                                   val_data=None,
                                   test_data=None,
                                   optimiser=args.optim_type,
                                   optim_params=optimizer_params,
                                   scheduler=args.sched_type,
                                   sched_params=scheduler_params,
                                   pretrained_weights_locations=args.pretrained_weights_locations)


if args.pretrained_weights_locations is None:
    experiment_metrics = bhc_experiment.run_experiment()
else:
    next(iter(train_loader))
    x, x_with_noise = next(iter(train_loader))

    image = x.to(device)
    image_with_noise = x_with_noise.to(device)

    out = bhc_experiment.model.forward(image_with_noise)  # forward the data in the model

    for i in range(0, out.size(0)):
        # Output
        out_image = out[i].detach().cpu()
        x = image[i].cpu()

        norm_mean = torch.Tensor(3, out.size(2), out.size(3))

        norm_mean[0] = normalization_mean[0]
        norm_mean[1] = normalization_mean[1]
        norm_mean[2] = normalization_mean[2]

        norm_var = torch.Tensor(3, out.size(2), out.size(3))

        norm_var[0] = normalization_var[0]
        norm_var[1] = normalization_var[1]
        norm_var[2] = normalization_var[2]

        out_image = (out_image * norm_var) + norm_mean
        x = (x * norm_var) + norm_mean

        x = x.permute(1, 2, 0)
        y = out_image.permute(1, 2, 0)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(x)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(y)

        plt.show()

        fig_size = (6, 3)  # Set figure size in inches (width, height)
        fig = plt.figure(figsize=fig_size)  # Create a new figure object
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(x)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()  # This minimises whitespace around the axes.
        fig.savefig(f'images/X - {i}.pdf', bbox_inches='tight')  # Save figure to current directory in PDF format


        fig_size = (6, 3)  # Set figure size in inches (width, height)
        fig = plt.figure(figsize=fig_size)  # Create a new figure object
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(y)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()  # This minimises whitespace around the axes.
        fig.savefig(f'images/Y - {i}.pdf', bbox_inches='tight')  # Save figure to current directory in PDF format

        plt.close()