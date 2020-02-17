import os

import data_providers as data_providers
from arg_extractor import get_args

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from experiment_builder import ExperimentBuilder
from model_architectures import BHCNetwork

args, device = get_args()  # get arguments from command line

# Seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# Set the mean and variance for normalization
if args.magnification == '40X':
    normalization_mean = (0.8035, 0.6524, 0.7726)
    normalization_var = (0.0782, 0.1073, 0.0765)
elif args.magnification == '100X':
    normalization_mean = (0.7956, 0.6360, 0.7709)
    normalization_var = (0.0973, 0.1314, 0.0853)
elif args.magnification == '200X':
    normalization_mean = (0.7886, 0.6228, 0.7681)
    normalization_var = (0.0996, 0.1317, 0.0782)
elif args.magnification == '400X':
    normalization_mean = (0.7565, 0.5902, 0.7428)
    normalization_var = (0.1163, 0.1556, 0.0857)
else:
    normalization_mean = (0.7868, 0.6263, 0.7642)
    normalization_var = (0.0974, 0.1310, 0.0814)

# Transformations
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    # transforms.Pad(),
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(normalization_mean, normalization_var)
])

# Data Loading
train_dataset, val_dataset, test_dataset = data_providers.get_datasets(os.path.abspath(args.dataset_location),
                                                                       transformations, magnification=args.magnification)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

custom_conv_net = BHCNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
    num_filters=args.num_filters, num_layers=args.num_layers,
    use_bias=True,
    num_output_classes=2)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
                  experiment_name=args.experiment_name,
                  num_epochs=args.num_epochs,
                  weight_decay_coefficient=args.weight_decay_coefficient,
                  continue_from_epoch=args.continue_from_epoch,
                  train_data=train_loader, val_data=validation_loader,
                  test_data=test_loader)

experiment_metrics, test_metrics = conv_experiment.run_experiment()
