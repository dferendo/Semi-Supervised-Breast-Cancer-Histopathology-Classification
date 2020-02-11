import os

import data_providers as data_providers
from arg_extractor import get_args

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

args, device = get_args()  # get arguments from command line

# Seeds
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(seed=args.seed)

# Transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    # TODO: Change this with zero mean
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Data Loading
train_dataset, val_dataset, test_dataset = data_providers.get_datasets(os.path.abspath('../data/BreaKHis_v1'), transformations)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

for batch in train_loader:
    print(batch[0].size())
#
# elif args.dataset_name == 'cifar10':
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         Cutout(n_holes=1, length=14),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     trainset = data_providers.CIFAR10(root='data', set_name='train', download=True, transform=transform_train)
#     train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
#
#     valset = data_providers.CIFAR10(root='data', set_name='val', download=True, transform=transform_test)
#     val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)
#
#     testset = data_providers.CIFAR10(root='data', set_name='test', download=True, transform=transform_test)
#     test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
#
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     num_output_classes = 10
#
# elif args.dataset_name == 'cifar100':
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         Cutout(n_holes=1, length=14),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     trainset = data_providers.CIFAR100(root='data', set_name='train', download=True, transform=transform_train)
#     train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
#
#     valset = data_providers.CIFAR100(root='data', set_name='val', download=True, transform=transform_test)
#     val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)
#
#     testset = data_providers.CIFAR100(root='data', set_name='test', download=True, transform=transform_test)
#     test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
#
#     num_output_classes = 100
#
# custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
#     input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
#     dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers,
#     use_bias=False,
#     num_output_classes=num_output_classes)
#
# conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
#                                     experiment_name=args.experiment_name,
#                                     num_epochs=args.num_epochs,
#                                     weight_decay_coefficient=args.weight_decay_coefficient,
#                                     continue_from_epoch=args.continue_from_epoch,
#                                     train_data=train_data, val_data=val_data,
#                                     test_data=test_data)  # build an experiment object
# experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics