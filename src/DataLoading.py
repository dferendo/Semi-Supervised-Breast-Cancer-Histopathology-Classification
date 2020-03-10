from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import pandas as pd
import numpy as np


class BreaKHisDataset(Dataset):
    """
    Reading the BreaKHis Dataset. Please keep the original dataset structure
    """
    def __init__(self, df, transform=None, noise=None):
        self.transform = transform
        self.noise = noise
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_location = self.df.iloc[idx]['Image Location']
        original_image = Image.open(image_location)

        if self.transform is not None:
            original_image = self.transform(original_image)

        if self.noise is not None:
            noise = self.noise(original_image)
        else:
            noise = original_image

        return original_image, noise


class DataParameters(object):
    # From 0 to 1
    val_size = 0.2
    test_size = 0.2
    magnification = None
    unlabeled_split = None
    labelled_images_amount = None
    unlabeled_transformations = None
    num_workers = 4

    def __init__(self, data_root, batch_size, transformations, transformations_test, multi_class):
        self.data_root = data_root
        self.batch_size = batch_size
        self.transformations = transformations
        self.transformations_test = transformations_test
        self.multi_class = multi_class


def print_statistics(df, dataset):
    print(f'{dataset} dataset statistics.')
    print(df['Class Name'].value_counts())
    print(df['Subclass Name'].value_counts())


def get_all_images_location_with_classes(data_root):
    dataset = {}
    dataset_temp = []

    # Skip histology_slides folder
    histology_folder = os.path.join(data_root, os.listdir(data_root)[0])

    # Skip breast folder
    classes_folder = os.path.join(histology_folder, os.listdir(histology_folder)[0])

    for class_name in os.listdir(classes_folder):
        class_folder = os.path.join(classes_folder, class_name)

        # Ignore files
        if not os.path.isdir(class_folder):
            continue

        # Skip statistics folder, go to the folder containing patients
        statistics_folders = os.path.join(class_folder,
                                          [SOB for SOB in os.listdir(class_folder)
                                           if os.path.isdir(os.path.join(class_folder, SOB))][0])

        # Loop through the subclasses
        for subclass_name in os.listdir(statistics_folders):
            subclass_folder = os.path.join(statistics_folders, subclass_name)

            # Every patient has unique folder
            for patient_name in os.listdir(subclass_folder):
                patient_folder = os.path.join(subclass_folder, patient_name)

                # For every patient, there are multiple images with different magnifications
                for magnification in os.listdir(patient_folder):
                    magnification_folder = os.path.join(patient_folder, magnification)

                    for image in os.listdir(magnification_folder):
                        image_location = os.path.join(magnification_folder, image)
                        dataset[image] = [patient_name, class_name, subclass_name, magnification, image_location]
                        dataset_temp.append([patient_name, class_name, subclass_name, magnification, image_location])

    return dataset_temp


def split_dataset_into_sets(data_parameters, dataset):
    """
    Note: The patients are disjoints from the sets (Ie the same patient is not found in multiple sets)
    :param data_parameters:
    :return:
    """
    columns = ['Patient Name', 'Class Name', 'Subclass Name', 'Magnification', 'Image Location']

    magnification = data_parameters.magnification
    val_size = data_parameters.val_size
    test_size = data_parameters.test_size
    unlabeled_split = data_parameters.unlabeled_split
    labelled_images_amount = data_parameters.labelled_images_amount

    df = pd.DataFrame(dataset, columns=columns)

    if magnification is not None:
        assert magnification == '40X' or magnification == '100X' or magnification == '200X' or magnification == '400X'
        df = df[df['Magnification'] == magnification]

    df_train = pd.DataFrame(columns=columns)
    df_val = pd.DataFrame(columns=columns)
    df_test = pd.DataFrame(columns=columns)

    # Get the validation set from the test set
    df_grouped_by_subclass = df.groupby(['Subclass Name'])

    for group_name, df_group in df_grouped_by_subclass:
        df_group_by_patients = df_group.groupby(['Patient Name'])
        patients_sizes = df_group_by_patients.size().sort_values(ascending=True)

        current_patients_amount = 0

        for patient_name in patients_sizes.index:
            selected_group = df_group_by_patients.get_group(patient_name)

            if current_patients_amount < len(df_group_by_patients) * val_size:
                df_val = df_val.append(selected_group)
            elif current_patients_amount < len(df_group_by_patients) * (val_size + test_size):
                df_test = df_test.append(selected_group)
            else:
                df_train = df_train.append(selected_group)

            current_patients_amount += 1

    df_train_labeled = pd.DataFrame(columns=columns)
    df_train_unlabeled = pd.DataFrame(columns=columns)

    if unlabeled_split is not None and unlabeled_split != 0.:
        print(unlabeled_split)
        assert 0 < unlabeled_split < 1
        assert labelled_images_amount is None

        # Group by subclasses and into two dataframes, one with labeled, one without.
        df_trained_grouped_by_subclass = df_train.groupby(['Subclass Name'])

        for group_name, df_group in df_trained_grouped_by_subclass:
            split_group = np.array_split(df_group.sample(frac=1), [int(unlabeled_split * len(df_group))])

            df_train_unlabeled = df_train_unlabeled.append(split_group[0])
            df_train_labeled = df_train_labeled.append(split_group[1])
    elif labelled_images_amount is not None:
        assert unlabeled_split is None or unlabeled_split == 0
        assert 0 < labelled_images_amount <= 50

        # Group by subclasses and into two dataframes, one with labeled, one without.
        df_trained_grouped_by_subclass = df_train.groupby(['Subclass Name'])

        for group_name, df_group in df_trained_grouped_by_subclass:
            randomize_group = df_group.sample(frac=1)

            df_train_labeled = df_train_labeled.append(randomize_group[:labelled_images_amount])
            df_train_unlabeled = df_train_unlabeled.append(randomize_group[labelled_images_amount:])
    else:
        df_train_labeled = df_train

    print(f'Total number of images considered: {len(df)}')
    print_statistics(df_train_labeled, 'Train')
    print_statistics(df_train_unlabeled, 'Unlabeled Trained')
    print_statistics(df_val, 'Validation')
    print_statistics(df_test, 'Test')

    return df_train_labeled, df_train_unlabeled, df_val, df_test


def get_datasets(data_parameters, noising):
    dataset = get_all_images_location_with_classes(data_parameters.data_root)
    df_train_labeled, df_train_unlabeled, df_val, df_test = split_dataset_into_sets(data_parameters, dataset)

    train_dataset = BreaKHisDataset(df_train_labeled, data_parameters.transformations, noising)

    train_loader = DataLoader(train_dataset, batch_size=data_parameters.batch_size, shuffle=True,
                              num_workers=data_parameters.num_workers, drop_last=True)

    return train_loader


def calculate_the_mean_and_variance_of_the_dataset(loader):
    """
    Reference: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    :param train_loader:
    :return:
    """
    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in loader:
        data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std
