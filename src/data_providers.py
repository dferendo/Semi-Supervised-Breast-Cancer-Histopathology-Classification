import os
from torch.utils.data import Dataset
from PIL import Image

import os
import pandas as pd
import numpy as np


class BreaKHisDataset(Dataset):
    """
    Reading the BreaKHis Dataset. Please keep the original dataset structure
    """
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load the image from the file and return the label
        class_name = self.df.iloc[idx]['Class Name']

        if class_name == 'benign':
            target = np.array([1, 0])
        else:
            target = np.array([0, 1])

        image_location = self.df.iloc[idx]['Image Location']

        img = Image.open(image_location)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_all_images_location_with_classes(data_root):
    dataset = []

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
                        dataset.append([patient_name, class_name, subclass_name, magnification, image_location])

    return dataset


def split_dataset_into_sets(dataset, val_size, test_size):
    """
    Note: The patients are disjoints from the sets (Ie the same patient is not found in multiple sets)
    :param dataset:
    :return:
    """
    df = pd.DataFrame(dataset, columns=['Patient Name', 'Class Name', 'Subclass Name', 'Magnification', 'Image Location'])
    total_number_of_images = len(df)
    df_grouped_by_patients = df.groupby(['Patient Name'])

    all_groups_names = []

    for group_name, df_group in df_grouped_by_patients:
        all_groups_names.append(group_name)

    # Randomize groups
    np.random.shuffle(all_groups_names)

    current_amount_of_images = 0

    df_train = pd.DataFrame(columns=['Patient Name', 'Class Name', 'Subclass Name', 'Magnification', 'Image Location'])
    df_val = pd.DataFrame(columns=['Patient Name', 'Class Name', 'Subclass Name', 'Magnification', 'Image Location'])
    df_test = pd.DataFrame(columns=['Patient Name', 'Class Name', 'Subclass Name', 'Magnification', 'Image Location'])

    for group_key in all_groups_names:
        selected_group = df_grouped_by_patients.get_group(group_key)

        if current_amount_of_images < total_number_of_images * val_size:
            df_test = df_test.append(selected_group)
        elif current_amount_of_images < total_number_of_images * (val_size + test_size):
            df_val = df_val.append(selected_group)
        else:
            df_train = df_train.append(selected_group)

        current_amount_of_images += len(selected_group)

    return df_train, df_val, df_test


def get_datasets(data_root, transforms, val_size=0.15, test_size=0.15):
    dataset = get_all_images_location_with_classes(data_root)
    df_train, df_val, df_test = split_dataset_into_sets(dataset, val_size, test_size)

    return BreaKHisDataset(df_train, transforms), BreaKHisDataset(df_val, transforms), BreaKHisDataset(df_test, transforms)
