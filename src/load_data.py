import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class BreaKHisDataset(Dataset):
    """
    Reading the BreaKHis Dataset. Please keep the original dataset structure
    """
    def __init__(self, data_root, transform=None):
        self.dataset = []
        self.transform = transform

        samples = []

        # Skip histology_slides folder
        histology_folder = os.path.join(data_root, os.listdir(data_root)[0])

        # Skip breast folder
        classes_folder = os.path.join(histology_folder, os.listdir(histology_folder)[0])

        for class_idx, class_name in enumerate(os.listdir(classes_folder)):
            class_folder = os.path.join(classes_folder, class_name)

            # Ignore files
            if not os.path.isdir(class_folder):
                continue

            # Skip statistics folder, go to the folder containing patients
            statistics_folders = os.path.join(class_folder,
                                              [SOB for SOB in os.listdir(class_folder)
                                               if os.path.isdir(os.path.join(class_folder, SOB))][0])

            # Loop through the subclasses
            for subclass_idx, subclass_name in enumerate(os.listdir(statistics_folders)):
                subclass_folder = os.path.join(statistics_folders, subclass_name)

                # Every patient has unique folder
                for patient_name in os.listdir(subclass_folder):
                    patient_folder = os.path.join(subclass_folder, patient_name)

                    # For every patient, there are multiple images with different magnifications
                    for magnification_idx, magnification in enumerate(os.listdir(patient_folder)):
                        magnification_folder = os.path.join(patient_folder, magnification)

                        for image in os.listdir(magnification_folder):
                            image_location = os.path.join(magnification_folder, image)
                            self.dataset.append([class_idx, subclass_idx, magnification_idx, image_location])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the image from the file and return the label
        sample = self.dataset[idx]
        image_location = sample[3]

        img = Image.open(image_location)

        if self.transform is not None:
            img = self.transform(img)

        label = sample[0]

        return img, label


if __name__ == '__main__':
    from torchvision import transforms

    transforms_temp = transforms.Compose([
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor()
    ])

    dataset = BreaKHisDataset(os.path.abspath('../data/BreaKHis_v1'), transforms_temp)
    train_loader = DataLoader(dataset, batch_size=8000, shuffle=True, num_workers=4)


    for batch_idx, (x, y) in enumerate(train_loader):
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])
        break

    print(dataset)
