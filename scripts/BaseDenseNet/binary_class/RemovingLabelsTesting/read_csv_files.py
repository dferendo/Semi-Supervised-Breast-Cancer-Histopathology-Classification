from glob import glob
import os
import csv
import pandas as pd


def print_test_summary(path_to_read):
    columns = ['Magnification', 'Seed', 'Labeled Amount Images', 'Test Accuracy', 'Test Loss', 'Test f1', 'Test Precision', 'Test Recall']

    with open('results_test.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []
            all_params = folder.split('_')[4:]
            magnification = all_params[0]
            seed = all_params[1]
            labeled_images_amount = all_params[2]

            row_to_output.extend([magnification, seed, labeled_images_amount])

            test_file = os.path.join(folder, 'result_outputs', 'test_summary.csv')

            with open(test_file) as f:
                reader = csv.reader(f)
                next(reader, None)
                row_to_output.extend(next(reader, None))

                writer.writerow(row_to_output)


def print_validation_summary(path_to_read, epoch_amount):
    columns = ['Magnification', 'Seed', 'Labeled Amount Images', 'Train acc', 'Train loss', 'Val acc',
               'Val loss', 'Val f1', 'Val Precision', 'Val Recall', 'Epoch']

    with open('results_val.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []
            all_params = folder.split('_')[4:]
            magnification = all_params[0]
            seed = all_params[1]
            labeled_images_amount = all_params[2]

            row_to_output.extend([magnification, seed, labeled_images_amount])

            val_file = os.path.join(folder, 'result_outputs', 'summary.csv')

            df = pd.read_csv(val_file)

            assert len(df) == epoch_amount, 'Not enough of epoch'

            row_to_output.extend(df.sort_values('val_loss').iloc[0].values)

            writer.writerow(row_to_output)


if __name__ == '__main__':
    path_to_read = './experiments/*'
    test_summary = True

    if test_summary:
        print_test_summary(path_to_read)
    else:
        print_validation_summary(path_to_read, 100)