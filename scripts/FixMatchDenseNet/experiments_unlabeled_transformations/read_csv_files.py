from glob import glob
import os
import csv
import pandas as pd


def print_test_summary(path_to_read):
    columns = ['Seed', 'Magnification', 'Labeled images', 'transformation_labeled_parameters', 'transformation_unlabeled_parameters', 'transformation_unlabeled_strong_parameters', 'Test Accuracy', 'Test Loss', 'Test f1', 'Test Precision', 'Test Recall']

    with open('results_test.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []

            all_params = folder.split('_')[2:]
            seed = all_params[0]
            magnification = all_params[1]
            labeled_images = all_params[2]
            m_raug = all_params[3]
            n_raug = all_params[4]
            unlabelled_factor = all_params[5]
            fm_conf_threshold = all_params[6]

            row_to_output.extend(
                [seed, magnification, labeled_images, m_raug, n_raug, unlabelled_factor, fm_conf_threshold])

            test_file = os.path.join(folder, 'result_outputs', 'test_summary.csv')

            with open(test_file) as f:
                reader = csv.reader(f)
                next(reader, None)
                row_to_output.extend(next(reader, None))

                writer.writerow(row_to_output)


def print_validation_summary(path_to_read, epoch_amount):
    columns = ['Seed', 'Magnification', 'Labeled images', 'transformation_labeled_parameters', 'transformation_unlabeled_parameters', 'transformation_unlabeled_strong_parameters',
               'Train acc', 'Train loss', 'Val acc', 'Val loss', 'Val f1', 'Val Precision', 'Val Recall', 'Epoch']

    with open('results_val.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []

            all_params = folder.split('_')[2:]
            seed = all_params[0]
            magnification = all_params[1]
            labeled_images = all_params[2]
            transformation_labeled_parameters = all_params[3]
            transformation_unlabeled_parameters = all_params[4]
            transformation_unlabeled_strong_parameters = all_params[5]

            row_to_output.extend([seed, magnification, labeled_images, transformation_labeled_parameters, transformation_unlabeled_parameters, transformation_unlabeled_strong_parameters])

            val_file = os.path.join(folder, 'result_outputs', 'summary.csv')

            df = pd.read_csv(val_file)

            # assert len(df) == epoch_amount, f'Not enough of epoch {folder}'

            row_to_output.extend(df.sort_values('val_loss').iloc[0].values)

            writer.writerow(row_to_output)


if __name__ == '__main__':
    path_to_read = './experiments/*'
    test_summary = False

    if test_summary:
        print_test_summary(path_to_read)
    else:
        print_validation_summary(path_to_read, 100)
