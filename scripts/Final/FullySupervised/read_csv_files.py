from glob import glob
import os
import csv
import pandas as pd


def print_test_summary(path_to_read):
    columns = ['Seed', 'Magnification', 'Dropout', 'weight_decay_value', 'Learning Rate',
               'Test Accuracy', 'Test Loss', 'Test f1', 'Test Precision', 'Test Recall']

    with open('results_test.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []

            all_params = folder.split('_')[1:]
            seed = all_params[0]
            magnification = all_params[1]
            Dropout = all_params[2]
            weight_decay_value = all_params[3]
            learning_rate = all_params[4]

            row_to_output.extend(
                [seed, magnification, Dropout, weight_decay_value, learning_rate])

            test_file = os.path.join(folder, 'result_outputs', 'test_summary.csv')

            with open(test_file) as f:
                reader = csv.reader(f)
                next(reader, None)
                row_to_output.extend(next(reader, None))

                writer.writerow(row_to_output)


def print_validation_summary(path_to_read, epoch_amount):
    columns = ['Seed', 'Magnification', 'Dropout', 'weight_decay_value', 'Learning Rate',
               'Train acc', 'Train loss', 'Val acc', 'Val loss', 'Val f1', 'Val Precision', 'Val Recall', 'Epoch']

    with open('results_val.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(columns)

        for folder in glob(path_to_read):
            row_to_output = []

            all_params = folder.split('_')[1:]
            seed = all_params[0]
            magnification = all_params[1]
            Dropout = all_params[2]
            weight_decay_value = all_params[3]
            learning_rate = all_params[4]

            row_to_output.extend([seed, magnification, Dropout, weight_decay_value, learning_rate])

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
