from glob import glob
import os
import csv

path_to_read = './experiments/*'

for folder in glob(path_to_read):
    all_params = folder.split('_')[4:]
    magnification = all_params[0]
    unlabeled_amount = all_params[1]
    dropout_value = all_params[2]
    weight_decay = all_params[3]
    learning_rate = all_params[4]

    test_file = os.path.join(folder, 'result_outputs', 'test_summary.csv')

    with open(test_file) as f:
        reader = csv.reader(f)

        for read in reader:
            print(read)
