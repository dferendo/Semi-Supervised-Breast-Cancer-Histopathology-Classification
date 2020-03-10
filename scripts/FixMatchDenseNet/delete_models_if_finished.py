from pathlib import Path
import shutil
import os

experiments_folder = 'experiments/'
list_experiment_folders = [f.path for f in os.scandir(experiments_folder) if f.is_dir()]


for experiment_folder_path in list_experiment_folders:
    if os.path.exists(experiment_folder_path + '/result_outputs/test_summary.csv') and \
            os.path.exists(experiment_folder_path + '/saved_models'):
        print(experiment_folder_path + '/result_outputs/test_summary.csv')
        shutil.rmtree(experiment_folder_path + '/saved_models')