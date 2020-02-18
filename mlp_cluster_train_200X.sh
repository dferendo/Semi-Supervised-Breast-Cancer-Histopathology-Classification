#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-04:00:00


export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets

# Activate the relevant virtual environment:
rsync -ua /home/${STUDENT_ID}/Leveraging-Unlabeled-Data-For-Breast-Cancer-Classification/data/BreaKHis_v1.tar.gz "${DATASET_DIR}"
tar -xzf "${DATASET_DIR}/BreaKHis_v1.tar.gz" -C "${DATASET_DIR}"

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python ./src/train.py --batch_size 20 --continue_from_epoch -1 --seed 0 \
                                                      --image_num_channels 3 --image_height 224 --image_width 224 \
                                                      --num_layers 3 --num_filters 16 \
                                                      --num_epochs 100 --experiment_name 'base_tune_test_1_200X' \
                                                      --use_gpu "True" --weight_decay_coefficient 0.0001 \
                                                      --optim_type "SGD" --momentum 0.9 --nesterov "True" \
                                                      --sched_type "ERF" --learn_rate_max 0.01 --learn_rate_min 0.0001 \
                                                      --erf_sched_alpha -3 --erf_sched_beta 3 \
                                                      --dataset_location "${DATASET_DIR}/BreaKHis_v1" --magnification "200X"