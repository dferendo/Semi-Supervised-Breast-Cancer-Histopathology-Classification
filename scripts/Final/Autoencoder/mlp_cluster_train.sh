#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Short
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
#rsync -ua /home/${STUDENT_ID}/Leveraging-Unlabeled-Data-For-Breast-Cancer-Classification/data/BreaKHis_v1.tar.gz "${DATASET_DIR}"
#
#if [ ! -d "${DATASET_DIR}/BreaKHis_v1/" ]; then
#  tar -xzf "${DATASET_DIR}/BreaKHis_v1.tar.gz" -C "${DATASET_DIR}"
#fi

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python ../../../src/RunAutoencoder.py \
          --seed ${1} \
          --num_epochs 300 \
          --experiment_name "${3}" \
          --use_gpu "True" \
          --continue_from_epoch -1 \
          \
          --batch_size 20 \
          --dataset_location "${DATASET_DIR}/BreaKHis_v1" \
          --magnification ${2} \
          \
          --image_num_channels 3 \
          --image_height 224 \
          --image_width 224 \
          \
          --block_config "4, 4, 4, 4" \
          --initial_num_filters 64 \
          --growth_rate 32 \
          --compression 0.5 \
          --bottleneck_factor 4 \
          --use_se "True" \
          --se_reduction 16 \
          \
          --weight_decay_coefficient 0.000001 \
          --learn_rate_max 0.006 \
          --learn_rate_min 0.00001 \
          --optim_type "Adam" \
          --momentum 0.9 \
          --sched_type "Cos" \
          --drop_rate 0

#            --pretrained_weights_locations "./1layer_dilation_lrelucbam_2exc_noaug_lrelu_1bn_autoencoder_test_(4,4,4,4)_40X_True/saved_models/"

          #            --increase_dilation_per_layer "True" \
          #            --val_size 0.2 \
#            --test_size 0.2 \
#            --num_of_workers 4 \
