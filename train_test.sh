#!/bin/sh

export DATASET_DIR="./data/BreaKHis_v1/"

python ./src/train.py --batch_size 20 --continue_from_epoch -1 --seed 0 \
                                                      --image_num_channels 3 --image_height 224 --image_width 224 \
                                                      --num_layers 3 --num_filters 16 \
                                                      --num_epochs 100 --experiment_name 'base_tune_test' \
                                                      --use_gpu "True" --weight_decay_coefficient 0.0001 \
                                                      --optim_type "SGD" --momentum 0.9 --nesterov "True" \
                                                      --sched_type "ERF" --learn_rate_max 0.01 --learn_rate_min 0.0001 \
                                                      --erf_sched_alpha -3 --erf_sched_beta 3 \
                                                       --dataset_location "${DATASET_DIR}" --magnification "40X"
