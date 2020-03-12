#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

magnifications=("40X")
use_ses=("True")
seeds=(9392)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for use_se in "${use_ses[@]}"
    do
      experiment_result_location="./experiments/2exc_noaug_lrelu_1bn_autoencoder_test_(4,4,4,4)_${magnification}_${use_se}"

      python ../../src/RunAutoencoder.py \
            --seed ${seed} \
            --num_epochs 300 \
            --experiment_name "${experiment_result_location}" \
            --use_gpu "True" \
            --continue_from_epoch -1 \
            \
            --batch_size 20 \
            --dataset_location "${DATASET_DIR}" \
            --magnification ${magnification} \
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
            --use_se "${use_se}" \
            --se_reduction 16 \
            \
            --weight_decay_coefficient 0.0001 \
            --learn_rate_max 0.003 \
            --learn_rate_min 0.00001 \
            --optim_type "Adam" \
            --momentum 0.9 \
            --sched_type "Cos" \
            --drop_rate 0

            #            --increase_dilation_per_layer "True" \
            #            --val_size 0.2 \
#            --test_size 0.2 \
#            --num_of_workers 4 \

    done
  done
done



block_configs=("4, 4, 4, 4")


