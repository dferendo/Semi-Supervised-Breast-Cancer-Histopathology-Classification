#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

#dropout=0.2
#weight_decay=0.00001
#learning_rate=0.1
loss_lambda_u=1
#
#seeds=(324832 9392 344 89436)
#dropouts=(0.1)
#weight_decays=(0.001 0.00001 0.0001)
#learning_rates=(0.1 0.01 0.001)

# best dropout 0.2 wd 0.001 lr 0.001
# 2nd dropout 0 wd 1e-5 lr 0.001

seeds=(9392)
labeled_images_amount=(5)

dropout=0
magnifications=("40X")
weight_decay=0.0001
learning_rate=0.001
use_se="True"

block_config="4, 4, 4, 4"
initial_num_filters=64
growth_rate=32

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      experiment_result_location="./experiments/fixmatch_architecture_${seed}_${magnification}_${labeled_images}_${dropout}_${weight_decay}_${learning_rate}_${use_se}"

      python ../../src/main.py \
              --use_gpu "True" \
              --batch_size 20 \
              --num_epochs 1 \
              --continue_from_epoch -1 \
              --seed ${seed} \
              \
              --magnification "${magnification}" \
              --dataset_location "${DATASET_DIR}" \
              --experiment_name "${experiment_result_location}" \
              --multi_class "False" \
              --labelled_images_amount ${labeled_images} \
              \
              --image_num_channels 3 \
              --image_height 224 \
              --image_width 224 \
              \
              --block_config "${block_config}" \
              --initial_num_filters ${initial_num_filters} \
              --growth_rate ${growth_rate} \
              --compression 0.5 \
              --bottleneck_factor 4 \
              --use_se "${use_se}" \
              --se_reduction 16 \
              \
              --use_mix_match "False" \
              --loss_lambda_u ${loss_lambda_u} \
              \
              --use_fix_match "True" \
              --n_raug 3 \
              --m_raug 10 \
              --unlabelled_factor 1 \
              --fm_conf_threshold 0.95 \
              \
              --optim_type "SGD" \
              --momentum 0.9 \
              --nesterov "True" \
              --weight_decay_coefficient ${weight_decay} \
              --sched_type "FixMatchCos" \
              --learn_rate_max ${learning_rate} \
              --drop_rate ${dropout} \
              \
              --transformation_labeled_parameters "0, 0.5, 1, 0.5" \
              --transformation_unlabeled_parameters "0, 0.5, 1, 0.5" \
              --transformation_unlabeled_strong_parameters "0, 0.5, 1, 0.5" \


#                        --pretrained_weights_locations "../Autoencoder/experiments/autoencoder_test_40X_0/saved_models"
    done
  done
done
