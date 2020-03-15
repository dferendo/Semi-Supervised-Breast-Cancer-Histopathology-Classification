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
weight_decay=0.000000001
#weight_decay=0.00001
learning_rate=0.01
use_se="True"

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      experiment_result_location="./experiments/fine_tuning_useunl_alllk_${seed}_${magnification}_${labeled_images}_${dropout}_${weight_decay}_${learning_rate}_${use_se}"

      python ../../src/main.py \
        --use_gpu "True" \
        --batch_size 15 \
        --num_epochs 200 \
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
        --block_config "4, 4, 4, 4" \
        --initial_num_filters 64 \
        --growth_rate 32 \
        --compression 0.5 \
        --bottleneck_factor 4 \
        --use_se ${use_se} \
        --se_reduction 16 \
        \
        --use_mix_match "False" \
        --loss_lambda_u ${loss_lambda_u} \
        \
        --use_fix_match "True" \
        --n_raug 3 \
        --m_raug 5 \
        --unlabelled_factor 3 \
        --fm_conf_threshold 0.85 \
        \
        --optim_type "SGD" \
        --momentum 0.9 \
        --nesterov "True" \
        --weight_decay_coefficient ${weight_decay} \
        --sched_type "Cos" \
        --learn_rate_max ${learning_rate} \
        --learn_rate_min 0.0000001 \
        --drop_rate 0 \
        --transformation_labeled_parameters "0, 0.5" \
        --transformation_unlabeled_parameters "0, 0.5" \
        --transformation_unlabeled_strong_parameters "0, 0.5" \
        --pretrained_weights_locations "../Autoencoder/1layer_dilation_lrelucbam_2exc_noaug_lrelu_1bn_autoencoder_test_(4,4,4,4)_40X_True/saved_models"


#                        --pretrained_weights_locations "../Autoencoder/experiments/autoencoder_test_40X_0/saved_models"
    done
  done
done
