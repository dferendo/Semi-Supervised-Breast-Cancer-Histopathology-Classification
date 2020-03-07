#!/bin/sh

export DATASET_DIR="../../../../data/BreaKHis_v1/"

magnification="40X"
dropout=0.2
weight_decay=0.001
learning_rate=0.1

labeled_images_amount=(5 10 25 50)
seeds=(324832)

for seed in "${seeds[@]}"
do
  for labeled_images in "${labeled_images_amount[@]}"
  do
    experiment_result_location="./experiments/labeled_images_exp_1_${magnification}_${seed}_${labeled_images}"

    python ../../../../src/main.py --use_gpu "True" --batch_size 20 --num_epochs 100 --continue_from_epoch -1 --seed ${seed} \
                  --image_num_channels 3 --image_height 224 --image_width 224 \
                  --num_filters 24 \
                  --dataset_location "${DATASET_DIR}" --experiment_name "${experiment_result_location}" \
                  --optim_type "SGD" --momentum 0.9 --nesterov "True" --weight_decay_coefficient ${weight_decay} \
                  --sched_type "Step" --learn_rate_max ${learning_rate} --drop_rate ${dropout} \
                  --magnification "${magnification}" --use_mix_match "False" --multi_class "True" \
                  --labelled_images_amount ${labeled_images}
  done
done


#unlabeled_splits=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#
#for seed in "${seeds[@]}"
#do
#  for unlabeled_split in "${unlabeled_splits[@]}"
#  do
#    experiment_result_location="./experiments/labeled_images_exp_1_${magnification}_${seed}_${labeled_images}"
#
#    python ../../../../src/main.py --use_gpu "True" --batch_size 20 --num_epochs 100 --continue_from_epoch -1 --seed ${seed} \
#                  --image_num_channels 3 --image_height 224 --image_width 224 \
#                  --num_filters 24 \
#                  --dataset_location "${DATASET_DIR}" --experiment_name "${experiment_result_location}" \
#                  --optim_type "SGD" --momentum 0.9 --nesterov "True" --weight_decay_coefficient ${weight_decay} \
#                  --sched_type "Step" --learn_rate_max ${learning_rate} --drop_rate ${dropout} \
#                  --magnification "${magnification}" --use_mix_match "False" --multi_class "False" \
#                  --unlabelled_split ${unlabeled_split}
#  done
#done


