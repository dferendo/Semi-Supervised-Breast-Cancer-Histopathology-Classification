#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

magnifications=("40X")
unlabelled_splits=(0)
dropout_values=(0)
weight_decay_values=(0.001)
learning_rate=(0.01)

for magnification in "${magnifications[@]}"
do
  for unlabelled_split in "${unlabelled_splits[@]}"
  do
    for dropout in "${dropout_values[@]}"
    do
      for weight_decay in "${weight_decay_values[@]}"
      do
        for lr in "${learning_rate[@]}"
        do
          experiment_result_location="./experiments/autoencoder_test_${magnification}_${unlabelled_split}"

          python ../../src/RunAutoencoder.py --use_gpu "True" --batch_size 20 --num_epochs 100 --continue_from_epoch -1 --seed 0 \
                --image_num_channels 3 --image_height 224 --image_width 224 \
                --num_filters 24 \
                --dataset_location "${DATASET_DIR}" --experiment_name "${experiment_result_location}" \
                --optim_type "SGD" --momentum 0.9 --nesterov "True" --weight_decay_coefficient ${weight_decay} \
                --sched_type "Step" --learn_rate_max ${lr} --drop_rate ${dropout} \
                --magnification "${magnification}" --unlabelled_split ${unlabelled_split} \
                --use_mix_match "False" --multi_class "False"
          done
      done
    done
  done
done


