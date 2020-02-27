#!/bin/sh

#unlabelled_splits=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
unlabelled_splits=(0)
dropout_values=(0 0.2 0.5)
weight_decay_values=(0.00001 0.0001 0.001)
learning_rate=(0.1 0.01)

for unlabelled_split in "${unlabelled_splits[@]}"
do
  for dropout in "${dropout_values[@]}"
  do
    for weight_decay in "${weight_decay_values[@]}"
    do
      for lr in "${learning_rate[@]}"
      do
        experiment_result_location="./experiments/base_finetune_test_1_nomag_${unlabelled_split}_${dropout}_${weight_decay}_${lr}"

        sbatch mlp_cluster_train_no_mag.sh $experiment_result_location $unlabelled_split $dropout $weight_decay $lr
      done
    done
  done
done
