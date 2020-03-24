#!/bin/sh

seeds=(4545 342432 764365)
magnifications=("40X")
labeled_images_amount=(5)

arch_block=("4, 4, 4, 4")
arch_filters=(64)
arch_growth_rate=(32)

use_ses=("False")
increase_dilations=("False")

loss_lambda_us=(50 75)
dropout_values=(0. 0.1 0.2)
weight_decay_values=(0.00001 0.0001 0.001)
learning_rates=(0.1 0.01)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      for index in "${!arch_block[@]}"
      do
        for use_se in "${use_ses[@]}"
        do
          for increase_dilation in "${increase_dilations[@]}"
          do
            for loss_lambda_u in "${loss_lambda_us[@]}"
            do
              for dropout_value in "${dropout_values[@]}"
              do
                for weight_decay_value in "${weight_decay_values[@]}"
                do
                  for learning_rate in "${learning_rates[@]}"
                  do
                    experiment_result_location="./experiments/basemixmatch_${seed}_${magnification}_${labeled_images}_${arch_block[$index]}_${arch_filters[$index]}_${arch_growth_rate[$index]}_${use_se}_${increase_dilation}_${loss_lambda_u}_${dropout_value}_${weight_decay_value}_${learning_rate}"

                    sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} "${arch_block[$index]}" "${arch_filters[$index]}" "${arch_growth_rate[$index]}" "${use_se}" "${increase_dilation}" ${loss_lambda_u} ${dropout_value} ${weight_decay_value} ${learning_rate} "${experiment_result_location}"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
