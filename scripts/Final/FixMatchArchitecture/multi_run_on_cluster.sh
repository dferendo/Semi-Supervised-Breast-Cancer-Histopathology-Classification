#!/bin/sh

seeds=(4545 342432 764365)
magnifications=("40X")
labeled_images_amount=(5)

arch_block=("4, 4, 4, 4" "6, 6, 6, 6" "6, 12, 24, 16")
arch_filters=(64 24 24)
arch_growth_rate=(32 12 12)

use_ses=("True" "False")
increase_dilations=("True" "False")

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
            experiment_result_location="./experiments/basemixmatch_${seed}_${magnification}_${labeled_images}_${arch_block[$index]}_${arch_filters[$index]}_${arch_growth_rate[$index]}_${use_se}_${increase_dilation}"

            sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} "${arch_block[$index]}" "${arch_filters[$index]}" "${arch_growth_rate[$index]}" "${use_se}" "${increase_dilation}" "${experiment_result_location}"
          done
        done
      done
    done
  done
done
