#!/bin/sh

seeds=(4324 342432 764365)
magnifications=("40X" "100X" "200X" "400X")
labeled_images_amount=(2 5 10 20 50)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      experiment_result_location="./experiments/fixmatch_${seed}_${magnification}_${labeled_images}"

      sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} "${experiment_result_location}"
    done
  done
done
