#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

loss_lambda_u=1


seeds=(9392 342432 764365)
magnifications=("40X")
labeled_images_amount=(5)

transformation_labeled_parameters=("0, 0.5, 1, 0.5" "0, 0.5, 1, 0.5, 2, 20" "0, 0.5, 1, 0.5, 3, 0.1, 0.1" "0, 0.5, 1, 0.5, 2, 20, 3, 0.1, 0.1")
transformation_unlabeled_parameters=("0, 0.5" "0, 0.5" "0, 0.5" "0, 0.5")
transformation_unlabeled_strong_parameters=("0, 0.5" "0, 0.5" "0, 0.5" "0, 0.5")

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      for index in "${!transformation_labeled_parameters[@]}"
      do
        experiment_result_location="./experiments/fixmatch_rotations_${seed}_${magnification}_${labeled_images}_${transformation_labeled_parameters[$index]}_${transformation_unlabeled_parameters[$index]}_${transformation_unlabeled_strong_parameters[$index]}"

        if [ ! -f "${experiment_result_location}/result_outputs/test_summary.csv" ]; then
          sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} "${transformation_labeled_parameters[$index]}" "${transformation_unlabeled_parameters[$index]}" "${transformation_unlabeled_strong_parameters[$index]}" ${experiment_result_location}
        fi
      done
    done
  done
done
