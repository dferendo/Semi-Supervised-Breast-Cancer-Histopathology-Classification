#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

seeds=(4545 342432 764365)
magnifications=("40X")
labeled_images_amount=(5)
weight_decay_values=(0.000000001 0.0000001 0.00001)
learning_rates=(0.01 0.001 0.0001)
fm_conf_thresholds=(0.85 0.95)
loss_lambda_u=1

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      for weight_decay_value in "${weight_decay_values[@]}"
      do
        for learning_rate in "${learning_rates[@]}"
        do
          for fm_conf_threshold in "${fm_conf_thresholds[@]}"
          do
            experiment_result_location="./experiments/autoencoder_fixmatch_${seed}_${magnification}_${labeled_images}_${weight_decay_value}_${learning_rate}_${loss_lambda_u}_${fm_conf_threshold}"

            if [ ! -f "${experiment_result_location}/result_outputs/test_summary.csv" ]; then
              sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} ${weight_decay_value} ${learning_rate} ${loss_lambda_u} ${fm_conf_threshold} "${experiment_result_location}"
            fi
          done
        done
      done
    done
  done
done
