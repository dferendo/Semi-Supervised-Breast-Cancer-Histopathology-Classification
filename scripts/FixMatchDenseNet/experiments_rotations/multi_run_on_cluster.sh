#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

loss_lambda_u=1


seeds=(9392 342432 764365)
magnifications=("40X")
labeled_images_amount=(5)
m_raugs=(5 10)
n_raugs=(1 3)
unlabelled_factors=(1 3 4 5)
fm_conf_thresholds=(0.95 0.85)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      for m_raug in "${m_raugs[@]}"
      do
        for n_raug in "${n_raugs[@]}"
        do
          for unlabelled_factor in "${unlabelled_factors[@]}"
          do
            for fm_conf_threshold in "${fm_conf_thresholds[@]}"
            do
              experiment_result_location="./experiments/fixmatch_rotations_${seed}_${magnification}_${labeled_images}_${m_raug}_${n_raug}_${unlabelled_factor}_${fm_conf_threshold}"

              if [ ! -f "${experiment_result_location}/result_outputs/test_summary.csv" ]; then
                sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} ${m_raug} ${n_raug} ${unlabelled_factor} ${fm_conf_threshold} ${experiment_result_location}
              fi
            done
          done
        done
      done
    done
  done
done
