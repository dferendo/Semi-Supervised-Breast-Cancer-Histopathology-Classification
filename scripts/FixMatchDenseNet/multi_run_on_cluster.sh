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

#seeds=(9392 344 89436)
#labeled_images_amount=(5)
#dropouts=(0 0.1)
#weight_decays=(0.0001 0.00001 0.000001)
#learning_rates=(0.005 0.001 0.0001)
#magnifications=("40X")


seeds=(9392 342432 764365)
labeled_images_amount=(5)
dropouts=(0 0.1)
weight_decays=(0.0001 0.00001 0.000001)
learning_rates=(0.005 0.001 0.0001)
magnifications=("40X")

use_ses=("True")
arch_block=("4, 4, 4, 4")
arch_filters=(64)
arch_growth_rate=(32)


for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      for dropout in "${dropouts[@]}"
      do
        for weight_decay in "${weight_decays[@]}"
        do
          for learning_rate in "${learning_rates[@]}"
          do
            for use_se in "${use_ses[@]}"
            do
              for index in "${!arch_block[@]}"
              do
                experiment_result_location="./experiments/fixmatch_architecture_${seed}_${arch_block[$index]}_${arch_filters[$index]}_${arch_growth_rate[$index]}_${magnification}_${labeled_images}_${dropout}_${weight_decay}_${learning_rate}_${use_se}"

                if [ ! -f "${experiment_result_location}/result_outputs/test_summary.csv" ]; then
                  sbatch mlp_cluster_train.sh ${seed} "${arch_block[$index]}" ${arch_filters[$index]} ${arch_growth_rate[$index]} "${magnification}" ${labeled_images} ${dropout} ${weight_decay} ${learning_rate} ${use_se} "${experiment_result_location}"
                fi
                done
            done
          done
        done
      done
    done
  done
done
