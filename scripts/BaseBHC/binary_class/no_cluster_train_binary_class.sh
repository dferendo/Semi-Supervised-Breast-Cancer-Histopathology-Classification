#!/bin/sh

export DATASET_DIR="../../../data/BreaKHis_v1/"

magnifications=("40X")
unlabelled_splits=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for magnification in "${magnifications[@]}"
do
  for unlabelled_split in "${unlabelled_splits[@]}"
  do
    experiment_result_location="./experiments/class_test_1_${magnification}_${unlabelled_split}"

    if [ "$magnification" == "40X" ]
    then
      erf_sched_alpha=-4
      erf_sched_beta=4
    else
      erf_sched_alpha=-3
      erf_sched_beta=3
    fi

    python ../../../src/main.py --use_gpu "True" --batch_size 20 --num_epochs 300 --continue_from_epoch -1 --seed 0 \
                         --image_num_channels 3 --image_height 224 --image_width 224 \
                         --num_layers 3 --num_filters 16 \
                         --dataset_location "${DATASET_DIR}" --experiment_name "${experiment_result_location}" \
                         --optim_type "SGD" --momentum 0.9 --nesterov "True" --weight_decay_coefficient 0.0001 \
                         --sched_type "ERF" --learn_rate_max 0.01 --learn_rate_min 0.0001 \
                         --erf_sched_alpha ${erf_sched_alpha} --erf_sched_beta ${erf_sched_beta} \
                         --magnification "${magnification}" --unlabelled_split ${unlabelled_split} \
                         --use_mix_match "False" --multi_class "False"
  done
done


