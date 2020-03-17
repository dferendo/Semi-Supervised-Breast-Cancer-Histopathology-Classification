#!/bin/sh

seeds=(4324 342432 764365)
magnifications=("40X" "100X" "200X" "400X")
autoencoder_locations=("temp1" "temp2" "temp3" "temp4")
labeled_images_amount=(5 10 20 50)

for seed in "${seeds[@]}"
do
  for index in "${!magnifications[@]}"
  do
    for labeled_images in "${labeled_images_amount[@]}"
    do
      magnification=${magnifications[$index]}
      autoencoder_location=${autoencoder_locations[$index}
      experiment_result_location="./experiments/autoencoderfixmatch_${seed}_${magnification}_${labeled_images}_${autoencoder_location}"

      sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${labeled_images} "${autoencoder_location}" "${experiment_result_location}"
    done
  done
done
