#!/bin/bash

seed="$1" # Random seed
partition="$2" # Cluster Partition
wandb_key="$3"

# Runs the experiments
counter=0
for data in {imagenet,}; do
  for model in {vgg16,}; do
    for xai_methodname in {grad-cam,}; do
      for xai_n_noisedraws in {10,}; do
        for xai_noiselevel in {0.1,}; do
          for eval_layer_order in {"bottom_up",}; do
            for eval_n_perturbations in {1,}; do
              for eval_perturbation_noiselevel in {0.1,}; do
                for eval_n_randomizations in {1,}; do
                  # First Job
                  jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_n_perturbations}-${eval_perturbation_noiselevel}-${eval_n_randomizations}-${seed}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_n_perturbations} ${eval_perturbation_noiselevel} ${eval_n_randomizations} ${seed} ${wandb_key} ${partition})
                  counter=$((counter+1))
                done;
              done;
            done;
          done;
        done;
      done;
    done;
  done;
done;

echo "Started " ${counter} " JOBS"
