#!/bin/bash

partition="$1" # Cluster Partition
wandb_key="$2"
wandb_projectname="$3"
nr_test_samples="$4"

# Runs the experiments
counter=0
for data in {imagenet,}; do
  for model in {resnet18,vgg16}; do
    for xai_n_noisedraws in {1,20,50,300}; do
      for xai_methodname in {gradient,gradient-noabs,smoothgrad,intgrad,lrp-epsilon,lrp-zplus,guided-backprop}; do # grad-cam
        for eval_metricname in {emprt,smprt}; do
          for xai_noiselevel in {0.01,}; do
            for eval_layer_order in {"bottom_up","top_down"}; do
              for eval_normalise in {True,}; do
                # First Job
                jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
                counter=$((counter+1))
              done;
            done;
          done;
        done;
      done;
    done;
  done;
done;
echo "Started " ${counter} " JOBS"
