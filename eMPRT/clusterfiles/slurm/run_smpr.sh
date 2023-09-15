#!/bin/bash

partition="$1" # Cluster Partition
wandb_key="$2"
wandb_projectname="$3"
nr_test_samples="$4"

# Runs the experiments
counter=0
for data in {imagenet,}; do
  for model in {vgg16,resnet18}; do
    for xai_n_noisedraws in {1,5,10,15,20,25,30,35,40,45,50,100,150,200,250,300}; do
      for xai_methodname in {gradient,lrp-epsilon,lrp-zplus,guided-backprop,}; do # grad-cam
        for eval_metricname in {smprt,emprt}; do
          for xai_noiselevel in {0.1,}; do
            for eval_layer_order in {"bottom_up","top_down"}; do
              for eval_normalise in {True,False}; do
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
