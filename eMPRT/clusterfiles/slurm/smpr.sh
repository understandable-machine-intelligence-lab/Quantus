#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=smpr
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

dataset_name="$1"
model_name="$2"
xai_methodname="$3"
eval_metricname="$4"
nr_test_samples="$5"
xai_n_noisedraws="$6"
xai_noiselevel="$7"
eval_layer_order="$8"
wandb_key="${9}"
wandb_projectname="${10}"

source "/etc/slurm/local_job_dir.sh"
mkdir -p ${LOCAL_JOB_DIR}/results
mkdir -p ${LOCAL_JOB_DIR}/datasets
mkdir -p ${LOCAL_JOB_DIR}/labelmaps

echo "COPYING LABELMAPS..."
cp -r ${HOME}/Quantus/eMPRT/scripts/label_map_imagenet.json ${LOCAL_JOB_DIR}/labelmaps

dir ${LOCAL_JOB_DIR}
dir ${LOCAL_JOB_DIR}/results

# Handle torch not recognizing GPU on Cluster
for i in {1..30}; do
  /usr/bin/time -v apptainer \
    run \
          --nv \
          --bind /data/datapool/datasets/ImageNet-complete/:/mnt/datasets/${dataset_name} \
          --bind ${LOCAL_JOB_DIR}/labelmaps:/mnt/labelmaps/ \
          --bind ${LOCAL_JOB_DIR}/results:/mnt/output/ \
          ../singularity/smpr.sif evaluate-randomisation \
            /mnt/output/ \
            ${dataset_name} \
            /mnt/datasets/${dataset_name} \
            /mnt/labelmaps/label_map_imagenet.json \
            ${model_name} \
            ${xai_methodname} \
            ${eval_metricname} \
            --nr-test-samples ${nr_test_samples} \
            --xai-n-noisedraws ${xai_n_noisedraws} \
            --xai-noiselevel ${xai_noiselevel} \
            --eval-layerorder ${eval_layer_order} \
            --use-cpu False \
            --batch-size 32 \
            --shuffle False \
            --wandb-key ${wandb_key} \
            --wandb-projectname ${wandb_projectname}
  ret_val=$?;

  if (( $ret_val == 77 )); then 
    echo "Apptainer call failed at torch.cuda.is_available. Trying again for the $i th time"
    sleep 10;
    continue;
  fi
  
  break;
done

rm -rf ${LOCAL_JOB_DIR}/results

