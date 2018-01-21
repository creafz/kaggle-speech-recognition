#!/usr/bin/env bash

# Without this setting PyTorch Dataloader may stuck
# in a deadlock when librosa is used to load wav files
export OMP_NUM_THREADS=1


python make_dirs.py


for fold_num in $(seq 0 4);
    do python train.py \
     --experiment-name tensorflow_audio_nasnetalarge_mel \
     --batch-size 52 \
     --lr 0.001 \
     --augmentation mel \
     --fold-num ${fold_num};
done


for fold_num in $(seq 0 4);
    do python train.py \
     --experiment-name tensorflow_audio_nasnetalarge_mfcc \
     --batch-size 52 \
     --lr 0.001 \
     --augmentation mfcc \
     --fold-num ${fold_num};
done


python predict.py \
    --experiment-name tensorflow_audio_nasnetalarge_mel \
    --augmentation mel;


python predict.py \
    --experiment-name tensorflow_audio_nasnetalarge_mfcc \
    --augmentation mfcc;


python make_submission.py \
    tensorflow_audio_nasnetalarge_mel.pkl \
    tensorflow_audio_nasnetalarge_mfcc.pkl \
    --submission-filename submission.csv;