#!/bin/bash

# Training script for flat_cloth dataset
# For inference without evaluation (skips GT evaluation)

datapath=/home/arif/Projects/GLASS-new/datasets/gray
augpath=/home/arif/Projects/GLASS-new/datasets/dtd/images
classes=('flat_cloth')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd /home/arif/Projects/GLASS-new
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 640 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 384 \
    --imagesize 384 "${flags[@]}" mvtec $datapath $augpath