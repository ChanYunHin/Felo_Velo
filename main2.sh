#!/bin/bash

python main.py \
    --GPU_num 3 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 100 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --VAE \
    --latent_size 1024 \
    --homo_flag

python main.py \
    --GPU_num 3 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 100 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --VAE \
    --latent_size 1024 \
    --homo_flag \
    --resnet_flag
