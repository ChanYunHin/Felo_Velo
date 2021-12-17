#!/bin/bash
# python main.py \
#     --clients_number 100 \
#     --clients_sample_ratio 0.1 \
#     --epoch 5000 \
#     --learning_rate 0.1 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0 \
#     --num_classes 10 \
#     --distorted_data \
#     --dataset mnist

# python main.py \
#     --clients_number 100 \
#     --clients_sample_ratio 0.1 \
#     --epoch 10000 \
#     --learning_rate 0.01 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0.6 \
#     --num_classes 10 \
#     --distorted_data \
#     --dataset cifar10

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --aggregate_flag \
    # --pre_training_flag \


# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --pre_training_flag \

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.0001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --pre_training_flag \


# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 1000 \
#     --learning_rate 0.0001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 128


python main.py \
    --GPU_num 3 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 100 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --beta 10 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --homo_flag
    # --resnet_flag \
    # --VAE \
    # --latent_size 1024

python main.py \
    --GPU_num 3 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 100 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --beta 10 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --homo_flag \
    --resnet_flag


# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --resnet_flag \
#     --VAE \
#     --latent_size 1024

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --GAN \
#     --pre_train \


# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --GAN \
#     --pre_train \


# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     # --BN_features \

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     # --BN_features \


# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --BN_features \
#     # --baseline_middle_flag

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 2 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --BN_features \


# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --public_flag \
#     --noniid_flag \
#     --homo_flag \
#     --pre_training_flag \

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --aggregate_flag \
#     --noniid_flag \
#     --homo_flag \

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --aggregate_flag \
#     --noniid_flag \
#     --homo_flag \

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --aggregate_flag \
#     --noniid_flag \
#     --homo_flag \

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.1 \
#     --epoch 50000 \
#     --learning_rate 0.001 \
#     --decay_rate 1 \
#     --batch_size 50 \
#     --clients_training_epoch 3 \
#     --alpha 1 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --noniid_flag \
#     --homo_flag \



