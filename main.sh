#!/bin/bash

#Heterogeneous

# ----------------------------old_CNN----------------------------
# FedAE
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --beta 10 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --old_CNN \
    --VAE \
    --latent_size 10 \

# Private (FedAE)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --old_CNN \
    --VAE \
    --latent_size 10 \
    --y_distillation_flag 

# Private (FedHe)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --old_CNN \

# FedHe
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --old_CNN \

# ----------------------------CNN----------------------------

# FedAE
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --beta 10 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --VAE \
    --latent_size 10 \

# Private (FedAE)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --VAE \
    --latent_size 10 \
    --y_distillation_flag 

# Private (FedHe)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \

# FedHe
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \


# ----------------------------ResNet----------------------------

# FedAE
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --beta 10 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --VAE \
    --latent_size 10 \
    --resnet_flag

# Private (FedAE)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --VAE \
    --latent_size 10 \
    --y_distillation_flag 
    --resnet_flag

# Private (FedHe)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --resnet_flag

# FedHe
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --resnet_flag


