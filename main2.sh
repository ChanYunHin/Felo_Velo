

############################################################# 32 #############################################################

# Fedop

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 3000 \
#     --learning_rate 0.001 \
#     --batch_size 32 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --KLD_flag \
#     --no_logits_flag

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --op \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag 


# FedMD

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --latent_size 1024 \
    --resnet_flag \
    --MD


# FedGKT

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \
    --GKT





# FedAE

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 3000 \
#     --learning_rate 0.001 \
#     --batch_size 32 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --KLD_flag \
#     --no_logits_flag


python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --new_VAE \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \




# Private (FedHe)
python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \


# FedHe
python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 3000 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \


############################################################# 16 #############################################################

# Fedop

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 4000 \
#     --learning_rate 0.001 \
#     --batch_size 16 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --KLD_flag \
#     --no_logits_flag

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --op \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag 


# FedMD

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --latent_size 1024 \
    --resnet_flag \
    --MD


# FedGKT

python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \
    --GKT





# FedAE

# python main.py \
#     --GPU_num 1 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 4000 \
#     --learning_rate 0.001 \
#     --batch_size 16 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --KLD_flag \
#     --no_logits_flag


python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --new_VAE \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \




# Private (FedHe)
python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \


# FedHe
python main.py \
    --GPU_num 1 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 4000 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --latent_size 1024 \
    --resnet_flag \
    --KLD_flag \