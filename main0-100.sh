#Heterogeneous

# # ----------------------------old_CNN----------------------------
# # FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --old_CNN \
#     --VAE \
#     --latent_size 1024 \
#   

# # Private (FedAE)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --old_CNN \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \
#   


# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --old_CNN \
#   

# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --old_CNN \
#   

# # --------------------------------------------------------0.2sr--------------------------------------------------------
# # ----------------------------CNN----------------------------

# # FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \

# # Private (FedAE)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \

# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \

# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \

# # ----------------------------ResNet----------------------------

# # FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --resnet_flag \

# # Private (FedAE)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \
#     --resnet_flag \

# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \

# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \


# --------------------------------------------------------1sr--------------------------------------------------------

# ----------------------------CNN----------------------------

# FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 200 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --VAE_size 1024 \

# # Private (FedAE)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 200 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --VAE \
#     --latent_size 1024 \
#     --VAE_size 1024 \
#     --y_distillation_flag \

# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 200 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --VAE_size 1024 \

# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 200 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --VAE_size 1024 \

# # ----------------------------ResNet----------------------------

# # FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 200 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 100 \
#     --dataset cifar100 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --VAE_size 1024 \
#     --resnet_flag \

# Private (FedAE)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 200 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --beta 0 \
    --cita 0 \
    --num_classes 100 \
    --dataset cifar100 \
    --VAE \
    --latent_size 1024 \
    --VAE_size 1024 \
    --y_distillation_flag \
    --resnet_flag \

# Private (FedHe)
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 200 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0 \
    --num_classes 100 \
    --dataset cifar100 \
    --y_distillation_flag \
    --latent_size 1024 \
    --VAE_size 1024 \
    --resnet_flag \

# FedHe
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 1 \
    --epoch 200 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 0.5 \
    --num_classes 100 \
    --dataset cifar100 \
    --y_distillation_flag \
    --latent_size 1024 \
    --VAE_size 1024 \
    --resnet_flag \
