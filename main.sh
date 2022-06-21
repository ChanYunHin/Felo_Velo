#Heterogeneous

# # ----------------------------old_CNN----------------------------
# # FedAE
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
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
#     --homo_flag

# # Private (FedAE)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
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
#     --homo_flag


# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --old_CNN \
#     --homo_flag

# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --old_CNN \
#     --homo_flag



# --------------------------------------------------------0.2sr--------------------------------------------------------

# # ----------------------------CNN----------------------------


# # Fedop
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --homo_flag \
#     --KLD_flag \


# # FedAE
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --homo_flag \
#     --KLD_flag \

# # Private (FedAE)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \
#     --homo_flag \
#     --KLD_flag \

# # Private (FedHe)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --homo_flag \
#     --KLD_flag \

# # FedHe
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --homo_flag \
#     --KLD_flag \

# ----------------------------ResNet----------------------------

# # FedNoise
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 1 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --NVAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --KLD_flag \

# # FedNoise
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 1 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --NVAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \


############################################################## 50 ##############################################################
# # Fedop

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 




# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --no_logits_flag 


# op_agg
python main.py \
    --GPU_num 2 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --epoch 100 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 5 \
    --alpha 1.0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --op_agg \
    --latent_size 1024 \
    --resnet_flag \
    --homo_flag \
    --no_logits_flag 





# # FedAE

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 


# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --no_logits_flag



# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 


# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 

############################################################## 64 ##############################################################
# # Fedop

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --no_logits_flag

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 



# # FedGKT

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --KLD_flag \
#     --homo_flag \
#     --GKT





# # FedAE

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --no_logits_flag


# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --new_VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag 




# # Private (FedHe)
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --KLD_flag


# # FedHe
# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --KLD_flag

# # FedMD

# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 64 \
#     --clients_training_epoch 5 \
#     --alpha 1.0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --MD


# # ----------------------------FedAvg----------------------------

# python main.py \
#     --GPU_num 0 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 1000 \
#     --learning_rate 0.001 \
#     --batch_size 128 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --latent_size 1024 \
#     --asyn_FL_flag \
#     --homo_flag \


# python main.py \
#     --GPU_num 2 \
#     --clients_number 10 \
#     --clients_sample_ratio 0.2 \
#     --epoch 2000 \
#     --learning_rate 0.001 \
#     --batch_size 128 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --latent_size 1024 \
#     --asyn_FL_flag \
#     --resnet_flag \
#     --homo_flag 

# # FedAE_ft

# python main.py \
#     --GPU_num 3 \
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
#     --new_VAE_ft \
#     --latent_size 1024\
#     --resnet_flag \
#     --homo_flag \
#     --KLD_flag \
#     --no_logits_flag


# python main.py \
#     --GPU_num 3 \
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
#     --new_VAE_ft \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --no_logits_flag


# python main.py \
#     --GPU_num 3 \
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
#     --new_VAE_ft \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \
#     --KLD_flag \

# python main.py \
#     --GPU_num 3 \
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
#     --new_VAE_ft \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag \



# # --------------------------------------------------------1sr--------------------------------------------------------

# # ----------------------------CNN----------------------------

# # FedOP
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --homo_flag


# # FedAE
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --homo_flag

# # Private (FedAE)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \
#     --homo_flag

# # Private (FedHe)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --homo_flag

# # FedHe
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --homo_flag

# # ----------------------------ResNet----------------------------


# # Fedop
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --op \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag


# # FedAE
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --beta 10 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --VAE \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag

# # Private (FedAE)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --beta 0 \
#     --cita 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --VAE \
#     --latent_size 1024 \
#     --y_distillation_flag \
#     --resnet_flag \
#     --homo_flag

# # Private (FedHe)
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag

# # FedHe
# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0.5 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --y_distillation_flag \
#     --latent_size 1024 \
#     --resnet_flag \
#     --homo_flag


# # # ----------------------------FedAvg----------------------------

# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --latent_size 1024 \
#     --asyn_FL_flag \
#     --homo_flag 

# python main.py \
#     --GPU_num 3 \
#     --clients_number 10 \
#     --clients_sample_ratio 1 \
#     --epoch 100 \
#     --learning_rate 0.001 \
#     --batch_size 50 \
#     --clients_training_epoch 10 \
#     --alpha 0 \
#     --num_classes 10 \
#     --dataset cifar10 \
#     --latent_size 1024 \
#     --asyn_FL_flag \
#     --resnet_flag \
#     --homo_flag 


