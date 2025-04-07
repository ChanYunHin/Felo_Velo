
# Felo
python main.py \
    --GPU_num 0 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --VAE_training_interval 10 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 1.0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --Felo \
    --latent_size 1024 \
    --KLD_flag \
    --resnet_flag 


# Velo
python main.py \
    --GPU_num 0 \
    --clients_number 10 \
    --clients_sample_ratio 0.2 \
    --VAE_training_interval 10 \
    --epoch 1000 \
    --learning_rate 0.001 \
    --batch_size 50 \
    --clients_training_epoch 10 \
    --alpha 1.0 \
    --num_classes 10 \
    --dataset cifar10 \
    --y_distillation_flag \
    --Velo \
    --latent_size 1024 \
    --KLD_flag \
    --resnet_flag 
