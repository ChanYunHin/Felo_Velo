python main.py \
    --GPU_num 3 \
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

python main.py \
    --GPU_num 3 \
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
    --homo_flag
    