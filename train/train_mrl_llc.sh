#!/bin/sh

# export CUDA_VISIBLE_DEVICES=0,1 && python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
# --data.train_dataset=../../train_500_0.50_90.ffcv --data.val_dataset=../../val_500_0.50_90.ffcv \
# --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
# --dist.world_size=2 --training.distributed=1 --lr.lr=0.2125 --dist.port=8088 --training.batch_size=1024 \
# --model.binarized_nesting_list='[6, 8, 16, 32]'

# export CUDA_VISIBLE_DEVICES=0,1 && python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
# --data.train_dataset=../../train_500_0.50_90.ffcv --data.val_dataset=../../val_500_0.50_90.ffcv \
# --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
# --dist.world_size=2 --training.distributed=1 --lr.lr=0.2125 --dist.port=8088 --training.batch_size=1024 \
# --model.binarized_nesting_list='[4, 8, 16, 32]'

# export CUDA_VISIBLE_DEVICES=0,1 && python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
# --data.train_dataset=../../train_500_0.50_90.ffcv --data.val_dataset=../../val_500_0.50_90.ffcv \
# --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
# --dist.world_size=2 --training.distributed=1 --lr.lr=0.2125 --dist.port=8088 --training.batch_size=1024 \
# --model.binarized_nesting_list='[8, 16, 32]'

export CUDA_VISIBLE_DEVICES=0,1 && python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
--data.train_dataset=../../train_500_0.50_90.ffcv --data.val_dataset=../../val_500_0.50_90.ffcv \
--data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
--dist.world_size=2 --training.distributed=1 --lr.lr=0.2125 --dist.port=8088 --training.batch_size=1024 \
--model.binarized_nesting_list='[]' --model.mrl_nesting_list='[32, 2048]'

# export CUDA_VISIBLE_DEVICES=0,1 && python train_imagenet.py --config-file rn50_configs/rn50_40_epochs.yaml --model.mrl=1 \
# --data.train_dataset=../../train_500_0.50_90.ffcv --data.val_dataset=../../val_500_0.50_90.ffcv \
# --data.num_workers=12 --data.in_memory=1 --logging.folder=trainlogs --logging.log_level=1 \
# --dist.world_size=2 --training.distributed=1 --lr.lr=0.2125 --dist.port=8088 --training.batch_size=1024 \
# --model.binarized_nesting_list='[32]' --model.mrl_nesting_list='[2048]'