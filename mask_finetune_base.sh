# export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
NUM_GPUS=4

export LOGDIR=./logs/coco-mask/coco-64-stage1/
export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--learn_sigma True --uncond_p 0. --image_size 64 --finetune_decoder False"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 28  --schedule_sampler loss-second-moment  --model_path ./ckpt/base.pt --lr_anneal_steps 20000"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train.txt --val_data_dir ./dataset/COCOSTUFF_test.txt --mode ade20k"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS

export LOGDIR=./logs/coco-mask/coco-64-stage1-cont/
export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder False --encoder_path ./logs/coco-mask/coco-64-stage1/checkpoints/ema_0.9999_020000.pt"
TRAIN_FLAGS="--lr 2e-5 --batch_size 28  --schedule_sampler loss-second-moment  --model_path ./ckpt/base.pt --lr_anneal_steps 15000"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1.5"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train.txt --val_data_dir ./dataset/COCOSTUFF_test.txt --mode ade20k"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS

export LOGDIR=./logs/coco-mask/coco-64-stage2-decoder/
export PYTHONPATH=$PYTHONPATH:$(pwd)
# NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder True"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 28 --schedule_sampler loss-second-moment --model_path ./ckpt/base.pt --encoder_path ./logs/coco-mask/coco-64-stage1-cont/checkpoints/ema_0.9999_015000.pt --lr_anneal_steps 15000"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 1.5"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_train.txt --val_data_dir ./dataset/COCOSTUFF_test.txt --mode ade20k"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
