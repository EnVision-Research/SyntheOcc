export WANDB_DISABLED=True
export HF_HUB_OFFLINE=True

export MODEL_DIR="./ckp/stable-diffusion-v2-1"




export EXP_NAME="train_syntheocc"
export OUTPUT_DIR="./ckp/$EXP_NAME"
export SAVE_IMG_DIR="vis_dir/$EXP_NAME/samples"
export DATA_USED="samples_syntheocc_surocc"



# accelerate launch --gpu_ids 0,1,2,3,4,5,6,7  --num_processes 8  --main_process_port 3226  train.py \
accelerate launch --gpu_ids 0, --num_processes 1  --main_process_port 3226  train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --width=800 \
 --height=448 \
 --learning_rate=2e-5 \
 --num_train_epochs=6 \
 --train_batch_size=1 \
 --mixed_precision="fp16" \
 --num_validation_images=2 \
 --validation_steps=1000 \
 --checkpointing_steps=5000 \
 --checkpoints_total_limit=10 \
 --ctrl_channel=257 \
 --enable_xformers_memory_efficient_attention \
 --report_to='wandb' \
 --use_cbgs=True \
 --mtp_path='samples_syntheocc_surocc' \
 --resume_from_checkpoint="latest" 

