
export MODEL_DIR="./ckp/stable-diffusion-v2-1"

export EXP_NAME="train_syntheocc"
export OUTPUT_DIR="./ckp/$EXP_NAME"
export SAVE_IMG_DIR="vis_dir/$EXP_NAME/samples"
export DATA_USED="samples_syntheocc_surocc"

export TRAIN_OR_VAL="val"


CUDA_VISIBLE_DEVICES=0 python infer.py  --save_img_path=$SAVE_IMG_DIR  --model_path_infer=$OUTPUT_DIR  --curr_gpu=0  --mtp_path=$DATA_USED --ctrl_channel=257 --gen_train_or_val=$TRAIN_OR_VAL
