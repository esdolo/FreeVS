export NCCL_P2P_LEVEL=NVL
export MODEL_NAME="./pretrained/stable-video-diffusion-img2vid-xt" # SVD official
# export DATASET_NAME='waymo_process/waymo_multiframe_subsegbycampos.pkl'
export DATASET_NAME='waymo_process/waymo_multiframe_subsegbycampos_transform_simulation.pkl'
export HF_HOME="./huggingface"

WORK=freevs_waymo_halfreso_multiframe_transformation_simulate
EXP=freevs_waymo_halfreso_multiframe_transformation_simulate

#notice:initialize_pseudoimg_encoder=False when load ckpt

accelerate launch --config_file fsdp.yaml --main_process_port 12012 examples/freevs/train_svd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=15000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="work_dirs/$WORK" \
  --dataloader_num_workers 6 \
  --nframes 6 \
  --conditioning_dropout_prob=0.2 \
  --mismatch_aug_ratio=0.5 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 2500 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --checkpoints_total_limit 4 \
  --initialize_pseudoimg_encoder #\

  # --resume_from_checkpoint "latest" 
    # --gradient_checkpointing \
  # --report_to wandb \


