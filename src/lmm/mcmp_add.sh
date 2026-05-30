#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python main_vlm.py \
  --model_name llava_onevision \
  --model_args "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov" \
  --calib_dataset coco \
  --data_path "/root/autodl-tmp/hf_home/datasets/coco/sharegpt4v_coco_only.json" \
  --image_folder "/root/autodl-tmp/hf_home/datasets" \
  --quant_type weight_only \
  --wbits 3 \
  --k_bits 16 \
  --v_bits 16 \
  --train_size 128 \
  --epochs 15 \
  --rank_multiplier 16 \
  --clip_lr 2e-1 \
  --weight_lr 1e-4 \
  --topk_token 0.5 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --output_dir "./log/llava_ov_w3a16/" \
  --output_path "./output/llava_ov_w3a16/"