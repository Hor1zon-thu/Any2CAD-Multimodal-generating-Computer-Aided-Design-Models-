export CONDA_HOME=/root/swift/miniconda3/envs/swift
export CUDA_HOME=/root/cuda_12.4.0_cudnn_9.6.0

export PATH=$CONDA_HOME/external_pkgs/gcc-9.5/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/external_pkgs/gcc-9.5/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib64:$LD_LIBRARY_PATH
export PATH=$CONDA_HOME/bin:$CONDA_HOME/condabin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib/python3.9/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib64:$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CUTLASS_PATH=$CONDA_HOME/external_resources/cutlass/
export NCCL_ALGO=Ring
export NCCL_NVLS_ENABLE=0

IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model output/qwen2_5_vl_7b_coder_8b_stage1/v1-20251211-215219/checkpoint-1151 \
    --dataset GenCAD-Code/train-00000-of-00002.swift.jsonl \
              GenCAD-Code/train-00001-of-00002.swift.jsonl \
    --train_type full \
    --model_type qwen3_vl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --padding_free true \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/qwen2_5_vl_7b_coder_8b_stage2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --temperature 0.6 \
    --repetition_penalty 0.7 \
    --deepspeed zero3