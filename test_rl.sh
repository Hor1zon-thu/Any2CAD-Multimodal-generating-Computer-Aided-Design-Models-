export CONDA_HOME=/root/miniconda3/envs/eb_doc_vllm
export CUDA_HOME=/root/cuda-12.8/

export PATH=$CONDA_HOME/external_pkgs/gcc-9.5/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/external_pkgs/gcc-9.5/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib64:$LD_LIBRARY_PATH
export PATH=$CONDA_HOME/bin:$CONDA_HOME/condabin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib/python3.11/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib64:$CONDA_HOME/lib:$LD_LIBRARY_PATH

export CUTLASS_PATH=$CONDA_HOME/external_resources/cutlass/
export NCCL_ALGO=Ring
export NCCL_NVLS_ENABLE=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_ATTENTION_BACKEND=FLASHINFER
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
  --rlhf_type grpo \
  --model output/qwen2_5_vl_7b_coder_8b_stage2/v1-20251211-225059/checkpoint-9206 \
  --model_type qwen3_vl \
  --train_type lora \
  --dataset grpo_dataset.jsonl \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --reward_funcs cad_iou_reward \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_enforce_eager true \
  --vllm_gpu_memory_utilization 0.25 \
  --vllm_tensor_parallel_size 1 \
  --max_model_len 3000 \
  --max_completion_length 2048 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 9 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --num_generations 9 \
  --temperature 1.0 \
  --top_p 0.85 \
  --external_plugins grpo/plugin.py \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2 \
  --output_dir output/model_rl \
  --dataloader_num_workers 4 \
  --log_completions true \
  --offload_optimizer true \
  --offload_model true \
  --sleep_level 1 \
  --seed 42 
