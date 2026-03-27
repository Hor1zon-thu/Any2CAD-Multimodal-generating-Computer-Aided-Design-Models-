export CONDA_HOME=/root/swift/miniconda3/envs/swift
export CUDA_HOME=/root/cuda_12.4.0_cudnn_9.6.0

export PATH=$CONDA_HOME/external_pkgs/gcc-9.5/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/external_pkgs/gcc-9.5/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib64:$LD_LIBRARY_PATH
export PATH=$CONDA_HOME/bin:$CONDA_HOME/condabin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib/python3.9/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib64:$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CUTLASS_PATH=$CONDA_HOME/external_resources/cutlass/
export NCCL_ALGO=Ring
export NCCL_NVLS_ENABLE=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
--rlhf_type grpo \
--model qwen2_5_vl_7b \
--model_type qwen2_5_vl \
--train_type full \
--dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
--torch_dtype bfloat16 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 1e-6 \
--save_total_limit 2 \
--logging_steps 5 \
--output_dir output/qwen2_5_vl_7b_coder_8b_stage3_grpo \
--gradient_accumulation_steps 1 \
--warmup_ratio 0.05 \
--dataloader_num_workers 4 \
--max_completion_length 1024 \
--vllm_max_model_len 1024 \
--reward_funcs accuracy \
--num_generations 8 \
--use_vllm true \
--vllm_mode colocate \
--vllm_gpu_memory_utilization 0.6 \
--sleep_level 1 \
--offload_model true \
--offload_optimizer true \
--vllm_tensor_parallel_size 1 \
--temperature 1.0 \
--top_p 0.85 \
--deepspeed zero3 \
--log_completions true \
--overlong_filter true \
--split_dataset_ratio 0 \