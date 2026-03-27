CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model output/qwen2_5_vl_7b_coder_8b_stage2/v1-20251211-225059/checkpoint-9206 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --served_model_name Qwen3-VL