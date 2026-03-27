export CONDA_HOME=/root/miniconda3/envs/eb_doc_vllm
export CUDA_HOME=/root/cuda-12.8/

export PATH=$CONDA_HOME/external_pkgs/gcc-9.5/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/external_pkgs/gcc-9.5/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib64:$LD_LIBRARY_PATH
export PATH=$CONDA_HOME/bin:$CONDA_HOME/condabin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib/python3.11/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib64:$CONDA_HOME/lib:$LD_LIBRARY_PATH

export CUTLASS_PATH=$CONDA_HOME/external_resources/cutlass/
export NCCL_ALGO=Ring
export NCCL_NVLS_ENABLE=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model qwen3_vl_4b\
    --model_type qwen3_vl \
    --stream true \
    --infer_backend pt\
    --max_new_tokens 2048 \
    --result_path ref_result.json \