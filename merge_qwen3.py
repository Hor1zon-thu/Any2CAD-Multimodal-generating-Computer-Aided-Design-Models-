import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger, Qwen3VLModel
from accelerate import Accelerator

# 加载原始 VL 模型和 Qwen3-0_6b 模型
qwen3_vl_2b_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "qwen3_vl_2b",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
device = qwen3_vl_2b_model.device
qwen3_0_6b = AutoModelForCausalLM.from_pretrained(
    "qwen3_0_6b",
    device_map='auto',
    torch_dtype=torch.bfloat16
)

# 加载配置
old_config = AutoConfig.from_pretrained("qwen3_vl_2b")
new_config = AutoConfig.from_pretrained("qwen3_vl_2b_encoder_0_6b") # 新 config 的文件夹路径
new_visual_config = new_config.vision_config

# 1. 替换 ViT 到 LLM 的 aligner层
new_merger = Qwen3VLVisionPatchMerger(
    config = new_visual_config
).to(device).to(torch.bfloat16)
qwen3_vl_2b_model.visual.merger = new_merger

# 2. 替换 VL 模型的 LLM 部分
new_llm_model = Qwen3VLModel(new_config).to(device).to(torch.bfloat16)

for name, param in qwen3_0_6b.model.named_parameters():
    if name in new_llm_model.state_dict():
        new_llm_model.state_dict()[name].copy_(param)

qwen3_vl_2b_model.model = new_llm_model
qwen3_vl_2b_model.lm_head = qwen3_0_6b.lm_head

# 3. 保存修改后的模型
accelerator = Accelerator()
accelerator.save_model(
model=qwen3_vl_2b_model,
save_directory="qwen3_vl_2b_encoder_0_6b",
max_shard_size="5GB",
safe_serialization=True
)