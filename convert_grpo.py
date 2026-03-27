import json

input_file = "GenCAD-Code/train-00000-of-00002.swift.jsonl"
output_file = "rl_grpo_data.jsonl"

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        
        # 提取 User Prompt
        user_msg = next(m for m in data['messages'] if m['role'] == 'user')
        
        # 提取 Ground Truth Code
        assistant_msg = next(m for m in data['messages'] if m['role'] == 'assistant')
        
        new_record = {
            # GRPO 需要的 Prompt 结构
            "prompt": [user_msg], 
            # 图片列表保持不变
            "images": data['images'],
            # 专门给 Reward Model 看的标准答案
            "ground_truth": assistant_msg['content']
        }
        
        f_out.write(json.dumps(new_record) + '\n')