import json
import sys

def convert_sample(sample):
    """
    将原始样本转换为目标格式
    """
    # 取 image（保持原样）
    images = sample.get("images", [])

    # 取 prompt 文本
    prompt = sample.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        content = prompt[0].get("content", "")
    else:
        content = ""

    # 去掉 <image> 占位符（如果存在）
    content = content.replace("<image>", "").strip()

    return {
        "images": images,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "ground_truth": sample.get("ground_truth", "")
    }

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            sample = json.loads(line)
            new_sample = convert_sample(sample)
            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_rl_dataset.py input.jsonl output.jsonl")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
