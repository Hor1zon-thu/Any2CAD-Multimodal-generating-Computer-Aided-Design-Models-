import json
import os
from pathlib import Path
from tqdm import tqdm

def process_cc3m_pretrain_format(source_dir, output_file):
    """
    读取文件夹中的txt和jpg，生成预训练格式的jsonl文件。
    格式: {"messages": [{"role": "assistant", "content": "<image>text..."}], "images": ["path/to/img"]}
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"错误: 文件夹 '{source_dir}' 不存在。")
        return

    print(f"正在扫描文件夹 '{source_dir}' ...")
    txt_files = list(source_path.glob("*.txt"))
    print(f"找到 {len(txt_files)} 个文件，开始处理...")

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for txt_file in tqdm(txt_files, desc="Converting"):
            
            # 获取对应的图片路径
            img_file = txt_file.with_suffix('.jpg')
            
            if img_file.exists():
                try:
                    # 读取文本内容
                    with open(txt_file, 'r', encoding='utf-8') as f_txt:
                        text_content = f_txt.read().strip()
                    
                    # 构建新的数据结构
                    data = {
                        "messages": [
                            {
                                "role": "assistant",
                                # 将 <image> 标签直接拼接到文本前面
                                "content": f"<image>{text_content}"
                            }
                        ],
                        # 图片路径
                        "images": [str(img_file)]
                    }
                    
                    # 写入 JSONL
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    
                except Exception as e:
                    print(f"处理出错: {txt_file.name} - {e}")

    print(f"\n处理完成！共生成 {count} 条数据。")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # --- 配置 ---
    INPUT_FOLDER = "cc3m_500k_unpacked_2"
    OUTPUT_FILE = "cc3m_pretrain2.jsonl"
    # -----------
    
    process_cc3m_pretrain_format(INPUT_FOLDER, OUTPUT_FILE)