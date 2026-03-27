import json
import os
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条

def process_cc3m_to_jsonl(source_dir, output_file):
    """
    读取文件夹中的txt和jpg，生成多模态训练用的jsonl文件。
    """
    # 转换为Path对象
    source_path = Path(source_dir)
    
    # 检查源文件夹是否存在
    if not source_path.exists():
        print(f"错误: 文件夹 '{source_dir}' 不存在。")
        return

    # 获取所有txt文件列表
    print(f"正在扫描文件夹 '{source_dir}' ...")
    txt_files = list(source_path.glob("*.txt"))
    total_files = len(txt_files)
    
    print(f"找到 {total_files} 个文本文件，开始处理...")

    # 打开输出文件准备写入
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 使用tqdm显示进度
        for txt_file in tqdm(txt_files, desc="Processing"):
            
            # 推导对应的图片路径 (同名，后缀改为.jpg)
            img_file = txt_file.with_suffix('.jpg')
            
            # 只有当对应的图片存在时才处理
            if img_file.exists():
                try:
                    # 读取txt内容
                    with open(txt_file, 'r', encoding='utf-8') as f_txt:
                        content = f_txt.read().strip()
                    
                    # 如果文本为空，可以选择跳过，或者保留空字符串
                    # if not content: continue 

                    # 构建数据结构
                    data = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": "<image>Give a brief description of the image"
                            },
                            {
                                "role": "assistant", 
                                "content": content
                            }
                        ],
                        # 将Path对象转换为字符串路径
                        # 注意：这里会保留 source_dir 的前缀，例如 "cc3m_500k_unpacked/000000.jpg"
                        "images": [str(img_file)]
                    }
                    
                    # 写入一行JSONL (ensure_ascii=False 保证中文正常显示，虽然CC3M主要是英文)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    
                except Exception as e:
                    print(f"\n处理文件 {txt_file.name} 时出错: {e}")
            else:
                # 如果找不到对应的图片，可以选择打印日志
                # print(f"警告: 图片缺失 {img_file}")
                pass

    print(f"\n处理完成！")
    print(f"共处理 {count} 条有效数据。")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 输入文件夹路径
    INPUT_FOLDER = "cc3m_500k_unpacked"
    
    # 输出文件名
    OUTPUT_FILE = "cc3m_dataset.jsonl"
    # ----------------
    
    process_cc3m_to_jsonl(INPUT_FOLDER, OUTPUT_FILE)