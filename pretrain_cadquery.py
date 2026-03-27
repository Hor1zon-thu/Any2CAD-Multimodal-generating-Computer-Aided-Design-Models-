import csv
import re
import json
import os

def clean_cadquery_code(code: str) -> str:
    """
    清洗CadQuery代码（修复正则后向断言问题）：
    1. 移除注释（保留行内注释但清理单独注释行）
    2. 移除空行
    3. 统一CadQuery导入方式为 import cadquery as cq
    4. 标准化字符串引号（统一为双引号）
    """
    # 步骤1：拆分代码行并初步清洗
    lines = code.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # 去除首尾空格
        line_stripped = line.strip()
        
        # 跳过空行和纯注释行
        if not line_stripped or line_stripped.startswith("#"):
            continue
        
        # 保留行内注释，拆分代码和注释部分（修复核心：分步处理）
        code_part = line_stripped
        comment_part = ""
        if "#" in line_stripped:
            split_parts = line_stripped.split("#", 1)
            code_part = split_parts[0].strip()
            comment_part = split_parts[1].strip()
        
        # 跳过纯注释的代码部分
        if not code_part:
            continue
        
        # 步骤2：处理代码部分的引号（仅处理代码部分，避免注释干扰）
        def replace_single_quotes(match):
            """替换单引号为双引号"""
            content = match.group(1)
            return f'"{content}"'
        
        # 仅匹配代码部分中的单引号字符串（无后向断言，避免可变长度问题）
        code_part = re.sub(r"'([^']*)'", replace_single_quotes, code_part)
        
        # 重组行（恢复注释）
        if comment_part:
            cleaned_line = f"{code_part} # {comment_part}"
        else:
            cleaned_line = code_part
        
        cleaned_lines.append(cleaned_line)
    
    # 步骤3：重组代码并处理导入语句
    cleaned_code = "\n".join(cleaned_lines)
    
    # 替换 import cadquery 为 import cadquery as cq
    cleaned_code = re.sub(r"^import cadquery$", "import cadquery as cq", cleaned_code, flags=re.MULTILINE)
    
    # 步骤4：确保开头有正确的导入语句（如果没有则添加）
    if cleaned_code and "import cadquery as cq" not in cleaned_code:
        cleaned_code = f"import cadquery as cq\n{cleaned_code}"
    
    return cleaned_code.strip()

def process_csv_to_jsonl(csv_path: str, output_jsonl_path: str = "cadquery_pretrain.jsonl"):
    """
    读取CSV文件，提取CadQuery代码并输出为JSONL文件
    :param csv_path: 输入的CSV文件路径（dataset_metadata.csv）
    :param output_jsonl_path: 输出的JSONL文件路径
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
    
    # 读取CSV并处理
    with open(csv_path, "r", encoding="utf-8") as csv_file, \
         open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        
        # 创建CSV阅读器
        reader = csv.DictReader(csv_file)
        
        # 验证CSV列是否包含content
        if "content" not in reader.fieldnames:
            raise ValueError("CSV文件缺少必要的'content'列")
        
        # 遍历每一行数据
        processed_count = 0
        error_count = 0
        skip_count = 0
        
        for row_idx, row in enumerate(reader, 1):
            try:
                # 提取content列的原始代码
                original_code = row.get("content", "").strip()
                
                # 跳过空内容
                if not original_code:
                    skip_count += 1
                    if row_idx % 1000 == 0:  # 每1000行打印一次进度
                        print(f"已处理{row_idx}行，跳过空行{skip_count}个，错误{error_count}个，有效{processed_count}个")
                    continue
                
                # 清洗代码
                cleaned_code = clean_cadquery_code(original_code)
                
                # 过滤无有效CadQuery操作的样本
                if cleaned_code and "cq.Workplane" in cleaned_code:
                    # 构建指定格式的JSON对象
                    json_obj = {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": cleaned_code
                            }
                        ]
                    }
                    # 写入JSONL文件（每行一个JSON）
                    jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    skip_count += 1
                
                # 进度打印
                if row_idx % 1000 == 0:
                    print(f"已处理{row_idx}行，跳过{skip_count}个，错误{error_count}个，有效{processed_count}个")
            
            except Exception as e:
                error_count += 1
                print(f"\n第{row_idx}行处理失败：{str(e)}")
                if error_count > 10:  # 最多打印10个错误，避免刷屏
                    if error_count == 11:
                        print("后续错误将不再打印...")
                continue
        
        print(f"\n===== 处理统计 =====")
        print(f"总行数：{row_idx}")
        print(f"有效数据：{processed_count} 条")
        print(f"跳过数据：{skip_count} 条")
        print(f"错误数据：{error_count} 条")
        print(f"\nJSONL文件已保存至：{os.path.abspath(output_jsonl_path)}")

# 主执行逻辑
if __name__ == "__main__":
    # 配置文件路径（可根据实际情况修改）
    INPUT_CSV = "dataset_metadata.csv"  # 输入的CSV文件名
    OUTPUT_JSONL = "cadquery_pretrain.jsonl"  # 输出的JSONL文件名
    
    # 执行处理
    try:
        process_csv_to_jsonl(INPUT_CSV, OUTPUT_JSONL)
        
        # 验证生成的JSONL文件（抽样验证前10行）
        print("\n===== 验证JSONL格式（前10行） =====")
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if idx > 10:
                    break
                try:
                    json.loads(line)
                    print(f"第{idx}行：格式正确")
                except json.JSONDecodeError as e:
                    print(f"第{idx}行：格式错误 - {e}")
    except Exception as e:
        print(f"\n程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()