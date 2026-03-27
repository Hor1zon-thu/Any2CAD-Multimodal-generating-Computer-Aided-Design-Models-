from huggingface_hub import list_repo_files, hf_hub_download
import os

# 配置
REPO_ID = "pixparse/cc3m-wds"
LOCAL_DIR = "./cc3m_subset_part2"  # 建议存放在新目录，或者用同一个目录累加
START_INDEX = 60   # 第61个分片 (因为索引从0开始，所以填60)
END_INDEX = 120    # 第120个分片 (Python切片不包含结束位，所以填120会取到索引119)

def download_range():
    print(f"正在获取 {REPO_ID} 的文件列表...")
    
    # 1. 获取并筛选排序
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    tar_files = [f for f in all_files if f.endswith(".tar")]
    tar_files.sort()
    
    # 2. 截取 [60:120] 范围
    # 这里的切片逻辑是：包含 start_index，不包含 end_index
    subset_files = tar_files[START_INDEX:END_INDEX]
    
    if not subset_files:
        print("错误：指定范围内没有文件，请检查索引是否超出总数。")
        return

    print(f"检测到总分片数: {len(tar_files)}")
    print(f"即将下载索引 {START_INDEX} 到 {END_INDEX-1} (即第 {START_INDEX+1} 到 {END_INDEX} 个分片)")
    print(f"本批次数量: {len(subset_files)}")
    
    # 3. 循环下载
    for i, file_path in enumerate(subset_files):
        # 显示当前下载进度
        print(f"[{i+1}/{len(subset_files)}] 正在下载: {file_path} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

    print("\n下载完成！")

if __name__ == "__main__":
    download_range()