import json
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
LOG_FILE = 'output/model_rl/v2-20251228-155235/logging.jsonl'  # 替换成你的日志文件名
SMOOTH_WINDOW = 5            # 移动平均的窗口大小，越大越平滑
# ===========================================

def moving_average(data, window_size):
    """计算移动平均线"""
    if len(data) < window_size:
        return data
    return [sum(data[i:i+window_size])/window_size for i in range(len(data)-window_size+1)]

def parse_log(file_path):
    steps = []
    losses = []
    rewards = []
    iou_rewards = []
    kls = []
    grad_norms = []
    lengths = []
    lrs = []

    print(f"正在读取文件: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                # 尝试解析 JSON
                entry = json.loads(line)
                
                # 提取 Step (格式如 "515/4602")
                if "global_step/max_steps" in entry:
                    step_str = entry["global_step/max_steps"]
                    current_step = int(step_str.split('/')[0])
                else:
                    # 如果没有 step 信息，就用列表索引代替
                    current_step = len(steps) + 1

                # 提取各个指标，使用 .get() 防止某些行缺失数据
                loss = entry.get("loss")
                reward = entry.get("reward")
                # 注意：这里专门提取你的 CadQueryIoUReward
                iou_reward = entry.get("rewards/CadQueryIoUReward/mean") 
                kl = entry.get("kl")
                grad_norm = entry.get("grad_norm")
                length = entry.get("completions/mean_length")
                lr = entry.get("learning_rate")

                # 只有当关键数据存在时才记录
                if loss is not None:
                    steps.append(current_step)
                    losses.append(loss)
                    rewards.append(reward if reward is not None else 0)
                    iou_rewards.append(iou_reward if iou_reward is not None else 0)
                    kls.append(kl if kl is not None else 0)
                    grad_norms.append(grad_norm if grad_norm is not None else 0)
                    lengths.append(length if length is not None else 0)
                    lrs.append(lr if lr is not None else 0)

            except json.JSONDecodeError:
                print(f"跳过非 JSON 行: {line[:50]}...")
                continue

    return {
        "steps": steps,
        "loss": losses,
        "reward": rewards,
        "iou_reward": iou_rewards,
        "kl": kls,
        "grad_norm": grad_norms,
        "length": lengths,
        "lr": lrs
    }

def plot_metrics(data):
    if not data["steps"]:
        print("未找到有效数据，请检查日志格式。")
        return

    # 设置绘图风格
    plt.style.use('bmh') # 使用一种比较干净的风格
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('RLHF Training Metrics (CadQuery Code Gen)', fontsize=16)

    # 辅助绘图函数
    def plot_sub(ax, x, y, title, color, ylabel, add_smooth=True):
        ax.plot(x, y, color=color, alpha=0.3, label='Raw')
        if add_smooth and len(y) > SMOOTH_WINDOW:
            smooth_y = moving_average(y, SMOOTH_WINDOW)
            # 对齐 x 轴
            smooth_x = x[SMOOTH_WINDOW-1:]
            ax.plot(smooth_x, smooth_y, color=color, linewidth=2, label=f'Smooth (w={SMOOTH_WINDOW})')
        ax.set_title(title)
        ax.set_xlabel('Global Step')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)

    # 1. Reward (总奖励 vs IoU 奖励)
    axs[0, 0].plot(data["steps"], data["reward"], label="Total Reward", color='tab:blue', alpha=0.6)
    axs[0, 0].plot(data["steps"], data["iou_reward"], label="IoU Reward", color='tab:green', linestyle='--', alpha=0.8)
    axs[0, 0].set_title("Rewards Trend")
    axs[0, 0].set_xlabel("Global Step")
    axs[0, 0].set_ylabel("Score")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. KL Divergence
    plot_sub(axs[0, 1], data["steps"], data["kl"], "KL Divergence", 'tab:red', "KL")
    # 添加一条参考线，通常 KL > 0.2 就危险了
    axs[0, 1].axhline(y=0.15, color='gray', linestyle=':', label='Warn Threshold')

    # 3. Loss
    plot_sub(axs[0, 2], data["steps"], data["loss"], "Training Loss", 'tab:orange', "Loss")

    # 4. Completion Length
    plot_sub(axs[1, 0], data["steps"], data["length"], "Mean Completion Length", 'tab:purple', "Tokens")

    # 5. Gradient Norm
    plot_sub(axs[1, 1], data["steps"], data["grad_norm"], "Gradient Norm", 'tab:brown', "Norm")

    # 6. Learning Rate
    axs[1, 2].plot(data["steps"], data["lr"], color='tab:pink')
    axs[1, 2].set_title("Learning Rate")
    axs[1, 2].set_xlabel("Global Step")
    axs[1, 2].set_ylabel("LR")
    axs[1, 2].grid(True)
    # 使用科学计数法显示 LR
    axs[1, 2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局给总标题留位置
    
    save_name = 'training_metrics.png'
    plt.savefig(save_name, dpi=300)
    print(f"绘图完成！图片已保存为: {save_name}")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        # 如果找不到文件，创建一个假文件用于测试（方便你直接运行代码查看效果）
        print(f"错误: 找不到文件 {LOG_FILE}。请修改脚本中的 LOG_FILE 变量。")
    else:
        data = parse_log(LOG_FILE)
        plot_metrics(data)