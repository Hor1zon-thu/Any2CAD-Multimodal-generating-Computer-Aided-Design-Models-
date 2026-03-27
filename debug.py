import sys
import os

# ============================================================
# 确保能 import 到你的 reward 插件
# ============================================================

sys.path.append(os.getcwd())

try:
    from grpo.plugin import run_code_and_export_step, CadQueryIoUReward
    print("✅ 成功导入 grpo.plugin 中的奖励函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确认：")
    print("1. 当前工作目录是项目根目录")
    print("2. grpo/plugin.py 文件存在")
    sys.exit(1)


# ============================================================
# 1. 构造测试代码
# ============================================================

# --- Ground Truth ---
gt_code = """
import cadquery as cq
wp = cq.Workplane("XY")
solid = wp.box(1, 1, 1)
"""

# --- 完美预测（应接近满分） ---
pred_code_perfect = """
import cadquery as cq
wp = cq.Workplane("XY")
solid = wp.box(1, 1, 1)
"""

# --- 尺寸略有偏差（应有 0.1 + IoU） ---
pred_code_ok = """
import cadquery as cq
wp = cq.Workplane("XY")
solid = wp.box(0.9, 0.9, 0.9)
"""

# --- 完全不可执行 ---
pred_code_bad = """
Here is the code you asked for.
"""

# --- 模拟真实模型输出（无 markdown、较复杂） ---
pred_code_real = """
import cadquery as cq

wp_sketch0 = cq.Workplane(
    cq.Plane(
        cq.Vector(-0.0625, 0.0, -0.0703125),
        cq.Vector(1.0, 6.123233995736766e-17, -6.123233995736766e-17),
        cq.Vector(6.123233995736766e-17, -1.0, 6.123233995736766e-17)
    )
)

loop0 = (
    wp_sketch0
    .moveTo(0.12434210526315789, 0.0)
    .lineTo(0.12434210526315789, 0.140625)
    .lineTo(0.0, 0.140625)
    .lineTo(0.0, 0.0)
    .close()
)

solid0 = wp_sketch0.add(loop0).extrude(0.75, both=True)
solid = solid0
"""


# ============================================================
# 2. 低层 sanity check：STEP 是否能生成
# ============================================================

def test_step_export():
    print("\n================ STEP 导出测试 ================")

    shape = run_code_and_export_step(gt_code, "test_gt")
    print("GT STEP 生成:", "✅ 成功" if shape is not None else "❌ 失败")

    shape = run_code_and_export_step(pred_code_bad, "test_bad")
    print("BAD STEP 生成:", "❌ 不应成功" if shape is not None else "✅ 正确失败")


# ============================================================
# 3. Reward 行为测试
# ============================================================

def test_reward():
    reward_engine = CadQueryIoUReward()

    print("\n================ Reward 测试 ================")

    # ---- 测试 1：完美匹配 ----
    print("\n[Test 1] 完美代码 vs GT")
    score = reward_engine.compute_score(
        pred_code_perfect, gt_code, idx=1
    )
    print(f"Score = {score:.4f}  (期望 > 1.5)")

    # ---- 测试 2：近似匹配 ----
    print("\n[Test 2] 略小 box vs GT")
    score = reward_engine.compute_score(
        pred_code_ok, gt_code, idx=2
    )
    print(f"Score = {score:.4f}  (期望 0.1 ~ 1.0)")

    # ---- 测试 3：垃圾代码 ----
    print("\n[Test 3] 垃圾代码 vs GT")
    score = reward_engine.compute_score(
        pred_code_bad, gt_code, idx=3
    )
    print(f"Score = {score:.4f}  (期望 = 0.0)")

    # ---- 测试 4：真实输出格式，自对齐 ----
    print("\n[Test 4] 真实模型输出 vs 自身")
    score = reward_engine.compute_score(
        pred_code_real, pred_code_real, idx=4
    )
    print(f"Score = {score:.4f}  (期望 > 1.5)")


# ============================================================
# 4. 主入口
# ============================================================

if __name__ == "__main__":
    print("🚀 开始 CadQuery IoU Reward 单元测试")
    test_step_export()
    test_reward()
