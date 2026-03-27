# ============================================================
# CadQuery IoU Reward - V8 (DIRECT ARGUMENT VERSION)
# ============================================================
import cadquery as cq
import numpy as np
import os, sys, subprocess, tempfile, uuid, re, datetime, traceback
from typing import List, Optional, Any
from scipy.spatial import cKDTree
from swift.plugin.orm import ORM, orms

# ============================================================
# Debug Configuration
# ============================================================
LOG_FILE = "grpo_debug.log"

def log_msg(msg: str):
    """同时打印到屏幕和日志文件"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    # print(f"[Reward] {msg}") # 可选：取消注释以在控制台实时看
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")

# ============================================================
# 1. Text & Code Extraction
# ============================================================
def extract_python_code(text: str) -> str:
    # 策略 A: Markdown
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1)
    
    # 策略 B: 裸代码锚点
    markers = ["import cadquery", "import cq", "from cadquery"]
    start_idx = -1
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            if start_idx == -1 or idx < start_idx: start_idx = idx
    if start_idx != -1: return text[start_idx:]
    
    return text

# ============================================================
# 2. Execution Sandbox (Subprocess)
# ============================================================
def run_code_and_export_step(raw_text: str, prefix: str) -> str:
    code_str = extract_python_code(raw_text)
    if not code_str.strip(): return None

    try: compile(code_str, "<string>", "exec")
    except SyntaxError: return None
        
    code = code_str.replace("\\n", "\n")
    indented = "\n".join("    " + l for l in code.splitlines())

    td = tempfile.mkdtemp()
    step_path = os.path.join(td, f"{prefix}.step")
    py_path = os.path.join(td, f"{prefix}.py")

    with open(py_path, "w") as f:
        f.write(f"""
import sys, os, warnings
warnings.filterwarnings("ignore")
try:
    import cadquery as cq
    # 用户代码
{indented}
    
    # 提取对象
    obj = None
    # 增加 solid0 以适配你的数据集真实输出
    target_names = ['solid', 'result', 'part', 'assembly', 'solid0']
    for name in target_names:
        if name in locals() and hasattr(locals()[name], 'val'):
            obj = locals()[name]; break
            
    if obj is None:
        cands = [v for v in locals().values() if hasattr(v, 'val')]
        if cands: obj = cands[-1]
    
    if obj:
        if hasattr(obj, 'toShape'): obj = cq.Workplane(obj=obj.toShape())
        elif hasattr(obj, 'wrapped'): obj = cq.Workplane(obj=obj)
        cq.exporters.export(obj, r"{step_path}")
    else:
        sys.exit(2)
except Exception:
    sys.exit(1)
""")

    try:
        # [关键] 必须传递环境变量
        subprocess.run(
            [sys.executable, py_path], 
            timeout=20, 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            env=os.environ.copy()
        )
        return step_path
    except:
        return None

# ============================================================
# 3. IoU Calculation
# ============================================================
def calc_iou(pred_step, gt_step):
    if not pred_step or not gt_step: return 0.0, 0.0
    try:
        pred = cq.importers.importStep(pred_step)
        gt = cq.importers.importStep(gt_step)
        
        ba = pred.val().BoundingBox()
        bb = gt.val().BoundingBox()
        
        # BBox IoU
        dx = max(0, min(ba.xmax, bb.xmax) - max(ba.xmin, bb.xmin))
        dy = max(0, min(ba.ymax, bb.ymax) - max(ba.ymin, bb.ymin))
        dz = max(0, min(ba.zmax, bb.zmax) - max(ba.zmin, bb.zmin))
        inter = dx * dy * dz
        union = (ba.xlen * ba.ylen * ba.zlen) + (bb.xlen * bb.ylen * bb.zlen) - inter
        iou_bbox = inter / union if union > 1e-9 else 0.0
        
        # Voxel IoU
        n = 1024
        def sample(box):
            pts = np.random.rand(n, 3)
            pts[:, 0] = pts[:, 0] * (box.xmax - box.xmin) + box.xmin
            pts[:, 1] = pts[:, 1] * (box.ymax - box.ymin) + box.ymin
            pts[:, 2] = pts[:, 2] * (box.zmax - box.zmin) + box.zmin
            return pts
        pts_a, pts_b = sample(ba), sample(bb)
        
        # 动态 Epsilon
        diag = np.sqrt(ba.xlen**2 + ba.ylen**2 + ba.zlen**2) + np.sqrt(bb.xlen**2 + bb.ylen**2 + bb.zlen**2)
        eps = max(1e-4, (diag / 2) * 0.1)
        
        ta, tb = cKDTree(pts_a), cKDTree(pts_b)
        match = (ta.query(pts_b)[0] < eps).sum() + (tb.query(pts_a)[0] < eps).sum()
        iou_voxel = match / (2 * n)
        
        return iou_bbox, iou_voxel
    except:
        return 0.0, 0.0

# ============================================================
# 4. Reward Class (Correct Signature)
# ============================================================
class CadQueryIoUReward(ORM):
    """
    CadQuery IoU 奖励函数
    
    输入：
        completions: 模型生成的代码列表
        ground_truth: 数据集中的真值代码列表 (直接传递)
    
    输出：
        分数列表 [0.0, 2.0]
    """
    
    def __init__(self):
        super().__init__()
        self._debug_checked = False
    
    def __call__(
        self,
        completions: List[str],
        ground_truth: List[str],  # <--- ✅ 按照你的要求直接接收 ground_truth
        **kwargs
    ) -> List[float]:
        
        scores = []
        
        # --- 诊断信息 ---
        if not self._debug_checked:
            log_msg(f"✅ __call__ invoked successfully!")
            log_msg(f"✅ ground_truth received type: {type(ground_truth)}")
            if ground_truth and len(ground_truth) > 0:
                log_msg(f"✅ ground_truth[0] preview: {ground_truth[0][:50]}...")
            else:
                log_msg(f"❌ ground_truth is empty or None!")
            self._debug_checked = True
        # ----------------
        
        # 确保 ground_truth 不为空
        if not ground_truth:
            log_msg("⚠️ Warning: ground_truth argument is empty.")
            return [0.0] * len(completions)

        # 遍历批次进行计算
        for i, (pred_text, gt_text) in enumerate(zip(completions, ground_truth)):
            score = self.compute_score(pred_text, gt_text, i)
            scores.append(score)
            
        return scores

    def compute_score(self, pred_code: str, gt_code: str, idx: int) -> float:
        # 1. 低保逻辑 (0.1分) - 只要有 import 就给分，防止前期完全学不到
        bootstrap = 0.0
        if "import cadquery" in pred_code or "import cq" in pred_code:
            bootstrap = 0.1
            
        uid = uuid.uuid4().hex[:4]
        
        # 2. GT 生成 (如果 GT 坏了，直接 0 分)
        gt_path = run_code_and_export_step(gt_code, f"gt_{uid}")
        if not gt_path: 
            return 0.0 
        
        # 3. Pred 生成
        pred_path = run_code_and_export_step(pred_code, f"pred_{uid}")
        
        if pred_path and idx < 2:
            log_msg(f"✅ Sample {idx}: Code Executed Successfully!")

        if not pred_path:
            return bootstrap

        # 4. 计算 IoU
        iou_bbox, iou_voxel = calc_iou(pred_path, gt_path)
        
        # 清理
        if os.path.exists(gt_path): os.remove(gt_path)
        if os.path.exists(pred_path): os.remove(pred_path)

        # 5. 评分公式
        reward = 0.1 # 基础生存分
        
        # 如果 BBox 都差很远，惩罚但保留低保
        if iou_bbox < 0.1: 
            return float(reward + iou_bbox * 0.5 + bootstrap)
            
        reward += iou_voxel
        if iou_voxel > 0.6: reward += 0.2
        if iou_voxel > 0.8: reward += 0.3
        
        return float(max(0.0, min(2.0, reward + bootstrap)))

# ============================================================
# Register
# ============================================================
orms["cad_iou_reward"] = CadQueryIoUReward