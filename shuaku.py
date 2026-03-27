import sys
sys.path.append(".")
import json
import requests
import os
import time
import random
import copy
import base64
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
 
class GeneralVision():
    """视觉中台通用访问类"""
    
    def __init__(self, 
                 job_name="ERINE_IMG_OCR_GPU_GENERAL_V6_NEW_ARCH_R200", 
                 token="4221c804-55ac-5c2b-847a-3394bbd84399",
                 feature_name='FEATURE_IMG_OCR_GPU_GENERAL_V6_NEW_ARCH_R200',
                 request_str=None,
                 threshold=0.5, 
                 logger=None):
        """初始化"""
        self.job_name = job_name
        self.token = token
        self.feature_name = feature_name
        self.request_str = request_str
        self.threshold = threshold
        self.feature_demo = XvisionDemo()
        
        self.logger = logger
        
    def prepare_request(self, data, new_request_str=None):
        """
        功能：构建算子的输入数据
        输入：
            data：dict类型，算子处理对象（image，video_url，audio等）以及必须的额外字段
        输出：
            返回算子的输入数据
        """
        request_str = self.request_str
        if new_request_str is not None:
            request_str = new_request_str
        
        base64image_str = str(base64.b64encode(data), 'utf-8')
         
        if request_str is None:
            new_data = json.dumps({"image": base64image_str}, cls=MyEncoder, \
                ensure_ascii=False, indent=4)
        else:
            new_data = f"{request_str}&image={base64image_str}"
        new_data = new_data.encode(encoding="utf-8")
 
        json_data = json.dumps({
                    'appid': '123456',
                    'logid': random.randint(1000000, 100000000),
                    'from': 'xvision',
                    'cmdid': '123',
                    'clientip': '0.0.0.0',
                    'data': base64.b64encode(new_data),
                }, cls=MyEncoder, ensure_ascii=False, indent=4)
        return json_data
 
    def preprocess_general_vision(self, image_bytes, new_request_str=None):
        """预处理图像"""
        feature_data = self.prepare_request(image_bytes, new_request_str)
        # 申请的作业名, 作业的token, 申请的特征名
        job_name, token, feature_name = self.job_name, self.token, self.feature_name  
        headers = {
            'Content-Type': 'application/json; charset=UTF-8',
            'resource_key': 'test.jpg',
            'auth_key': token,
            'business_name': job_name,
            'feature_name': feature_name,
            'X_BD_LOGID': str(random.randint(1000000, 100000000))
        }
        feature_data = self.prepare_request(image_bytes, new_request_str) 
 
        # 高可用型、均衡型作业：xvision_online_url，高吞吐型作业：xvision_offline_url，测试作业：xvision_test_url
        url = self.feature_demo.xvision_online_url
        url += self.feature_demo.xvision_sync_path
 
        # 高可用型、均衡型作业将job_name、feature_name放到 url 中
        params = {}
        if self.feature_demo.xvision_online_url in url:
            params = {
                "business_name": headers["business_name"],
                "feature_name": headers["feature_name"]
            }
        
        return params, feature_data, url, headers
 
    def post_baidu_general_vision(self, image_bytes, new_request_str=None):
        """
        功能：获取百度视觉中台通用服务结果
        输入：
            image_bytes:图像字节流
        输出：
            识别结果
        """
        # 预处理请求数据
        output = dict(status=-1)
        try:
            # 预处理
            params, feature_data, url, headers = \
                self.preprocess_general_vision(image_bytes, new_request_str)
            # 发送请求
            response = self.feature_demo.request_feat_new(params, feature_data, url, headers)
            response_data = json.loads(response)
            status = response_data.get("code", -1)
            if status != 0:
                return output
            
            # 解析结果
            ret = self.postprocess_general_vision(status, response_data)
            output["vision"] = ret.get("vision")
            
        except Exception as e:
            status = -1
            msg = f"GeneralVision request feat failed. error: {e}"
            self.logger.error(msg) if self.logger else print(msg)
        finally:
            output["status"] = status
            
        return output
 
    def postprocess_general_vision(self, status, response_data):
        """后处理"""
        output = {}
        try:
            feature_value = response_data.get("feature_result", {}).get("value", "{}")
            res_result = json.loads(feature_value).get("result", "")
            decode_result = base64.b64decode(res_result)
            decode_result_json = json.loads(decode_result)
            output["vision"] = decode_result_json
                
        except Exception as e:
            msg = f"GeneralVision parse json str failed. error: {e}"
            self.logger.error(msg) if self.logger else print(msg)
        
        return output
 
class BaseAPIServer(object):
    """API算子服务基类"""
    
    def __init__(self, config):
        """初始化"""
        self.config = config
        
    def set_serialized_callback(self, serialized_callback):
        """将格式化结果序列化为所需的字符串"""
        self.serialized_callback = serialized_callback
        
    def execute(self, *args, **kwargs):
        """执行服务"""
        raise NotImplementedError("execute is not defined now.")
 
 
class BaseGeneralVisionServer(BaseAPIServer, GeneralVision):
    """GeneralVision类型API服务"""
    
    def __init__(self, config, logger=None):
        """初始化API服务"""
        BaseAPIServer.__init__(self, config)
        self.job_name = self.config.get("job_name")
        self.token = self.config.get("token")
        self.feature_name = self.config.get("feature_name")
        self.request_str = self.config.get("request_str", None)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_sleep = self.config.get("retry_sleep", 1)
        self.logger = logger
 
        general_vision_params = {
            "job_name": self.job_name,
            "token": self.token,
            "feature_name": self.feature_name,
            "request_str": self.request_str,
            "logger": self.logger
        }
        
        GeneralVision.__init__(self, **general_vision_params)
        self.description = """GeneralVision类型API算子服务"""
 
 
class BaseLargeLanguageServer(BaseAPIServer):
    """LargeLanguage类型API请求"""
    
    def __init__(self, config, logger=None):
        """初始化API请求"""
        BaseAPIServer.__init__(self, config)
        self.server_url = self.config.get("server_url", "")
        self.model_name = self.config.get("model_name", "")
        self.top_p = self.config.get("top_p", 0.8)
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.system_prompt = self.config.get("system_prompt", "")
        self.query_prefix = self.config.get("query_prefix", "")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_sleep = self.config.get("retry_sleep", 1)
        self.logger = logger
 
        self.description = """LargeLanguage类型API请求"""
 
 
class BaseVisionLanguageServer(BaseAPIServer):
    """VisionLanguage类型API请求"""
    
    def __init__(self, config, logger=None):
        """初始化API请求"""
        BaseAPIServer.__init__(self, config)
        self.server_url = self.config.get("server_url", "")
        self.model_name = self.config.get("model_name", "")
        self.detail = self.config.get("detail", "auto")
        self.response_format = self.config.get("response_format", "")
        self.top_p = self.config.get("top_p", 0.8)
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.system_prompt = self.config.get("system_prompt", "")
        self.query_prefix = self.config.get("query_prefix", "")
        self.max_retries = int(self.config.get("max_retries", 3))
        self.retry_sleep = float(self.config.get("retry_sleep", 1))
        self.logger = logger
 
        self.description = """VisionLanguage类型API请求"""
 
 
class DoubaoServer(BaseVisionLanguageServer):
    """豆包API服务 (已修复图片为空时的崩溃问题)"""
    
    def __init__(self, config={}, logger=None):
        """初始化API服务"""
        self.config = copy.deepcopy(config)
        self.config["model_name"] = config.get("model_name", "doubao-1.5-vision-pro-32k")
        super().__init__(self.config, logger)
        self.api_key = config.get("api_key", "")
        self.logger = logger
        self.description = """Doubao请求API"""
 
    def execute(self, *args, **kwargs):
        """执行服务"""
        output, status = {}, -1
        start_t = time.time()
        url = ""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # 1. 获取并处理图片数据
            image_base64 = kwargs.get("image_base64", "")
            if image_base64 == "":
                image_bytes = kwargs.get("image_bytes", None)
                # 【修复点1】：必须检查 image_bytes 是否存在
                if image_bytes is not None:
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            content_prompt = kwargs.get("content_prompt", "")
            
            # 2. 动态构建 content 列表
            # 【修复点2】：如果没图片，就只发 text，不要发空的 image_url 结构
            content_payload = [
                {
                    "type": "text", 
                    "text": content_prompt
                }
            ]
            
            # 只有当成功获取到 image_base64 时，才添加图片部分
            if image_base64:
                content_payload.append({
                    "type": "image_url", 
                    "image_url": {'url': f"data:image/jpeg;base64,{image_base64}"},
                    "detail": self.detail,
                })

            # 3. 组装最终 Payload
            payload = { 
                "messages": [
                    {
                        "role": "user", 
                        "content": content_payload
                    }
                ],
                "model": self.model_name,
            }

            # 添加可选参数
            if float(self.temperature) > 0:
                payload["temperature"] = float(self.temperature)
            if float(self.top_p) > 0:
                payload["top_p"] = float(self.top_p)
            if self.response_format != "":
                payload["response_format"] = {"type": self.response_format}
 
            url = "http://" + self.server_url + "/v1/chat/completions/"
            
            # 发起请求
            response = requests.post(url, json=payload, headers=headers, timeout=1800)
            
            # 处理响应
            if response.status_code == 200:
                res_json = response.json()
                if 'choices' in res_json and len(res_json['choices']) > 0:
                    preds = res_json['choices'][0]['message']['content']
                    output["api_ret"] = preds
                    status = 0
                else:
                    print(f"⚠️ API返回格式异常: {res_json}")
            else:
                print(f"❌ HTTP错误 {response.status_code}: {response.text}")

        except Exception as e:
            status = -1
            msg = f"Error {self.__class__.__name__} failed: {e}."
            self.logger.error(msg) if self.logger else print(msg)
        finally:
            output["status"] = status
            output["req_time"] = time.time() - start_t
 
        return output
 
# ================= 基础工具函数 (保持原风格) =================
 
def load_jsonl(file_path):
    """加载 JSONL 文件"""
    entries = []
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
 
def save_jsonl(file_path, entries):
    """追加保存 JSONL 文件"""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
 
def read_image_content(image_path_or_url):
    """
    读取图像内容。
    兼容:
    1. 本地绝对路径 (/home/...)
    2. BOS 路径 (bos://...)
    3. HTTP 链接 (http://...)
    """
    BCECMD_PATH = "/home/disk2/guanzhicheng/script/linux-bcecmd-0.5.6/bcecmd"
 
    try:
        # === 情况 1: 本地文件 (新增支持) ===
        if image_path_or_url.startswith("/") and os.path.exists(image_path_or_url):
            with open(image_path_or_url, "rb") as f:
                return f.read()
 
        # === 情况 2: BOS 路径 ===
        if image_path_or_url.startswith("bos://"):
            temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
            temp_filepath = os.path.join(os.getcwd(), temp_filename)
            try:
                cmd = [BCECMD_PATH, "bos", "cp", image_path_or_url, temp_filepath]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if result.returncode == 0 and os.path.exists(temp_filepath):
                    with open(temp_filepath, "rb") as f:
                        content = f.read()
                    return content
                return None
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
 
        # === 情况 3: HTTP 链接 ===
        if image_path_or_url.startswith("http"):
            resp = requests.get(image_path_or_url, timeout=10)
            if resp.status_code == 200:
                return resp.content
                
        return None
    except Exception as e:
        print(f"❌ 读取异常 {image_path_or_url}: {e}")
        return None
 
def clean_json_response(text):
    """清洗 JSON 字符串"""
    if not text: return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(json)?', '', text)
        text = re.sub(r'```$', '', text)
    return text.strip()
 # ================= 核心生成逻辑 (修改后) =================

def process_single_cad_item(args):
    """
    处理单条 CadQuery 数据：
    1. 提取原始代码
    2. 让模型重构为硬编码、整数化格式
    3. 验证并替换
    """
    # 注意：这里参数签名变了，去掉了 max_text_count
    (item, server, base_prompt, max_retries) = args
    
    try:
        # === 1. 数据解析 (针对 GenCAD 格式) ===
        messages = item.get("messages", [])
        # 获取原始图片路径 (可选，作为辅助上下文)
        image_path = ""
        if item.get("images") and len(item["images"]) > 0:
            image_path = item["images"][0]
            
        # 提取 Assistant 的原始代码
        original_code = ""
        assistant_msg_index = -1
        
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                original_code = msg.get("content", "")
                assistant_msg_index = idx
                break
        
        if not original_code or assistant_msg_index == -1:
            # print("⚠️ 数据异常: 未找到 assistant 代码段")
            return None

        # 读取图片字节 (如果有)
        image_bytes = read_image_content(image_path)

        # === 2. 构造 Prompt ===
        # 将原始代码注入到 Prompt 模板中
        final_prompt = base_prompt.replace("{reference_code}", original_code)
        
        # === 3. 调用 API ===
        params = {
            "image_bytes": image_bytes,  # 传入图片辅助理解几何形状
            "content_prompt": final_prompt
        }
        
        attempt = 0
        output = None
        
        while attempt < max_retries:
            output = server.execute(**params)
            if output and output.get("status") == 0:
                break 
            attempt += 1
            time.sleep(1)

        if not output or output.get("status") != 0:
            return None

        # === 4. 解析结果 ===
        api_ret_text = output.get("api_ret", "")
        clean_json_str = clean_json_response(api_ret_text)
        
        try:
            res_json = json.loads(clean_json_str)
            
            # (A) 模型自检拒绝
            if res_json.get("status") == "invalid":
                reason = res_json.get("reason", "unknown")
                print(f"🚫 模型拒绝 (重构失败): {reason} | CodeLen: {len(original_code)}")
                return None

            # (B) 模型接受并返回代码
            if res_json.get("status") == "valid":
                new_code = res_json.get("label_code", "")
                if not new_code:
                    return None

                # === 5. 构造新数据 ===
                # 深拷贝原对象，只修改 content
                final_item = copy.deepcopy(item)
                final_item["messages"][assistant_msg_index]["content"] = new_code
                

                print(f"✅ 重构成功: 变量 {len(new_code)} chars")
                return final_item
                
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败: {clean_json_str[:50]}...")
            return None

    except Exception as e:
        print(f"❌ 处理异常: {e}")
        return None


def main_process_generation(input_jsonl, output_jsonl):
    
    # 1. 配置 (根据实际情况修改)
    # 1. 配置
    config = {
        "api_type": "doubao",
        "server_url": "yy.dbh.baidu-int.com",
        "model_name": "doubao-seed-1-6-251015", 
        "api_key": "sk-EIxiPo5jiRcLQsCCwvVQg9lZnCZ5Hul611MzQWN9D7q14Sov", # 【替换KEY】
        "thinking": {"type": "enabled"},
        "temperature": 0.1, 
        "max_retries": 3
    }
    server = DoubaoServer(config)

    # =========================================================
    # 2. 修改后的 Prompt：资深 CAD 工程师重构指令
    # =========================================================

    CAD_REFACTOR_PROMPT = """
You are a **Senior CAD Engineer**. Refactor the input CadQuery code into a "Linear Stacking" style.

【Input Data】
Reference Code: 
{reference_code} 

【Task 1: Modeling Strategy (CRITICAL)】
1.  **Linear Stacking Only:** You MUST build the model sequentially from bottom to top.
    * **Start:** `cq.Workplane('XY')...` (Base feature)
    * **Next Step:** Use `.faces('>Z').workplane()` to select the top face of the current object.
    * **Action:** Draw the next shape and `.extrude()` or `.hole()`.
2.  **FORBIDDEN Commands:** * **NO `.union()`**: Do not create separate bodies and union them. Build them in place.
    * **NO `.sketch()`**: Prefer direct 3D feature operations (`.circle().extrude()`, `.rect().hole()`) unless the 2D profile is extremely complex.
3.  **Coordinate Normalization:** Re-center model at Origin. Ignore absolute world coordinates.
    * **Infer Design Intent:** If cylinders or shapes are *nearly* concentric (e.g., center at 0.19 vs 0.34 but look like a single axis), **FORCE them to be concentric at (0,0)**. Correct any misalignment artifacts.

【Task 2: Code Syntax】
1.  **Fluent Chain Wrapper:** Wrap the whole chain in `result = ( ... )`.
2.  **Inline Integers:** Scale up small values (x100) and round to clean integers (e.g., 350, 50, 10). Do NOT use variables at the top.
3.  **Quote Style:** Use **single quotes** (`'`).

【Task 3: Example Pattern】
* *Bad (Union style):*
    ```python
    .extrude(10).union(cq.Workplane('XY')...) 
    ```
* *Good (Stacking style):*
    ```python
    .circle(100).extrude(10)      # Base
    .faces('>Z').workplane()      # Move to top
    .circle(50).extrude(20)       # Add Tower
    .faces('>Z').workplane()      # Move to top of Tower
    .hole(30)                     # Cut hole
    ```

【Output Format】
Return a pure JSON object.

If Accepted:
{
    "status": "valid",
    "label_code": "import cadquery as cq\\n\\nresult = (\\n    cq.Workplane('XY')\\n    # Base\\n    .circle(350).extrude(100)\\n    # Stack Cylinder\\n    .faces('>Z').workplane()\\n    .circle(200).extrude(750)\\n    ..."
}
    """
    # 3. 加载原始数据
    print(f"📂 正在加载数据: {input_jsonl}")
    all_items = load_jsonl(input_jsonl)
    total_count = len(all_items)
    print(f"📊 原始数据量: {total_count} 条")
 
    # =========================================================
    # 3.1 采样 (可选，这里设为 1.0 处理全部)
    # =========================================================
    SAMPLING_RATE = 1.0 
    sampled_items = [item for item in all_items if random.random() <= SAMPLING_RATE]
    
    # 4. 并发执行
    max_workers = 20 # 这里的并发数取决于你的 API 限流情况
    
    print(f"🚀 开始重构 CadQuery 代码 (并发: {max_workers})...")
    
    # 参数列表：去掉原有的 max_text_count，只传必要参数
    args_list = [(item, server, CAD_REFACTOR_PROMPT, 3) for item in sampled_items]
    
    success_count = 0
    buffer = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_cad_item, args) for args in args_list]
        
        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(sampled_items), unit="条"):
            res = future.result()
            if res:
                buffer.append(res)
                success_count += 1
            
            if len(buffer) >= 10:
                save_jsonl(output_jsonl, buffer)
                buffer = []
    
    if buffer:
        save_jsonl(output_jsonl, buffer)
 
    print("\n" + "="*50)
    print(f"✅ 全部完成！")
    print(f"📥 输入: {len(sampled_items)}")
    print(f"📤 成功重构: {success_count}")
    print(f"📁 结果文件: {output_jsonl}")
    print("="*50)
 
if __name__ == "__main__":
    #INPUT_FILE = "GenCAD-Code/train-00000-of-00002.swift.jsonl" 
    #OUTPUT_FILE = "GenCAD-Code/train-00-of-02-reconstruction.jsonl"
    INPUT_FILE = "GenCAD-Code/input_test.jsonl"
    OUTPUT_FILE="GenCAD-Code/test.jsonl"
    main_process_generation(INPUT_FILE, OUTPUT_FILE)