import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
import io

# --- 页面配置 ---
st.set_page_config(page_title="多模态CAD模型生成助手", layout="wide")

st.title("🖼️ 多模态CAD模型生成助手")
st.markdown("上传图片，一键得到想要的CAD模型")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("模型配置")
    # 默认地址通常是 Ollama (localhost:11434) 或 LM Studio (localhost:1234)
    base_url = st.text_input("API Base URL", value="http://localhost:11434/v1")
    api_key = st.text_input("API Key", value="lm-studio", type="password")
    model_name = st.text_input("模型名称 (Model ID)", value="llava")
    
    st.info("💡 提示：确保你的本地服务（Ollama/LM Studio等）已启动，并且加载了支持视觉的模型。")

# --- 辅助函数：将图片转为 Base64 ---
def encode_image_to_base64(uploaded_file):
    """将上传的文件对象转换为 Base64 字符串"""
    bytes_data = uploaded_file.getvalue()
    base64_str = base64.b64encode(bytes_data).decode('utf-8')
    return base64_str

# --- 主界面逻辑 ---
col1, col2 = st.columns([1, 1])

uploaded_file = st.sidebar.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 显示图片
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传图片", use_column_width=True)
    
    # 用户输入提示词
    with col2:
        prompt = st.text_area("输入你的问题", value="请详细描述这张图片的内容。")
        analyze_btn = st.button("开始分析", type="primary")

        if analyze_btn:
            try:
                # 初始化客户端
                client = OpenAI(base_url=base_url, api_key=api_key)
                
                # 图片转 Base64
                base64_image = encode_image_to_base64(uploaded_file)
                
                # 构建消息，支持流式输出
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    stream=True, # 启用流式输出，体验更好
                    temperature=0.7,
                )

                st.subheader("模型回复：")
                response_placeholder = st.empty()
                full_response = ""
                
                # 接收流式数据并实时显示
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"发生错误: {e}")
                st.warning("请检查：1. 本地服务是否启动？ 2. Base URL 是否正确？ 3. 模型名称是否匹配？")

else:
    with col2:
        st.info("👈 请先在左侧上传图片")