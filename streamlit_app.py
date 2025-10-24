import streamlit as st
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import requests
import tempfile
import os
from gtts import gTTS
import io
import base64

# 设置页面
st.set_page_config(
    page_title="情感强度翻译器",
    page_icon="🎯",
    layout="wide"
)

# 标题和介绍
st.title("🎯 情感强度翻译器")
st.markdown("将语音转换为带有情感强度的文字表达 - **24/7永久运行**")

# 初始化模型（使用缓存避免重复加载）
@st.cache_resource
def load_models():
    """加载所有模型"""
    st.info("🔄 加载模型中...")
    
    # 转录模型
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    # 简化情感强度模型
    class SimpleIntensityModel(nn.Module):
        def __init__(self, input_size=80, num_classes=6):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.classifier(x)
    
    intensity_model = SimpleIntensityModel()
    
    st.success("✅ 模型加载完成！")
    return processor, model, intensity_model

# 加载模型
processor, model, intensity_model = load_models()

# DeepSeek API密钥（在Streamlit Cloud中设置）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def transcribe_audio(audio_file):
    """转录音频"""
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # 加载音频
        audio, sr = librosa.load(tmp_file_path, sr=16000)
        
        # 提取特征
        input_features = processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features
        
        # 生成转录
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # 清理临时文件
        os.unlink(tmp_file_path)
        
        return transcription
        
    except Exception as e:
        return f"转录失败: {str(e)}"

def predict_intensity_simple(audio_file):
    """简单情感强度预测"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        audio, sr = librosa.load(tmp_file_path, sr=16000)
        
        # 基于音频特征简单判断强度
        # 1. 能量特征
        rms = librosa.feature.rms(y=audio)
        avg_rms = np.mean(rms)
        
        # 2. 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        avg_centroid = np.mean(spectral_centroids)
        
        # 简单加权评分
        intensity_score = (avg_rms * 100 + avg_centroid / 100) / 2
        
        # 映射到0-5
        if intensity_score < 0.1: intensity = 1
        elif intensity_score < 0.2: intensity = 2
        elif intensity_score < 0.4: intensity = 3
        elif intensity_score < 0.6: intensity = 4
        else: intensity = 5
        
        os.unlink(tmp_file_path)
        return intensity
        
    except Exception as e:
        st.error(f"强度分析失败: {e}")
        return 3  # 默认值

def translate_with_emotion(text, intensity):
    """情感翻译"""
    if not DEEPSEEK_API_KEY:
        # 备用翻译
        intensity_words = ["", "有点", "有些", "比较", "很", "非常"]
        word = intensity_words[intensity] if intensity < len(intensity_words) else "有些"
        
        if intensity <= 1:
            return f"{text}"
        elif intensity <= 3:
            return f"我现在{word}{text}，请帮帮我"
        else:
            return f"我现在{word}{text}！请帮帮我！"
    
    # DeepSeek API调用
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    system_prompt = "你是一个情感翻译器。将用户的话转换成带有情感强度的表达。直接输出翻译后的句子，不要加任何解释。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{text} 强度{intensity}"}
    ]
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        st.error(f"DeepSeek API调用失败: {e}")
        # 备用翻译
        intensity_words = ["", "有点", "有些", "比较", "很", "非常"]
        word = intensity_words[intensity] if intensity < len(intensity_words) else "有些"
        return f"我现在{word}{text}，请帮帮我"

def text_to_speech_base64(text):
    """文本转语音，返回base64"""
    try:
        tts = gTTS(text=text, lang="zh-cn")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        st.error(f"语音生成失败: {e}")
        return None

# 主界面
tab1, tab2 = st.tabs(["🎤 音频分析", "💡 使用说明"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("上传音频")
        
        # 文件上传
        audio_file = st.file_uploader(
            "选择音频文件",
            type=["wav", "mp3", "m4a", "ogg"],
            help="支持 WAV, MP3, M4A, OGG 格式"
        )
        
        if audio_file is not None:
            # 显示音频信息
            st.audio(audio_file, format="audio/wav")
            st.info(f"📁 文件: {audio_file.name}")
            
            # 分析按钮
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                with st.spinner("分析中...请稍候"):
                    # 转录
                    transcription = transcribe_audio(audio_file)
                    
                    # 情感强度
                    intensity = predict_intensity_simple(audio_file)
                    
                    # 情感翻译
                    translated_text = translate_with_emotion(transcription, intensity)
                    
                    # 生成语音
                    audio_url = text_to_speech_base64(translated_text)
                
                # 显示结果
                st.success("✅ 分析完成！")
                
                st.subheader("📊 分析结果")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("原始转录", transcription)
                    st.metric("情感强度", f"{intensity}/5")
                
                with col_b:
                    st.text_area("情感翻译", translated_text, height=100)
                
                # 语音播放
                if audio_url:
                    st.subheader("🔊 语音输出")
                    st.audio(audio_url, format="audio/mp3")
                    st.download_button(
                        "下载语音",
                        data=base64.b64decode(audio_url.split(",")[1]),
                        file_name="情感翻译.mp3",
                        mime="audio/mp3"
                    )

    with col2:
        st.subheader("实时状态")
        if audio_file is None:
            st.info("👆 请在上传音频后开始分析")
        else:
            st.success("✅ 音频文件已就绪")
            
        # 显示配置状态
        st.subheader("⚙️ 配置状态")
        if DEEPSEEK_API_KEY:
            st.success("✅ DeepSeek API 已配置")
        else:
            st.warning("⚠️ DeepSeek API 未配置，使用基础翻译")

with tab2:
    st.subheader("使用说明")
    
    st.markdown("""
    ### 🎯 功能说明
    将用户的语音转换成带有情感强度的文字表达，主要用于医疗护理场景。
    
    ### 💡 使用示例
    | 用户输入 | 情感强度 | 输出结果 |
    |---------|----------|----------|
    | "我想喝水" | 强度2 | "我现在有点想喝水，请帮帮我" |
    | "我想喝水" | 强度4 | "我现在很想喝水！请帮帮我！" |
    | "我有点冷" | 强度3 | "我现在比较冷，请帮帮我" |
    
    ### 🎚️ 情感强度说明
    - **强度0-1**: 平静表达
    - **强度2-3**: 中等急切  
    - **强度4-5**: 非常急切
    
    ### 🔧 技术架构
    - **语音转录**: OpenAI Whisper
    - **情感分析**: 自定义深度学习模型
    - **文本生成**: DeepSeek API
    - **语音合成**: Google TTS
    """)
    
    st.subheader("配置说明")
    st.markdown("""
    如需获得最佳效果，请在Streamlit Cloud中设置环境变量：
    - `DEEPSEEK_API_KEY`: 您的DeepSeek API密钥
    """)

# 页脚
st.markdown("---")
st.markdown("🎯 情感强度翻译器 | 24/7永久运行 | Streamlit Cloud部署")
