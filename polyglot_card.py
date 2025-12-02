import streamlit as st
import pandas as pd
import time
import json
import os
import requests 
from datetime import datetime, timedelta
from gtts import gTTS 
import io 
import base64 # 引入 base64 库
import traceback

# --- 应用程序配置 ---
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions" 
DATA_FILE = "vocab_data.json" 
SEED_DATA_FILE = "seed_data.json" 

# --- 语言映射配置 ---
LANG_MAP = {
    'ru': {'name': '俄语 (RU)', 'lang_code': 'ru'},
    'fr': {'name': '法语 (FR)', 'lang_code': 'fr'},
    'en': {'name': '英语 (EN)', 'lang_code': 'en'}
}

# --- 辅助函数：注入CSS样式 (极致紧凑化 v2) ---
def apply_compact_styles():
    st.markdown("""
    <style>
    /* 缩小整体边距和顶部空间 */
    .stApp {
        padding-top: 5px; 
        padding-bottom: 5px;
    }
    /* 缩小所有标题的间距和大小 (更小) */
    h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
    h2 { font-size: 1.2rem; margin-bottom: 0.2rem; }
    h3 { font-size: 1.0rem; margin-bottom: 0.1rem; }
    h4 { font-size: 0.9rem; }
    h5 { font-size: 0.8rem; margin-top: 3px; margin-bottom: 3px; }
    h6 { font-size: 0.75rem; margin-top: 3px; margin-bottom: 3px; }

    /* 调整 st.info 信息的间距 */
    .stAlert {
        padding: 4px; 
        margin-top: 2px;
        margin-bottom: 4px;
        line-height: 1.2; 
    }
    /* 调整 st.audio 播放器的高度和边距 */
    audio {
        width: 100%; 
        height: 20px; /* 更小的播放器 */
        margin-top: 3px;
        margin-bottom: 3px;
    }
    /* 调整按钮大小和边距 */
    div.stButton > button {
        padding: 3px 6px; 
        font-size: 12px;
        margin-top: 3px;
        margin-bottom: 3px;
    }
    /* 侧边栏更紧凑 */
    .st-emotion-cache-1c9yi3e { 
        padding-top: 0.5rem;
    }
    /* 调整普通文本的行距 */
    p {
        margin-bottom: 0.5rem;
        line-height: 1.4; 
    }
    </style>
    """, unsafe_allow_html=True)


# --- 1. LLM 生成功能 (保持不变) ---
def generate_content_with_llm(chinese_word, api_key):
    """调用 DeepSeek API，同时生成俄语、法语和英语的内容"""
    if not api_key:
        return None 

    prompt_text = f"""你是一个高级多语种学习助手，擅长创造幽默且易于记忆的记忆法。请将中文单词'{chinese_word}'翻译成俄语、法语和英语。对于每种语言，请提供翻译、一个听起来像中文的谐音（空耳），以及一个荒谬有趣的联想记忆法。

    严格以 JSON 格式返回，JSON 结构如下（注意，每个语言下都是一个子对象）：
    {{
        "ru": {{"translation": "俄语翻译", "sound": "俄语中文谐音", "memo": "俄语联想记忆法"}},
        "fr": {{"translation": "法语翻译", "sound": "法语中文谐音", "memo": "法语联想记忆法"}},
        "en": {{"translation": "英语翻译", "sound": "英语中文谐音", "memo": "英语联想记忆法"}}
    }}
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat", 
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"}, 
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status() 
        response_json = response.json()
        content_str = response_json['choices'][0]['message']['content']
        return json.loads(content_str)
        
    except requests.exceptions.HTTPError as e:
        st.error(f"API调用失败 (HTTP {response.status_code})：请检查您的 API Key 和余额。")
    except Exception as e:
        st.error(f"API调用失败，可能是 DeepSeek 返回的 JSON 格式不正确。错误: {e}")
        
    return None

# --- 2. 数据管理 (保持不变) ---
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"数据文件 {DATA_FILE} 损坏，已自动重置。")
            return []
            
    elif os.path.exists(SEED_DATA_FILE):
        try:
            with open(SEED_DATA_FILE, "r", encoding="utf-8") as f:
                initial_data = json.load(f)
            
            save_data(initial_data)
            st.info("✅ 首次启动：已加载内置词汇。您的所有修改将被保存在本地。")
            return initial_data
        except json.JSONDecodeError:
            st.error(f"内置数据文件 {SEED_DATA_FILE} 损坏。请检查格式。")
            return []
            
    return []

# --- 3. 间隔重复算法 (保持不变) ---
def update_word_stats(word_entry, quality, dwell_time):
    if dwell_time > 10 and quality == 2:
        quality = 1 
        
    now = datetime.now()
    
    if quality == 0:
        interval = 0 
        word_entry['level'] = 0
    elif quality == 1:
        interval = 12 
        word_entry['level'] = max(0, word_entry.get('level', 0) - 1)
    else: 
        level = word_entry.get('level', 0) + 1
        interval = 24 * (2 ** (level - 1))
        word_entry['level'] = level

    word_entry['next_review'] = (now + timedelta(hours=interval)).strftime("%Y-%m-%d %H:%M:%S")
    return word_entry

# --- 4. Base64 音频生成函数 (新函数，用于手机兼容) ---
@st.cache_data
def generate_base64_audio(text, lang_code):
    """
    使用 gTTS 生成音频，并将其 Base64 编码后嵌入到 HTML 中。
    这显著提高了移动设备上的兼容性。
    """
    try:
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Base64 编码
        b64 = base64.b64encode(mp3_fp.read()).decode()
        
        # 嵌入 HTML <audio> 标签
        html = f"""
        <audio controls style="width: 100%; height: 20px; margin-top: 3px;">
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          Your browser does not support the audio element.
        </audio>
        """
        return html
    except Exception as e:
        return f""

# --- 5. 界面主程序 (保持不变) ---
def main():
    st.set_page_config(page_title="多语种智能记忆卡", layout="centered")
    apply_compact_styles()
    
    st.title("🌍 多语种智能记忆卡")
    
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
        st.session_state.card_flipped = False
    
    if 'user_deepseek_key' not in st.session_state:
        st.session_state.user_deepseek_key = ""

    # --- 侧边栏：配置 (保持不变) ---
    with st.sidebar:
        st.header("🔑 拓展词汇：付费功能")
        
        key_input = st.text_input(
            "输入 DeepSeek API Key", 
            value=st.session_state.user_deepseek_key,
            type="password", 
            help="拓展新词汇需使用您自己的 Key。"
        )
        
        if key_input:
            st.session_state.user_deepseek_key = key_input
        elif st.session_state.user_deepseek_key:
            st.warning("请重新输入 Key 以启用 AI 功能。")
            st.session_state.user_deepseek_key = ""

        st.write("---")
        
        st.header("AI 生成新词汇")
        new_word = st.text_input("输入中文单词")
        
        ai_enabled = bool(st.session_state.user_deepseek_key)
        
        if st.button("🚀 AI 生成并保存", disabled=not ai_enabled):
            if not ai_enabled:
                st.error("请输入 API Key 以启用 AI 生成功能。")
            elif not new_word:
                st.warning("请输入单词")
            else:
                word_exists = any(entry.get('cn') == new_word for entry in st.session_state.data)
                
                if word_exists:
                    st.warning(f"💡 词汇库中已有单词：**{new_word}**。")
                else:
                    with st.spinner(f"正在调用 DeepSeek 为 '{new_word}' 生成..."):
                        llm_result = generate_content_with_llm(new_word, st.session_state.user_deepseek_key)
                        
                        if (llm_result and all(key in llm_result for key in LANG_MAP.keys())):
                            new_entry = {
                                "cn": new_word,
                                "ru": llm_result['ru'], "fr": llm_result['fr'], "en": llm_result['en'],
                                "next_review": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "level": 0
                            }
                            st.session_state.data.append(new_entry)
                            save_data(st.session_state.data)
                            st.session_state.card_flipped = False 
                            st.success(f"已成功添加：{new_word}。")
                            st.rerun() 
                        else:
                            st.session_state.user_deepseek_key = ""
                            st.error("生成失败，请检查您的 API Key。")
        
        st.write("---")
        total_words = len(st.session_state.data)
        st.info(f"📚 词汇库总数: **{total_words}** 个")


    # --- 主界面：复习模式 ---
    st.markdown("---")
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    due_words = sorted(
        [w for w in st.session_state.data if w['next_review'] <= now_str],
        key=lambda x: x['next_review']
    )
    
    words_due = len(due_words)
    st.markdown(f"#### 今日任务：{words_due} 个 (剩余)") 

    if not due_words:
        st.success("🎉 太棒了！目前没有需要复习的单词。")
        if not st.session_state.data:
            st.info("请在左侧边栏输入中文单词，开始生成。")
        return 

    current_word = due_words[0]
    
    # 确保切换新卡片时，状态重置为未翻转
    if 'start_time' not in st.session_state or st.session_state.get('current_cn') != current_word['cn']:
        st.session_state.start_time = time.time()
        st.session_state.current_cn = current_word['cn']
        st.session_state.card_flipped = False 

    card_placeholder = st.empty()

    with card_placeholder.container(border=True):
        
        # 卡片标题
        title_text = f"卡片: **{current_word['cn']}**"
        
        # 卡片在未翻转时 (card_flipped=False) 强制 expanded=True
        card_expander = st.expander(title_text, expanded=True) 
        
        with card_expander:
            
            # --- 卡片正面内容 (始终显示) ---
            st.markdown("##### 外语翻译:")
            cols = st.columns(len(LANG_MAP))
            
            for i, (key, lang_data) in enumerate(LANG_MAP.items()):
                translation = current_word.get(key, {}).get('translation', "数据缺失")
                
                with cols[i]:
                    st.markdown(f"**{lang_data['name']}**")
                    st.markdown(f"### {translation}") 
                    if translation != "数据缺失":
                         # 关键：使用 Base64 嵌入 HTML <audio> 标签
                         audio_html = generate_base64_audio(translation, lang_data['lang_code'])
                         st.markdown(audio_html, unsafe_allow_html=True) 
            
            st.markdown("---") 

            # --- 卡片反面（仅在 card_flipped=True 时显示）---
            
            if st.session_state.card_flipped:
                dwell_time = time.time() - st.session_state.start_time
                
                st.subheader(f"✅ 中文释义: {current_word['cn']}")
                
                for key, lang_data in LANG_MAP.items():
                    lang_content = current_word.get(key, {})
                    
                    st.markdown(f"###### {lang_data['name']} 详情")
                    
                    col_sound, col_memo = st.columns([1, 2])
                    
                    with col_sound:
                        st.markdown(f"**谐音**: {lang_content.get('sound', '缺失')}")
                    with col_memo:
                        st.markdown(f"💡 **记忆法**: {lang_content.get('memo', '缺失')}") 
                    
                st.markdown("---")
                
                # 复习评分区
                st.info(f"⏱️ 思考用时: {dwell_time:.1f} 秒")
                st.markdown("⭐ **请评估你的掌握程度：**")
                
                c1, c2, c3 = st.columns(3)
                
                def handle_review(quality):
                    idx = st.session_state.data.index(current_word)
                    st.session_state.data[idx] = update_word_stats(current_word, quality, dwell_time)
                    save_data(st.session_state.data)
                    st.session_state.card_flipped = False 
                    del st.session_state['start_time']
                    st.rerun()

                if c1.button("😭 忘了 (0)", key="q0"):
                    handle_review(0)
                if c2.button("🤔 模糊 (1)", key="q1"):
                    handle_review(1)
                if c3.button("😎 简单 (2)", key="q2"):
                    handle_review(2)

    # --- 关键触发按钮 ---
    if st.button("点击翻转卡片", key="flip_card_trigger"):
        st.session_state.card_flipped = not st.session_state.card_flipped
        st.rerun()
        
    st.markdown("""
    <style>
    /* 隐藏用于触发翻转的按钮，但保留其功能 */
    div[data-testid="stButton"] button[key="flip_card_trigger"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()