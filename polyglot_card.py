import streamlit as st
import pandas as pd
import time 
import json
import os
import requests 
from datetime import datetime, timedelta
from gtts import gTTS 
import io 
import base64 
import traceback

# --- 应用程序配置 ---
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions" 
DATA_FILE = "vocab_data.json" 
SEED_DATA_FILE = "seed_data.json" 
RERUN_TRIGGER = "rerun_pending" 


# --- 语言映射配置 ---
LANG_MAP = {
    'ru': {'name': '俄语 (RU)', 'lang_code': 'ru'},
    'fr': {'name': '法语 (FR)', 'lang_code': 'fr'},
    'en': {'name': '英语 (EN)', 'lang_code': 'en'}
}

# --- 辅助函数：注入CSS样式 (不变) ---
def apply_compact_styles():
    st.markdown("""
    <style>
    /* 缩小整体边距和顶部空间 */
    .stApp { padding-top: 5px; padding-bottom: 5px; }
    /* 缩小所有标题的间距和大小 */
    h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
    h2 { font-size: 1.2rem; margin-bottom: 0.2rem; }
    h3 { font-size: 1.0rem; margin-bottom: 0.1rem; }
    h4 { font-size: 0.9rem; }
    h5 { font-size: 0.8rem; margin-top: 3px; margin-bottom: 3px; }
    h6 { font-size: 0.75rem; margin-top: 3px; margin-bottom: 3px; }

    /* 调整 st.info 信息的间距 */
    .stAlert { padding: 4px; margin-top: 2px; margin-bottom: 4px; line-height: 1.2; }
    /* 调整按钮大小和边距 */
    div.stButton > button { padding: 3px 6px; font-size: 12px; margin-top: 3px; margin-bottom: 3px; }
    /* 侧边栏更紧凑 */
    .st-emotion-cache-1c9yi3e { padding-top: 0.5rem; }
    /* 调整普通文本的行距 */
    p { margin-bottom: 0.5rem; line-height: 1.4; }
    
    /* === 隐藏右下角所有浮动图标和 Manage app 按钮 (强力更新) === */
    [data-testid="stStatusWidget"], .st-emotion-cache-n0v05b, .st-emotion-cache-1j0083, .st-emotion-cache-12fmj6l, .st-emotion-cache-1jm69yr, .st-emotion-cache-10m726t, .st-emotion-cache-13l3q88 { 
        display: none !important;
        visibility: hidden !important; 
        z-index: -9999 !important; 
    }
    /* ========================================================= */
    
    /* 优化音标缺失的显示 */
    .ipa-missing-text { font-size: 12px; color: #888888; margin-top: 0; margin-bottom: 5px; display: block; }

    /* 隐藏用于触发翻转的按钮，但保留其功能 */
    div[data-testid="stButton"] button[key="flip_card_trigger"] { display: none; }
    
    </style>
    """, unsafe_allow_html=True)


# --- 1. LLM 生成功能 (不变) ---
def generate_content_with_llm(chinese_word, api_key):
    # (代码不变)
    if not api_key:
        return None 

    prompt_text = f"""你是一个高级多语种学习助手，擅长创造幽默且易于记忆的记忆法。请将中文单词'{chinese_word}'翻译成俄语、法语和英语。对于每种语言，请提供翻译、**国际音标 (IPA)**、一个听起来像中文的谐音（空耳），以及一个荒谬有趣的联想记忆法。

    严格以 JSON 格式返回，JSON 结构如下：
    {{
        "ru": {{"translation": "俄语翻译", "**ipa**": "俄语国际音标", "sound": "俄语中文谐音", "memo": "俄语联想记忆法"}},
        "fr": {{"translation": "法语翻译", "**ipa**": "法语国际音标", "sound": "法语中文谐音", "memo": "法语联想记忆法"}},
        "en": {{"translation": "英语翻译", "**ipa**": "英语国际音标", "sound": "英语中文谐音", "memo": "英语联想记忆法"}}
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
        if response.status_code == 401 or response.status_code == 403:
             st.error("API 密钥无效或余额不足。请检查侧边栏的 Key。")
        else:
             st.error(f"API 调用失败 (HTTP {response.status_code})。")
        return None
    except Exception as e:
        return None

# --- 2. 数据管理 (不变) ---
def save_data(data):
    # (代码不变)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data():
    # (代码不变)
    loaded_data = []
    
    def attempt_load(file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                if file_path == DATA_FILE:
                    st.error(f"数据文件 {DATA_FILE} 损坏，尝试加载内置数据。")
                return None
        return None

    data_list = attempt_load(DATA_FILE)
    if data_list is None:
        data_list = attempt_load(SEED_DATA_FILE)
        if data_list is None:
            return []
        else:
            save_data(data_list)
            st.info("✅ 首次启动：已加载内置词汇。您的所有修改将被保存在本地。")

    for word_entry in data_list:
        for key in LANG_MAP.keys():
            if key in word_entry:
                if 'ipa' not in word_entry[key]:
                    word_entry[key]['ipa'] = None 
                if word_entry[key].get('ipa') == "":
                    word_entry[key]['ipa'] = None
        loaded_data.append(word_entry)

    return loaded_data

# --- 3. 间隔重复算法 (不变) ---
def update_word_stats(word_entry, quality):
    # (代码不变)
    now = datetime.now()
    
    if quality == 0:
        interval = 0 
        word_entry['level'] = 0
        quality_text = "忘了 (重置)"
    elif quality == 1:
        interval = 720 # 12 小时 = 720 分钟
        word_entry['level'] = max(0, word_entry.get('level', 0) - 1)
        quality_text = "模糊 (12小时)"
    else: 
        level = word_entry.get('level', 0) + 1
        interval = 10 * level # 10 分钟 * Level
        word_entry['level'] = level
        quality_text = f"简单 ({interval}分钟)"

    word_entry['next_review'] = (now + timedelta(minutes=interval)).strftime("%Y-%m-%d %H:%M:%S")
    
    return word_entry, quality_text

# --- 4. 音频生成函数 (使用 @st.cache_data 确保持久化) ---
@st.cache_data(show_spinner=False)
def generate_audio_bytes(text, lang_code): 
    """
    使用 gTTS 生成音频字节流，并将其缓存在 Streamlit 的磁盘/内存中。
    """
    try:
        if not text or not lang_code:
            return None
            
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        return None 

# --- 5. 单词删除功能 (新增) ---
def delete_word(word_cn):
    """根据中文删除词汇，并重新定位到第一个词汇"""
    try:
        # 找到需要删除的词汇的全局索引
        global_index = next(i for i, w in enumerate(st.session_state.data) if w['cn'] == word_cn)
        
        # 移除词汇
        del st.session_state.data[global_index]
        
        # 保存更新后的数据
        save_data(st.session_state.data)
        
        # 清除该词汇对应的所有音频缓存，防止下次生成时仍使用旧的
        for key in LANG_MAP.keys():
            translation = next((w.get(key, {}).get('translation') for w in st.session_state.data if w.get('cn') == word_cn), None)
            if translation:
                generate_audio_bytes.clear_cache(translation, LANG_MAP[key]['lang_code'])

        
        # 重置状态，重新从头开始
        st.session_state.current_index = 0
        st.session_state.card_flipped = False
        st.session_state[RERUN_TRIGGER] = True 
        st.toast(f"🗑️ 已删除词汇：**{word_cn}**", icon="✅")
        
    except StopIteration:
        st.toast(f"🚫 错误：未找到词汇 **{word_cn}**", icon="❌")
    except Exception as e:
        st.error(f"删除失败: {e}")


# --- 6. 界面主程序 ---
def main():
    st.set_page_config(page_title="多语种智能记忆卡", layout="centered")
    apply_compact_styles()
    
    st.title("🌍 多语种智能记忆卡")
    
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
        st.session_state.card_flipped = False
        st.session_state.current_index = 0
    
    if 'user_deepseek_key' not in st.session_state:
        st.session_state.user_deepseek_key = ""

    # --- 侧边栏：配置 (不变) ---
    with st.sidebar:
        st.markdown("##### 联系方式:")
        st.markdown("**3717861@qq.com**")
        st.write("---") 
        st.header("词库管理")
        
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
                                "ru": {**llm_result['ru'], 'ipa': llm_result['ru'].get('ipa')},
                                "fr": {**llm_result['fr'], 'ipa': llm_result['fr'].get('ipa')},
                                "en": {**llm_result['en'], 'ipa': llm_result['en'].get('ipa')},
                                "next_review": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "level": 0
                            }
                            st.session_state.data.append(new_entry)
                            save_data(st.session_state.data)
                            st.session_state.card_flipped = False 
                            st.session_state.current_index = 0
                            st.session_state[RERUN_TRIGGER] = True 
                            st.success(f"已成功添加：{new_word}。")
                        else:
                            st.session_state.user_deepseek_key = ""
                            st.error("生成失败，请检查您的 API Key。")
        
        st.write("---")
        total_words = len(st.session_state.data)
        st.info(f"📚 词汇库总数: **{total_words}** 个")


    # --- 主界面：复习模式 ---
    st.markdown("---")
    
    # 1. 筛选出所有到期（due）的单词
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    due_words = sorted(
        [w for w in st.session_state.data if w['next_review'] <= now_str],
        key=lambda x: x['next_review']
    )
    
    words_due = len(due_words)

    if not due_words:
        st.success("🎉 太棒了！目前没有需要复习的单词。")
        return 

    if st.session_state.current_index >= words_due:
        st.session_state.current_index = 0
        
    current_word = due_words[st.session_state.current_index]

    
    # --- 导航按钮和卡片计数 (不变) ---
    def navigate_card(direction, current_word_cn):
        
        if st.session_state.get('current_cn') != current_word['cn']:
            st.session_state.current_index = (st.session_state.current_index + direction) % words_due
            st.session_state.card_flipped = False 
            st.session_state.start_time = time.time()
            st.session_state[RERUN_TRIGGER] = True 
            return
            
        dwell_time = time.time() - st.session_state.start_time
        card_flipped = st.session_state.card_flipped
        
        if dwell_time < 5.0 and not card_flipped:
            quality = 2 
            st.toast("✅ 自动评级：简单 (秒懂)", icon="😎")
        elif dwell_time > 10.0:
            quality = 0 
            st.toast("😭 自动评级：忘了 (耗时过长)", icon="😭")
        else:
            quality = 1
            st.toast("🤔 自动评级：模糊 (思考后确认)", icon="🤔")


        try:
            global_index = next(i for i, w in enumerate(st.session_state.data) if w['cn'] == current_word['cn'])
            updated_word, quality_text = update_word_stats(st.session_state.data[global_index], quality)
            st.session_state.data[global_index] = updated_word
            save_data(st.session_state.data)
            
        except StopIteration:
            st.error("程序错误：未找到当前词汇的全局索引。")
            return
        
        st.session_state.current_index = (st.session_state.current_index + direction) % words_due
        st.session_state.card_flipped = False 
        st.session_state.start_time = time.time()
        st.session_state[RERUN_TRIGGER] = True


    col1, col2, col3 = st.columns([1, 1, 2.5])

    # 导航按钮的参数修改为当前词汇
    with col1:
        st.button("⬅️ 上一个", on_click=navigate_card, args=(-1, current_word['cn']), key="prev_card")
    
    with col2:
        st.button("下一个 ➡️", on_click=navigate_card, args=(1, current_word['cn']), key="next_card")
    
    with col3:
        st.markdown(f"#### 任务: {st.session_state.current_index + 1} / {words_due} (剩余)")


    if 'start_time' not in st.session_state or st.session_state.get('current_cn') != current_word['cn']:
        st.session_state.start_time = time.time()
        st.session_state.current_cn = current_word['cn']
        st.session_state.card_flipped = False 

    card_placeholder = st.empty()

    with card_placeholder.container(border=True):
        
        title_text = f"卡片: **{current_word['cn']}**"
        card_expander = st.expander(title_text, expanded=True) 
        
        with card_expander:
            
            # --- 卡片正面内容 ---
            st.markdown("##### 外语翻译:")
            cols = st.columns(len(LANG_MAP))
            
            for i, (key, lang_data) in enumerate(LANG_MAP.items()):
                translation = current_word.get(key, {}).get('translation', "数据缺失")
                ipa = current_word.get(key, {}).get('ipa', None) 
                
                with cols[i]:
                    st.markdown(f"**{lang_data['name']}**")
                    st.markdown(f"### {translation}") 
                    
                    if ipa:
                         st.markdown(f"**[{ipa}]**") 
                    else:
                         st.markdown(f'<span class="ipa-missing-text">音标缺失</span>', unsafe_allow_html=True)

                    # st.audio 调用
                    if translation != "数据缺失":
                        audio_bytes = generate_audio_bytes(translation, lang_data['lang_code'])
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3', loop=False) 
                        else:
                            st.markdown("") 
            
            st.markdown("---") 

            # --- 卡片反面 ---
            
            if st.session_state.card_flipped:
                dwell_time = time.time() - st.session_state.start_time
                
                st.subheader(f"✅ 中文释义: {current_word['cn']}")
                
                for key in LANG_MAP:
                    if key in current_word:
                        lang_data = LANG_MAP[key]
                        lang_content = current_word[key]
                        
                        st.markdown(f"###### {lang_data['name']} 详情")
                        
                        col_sound, col_memo = st.columns([1, 2])
                        
                        with col_sound:
                            st.markdown(f"**谐音**: {lang_content.get('sound', '缺失')}")
                        with col_memo:
                            st.markdown(f"💡 **记忆法**: {lang_content.get('memo', '缺失')}") 
                    
                st.markdown("---")
                
                st.info(f"⏱️ 本轮思考用时: {dwell_time:.1f} 秒")
                
                # --- 新增：删除按钮 ---
                st.button(
                    f"❌ 永久删除 '{current_word['cn']}'", 
                    on_click=delete_word, 
                    args=(current_word['cn'],),
                    key="delete_button_final"
                )
                # -----------------------

                st.markdown("**(无需手动评分，点击 '上一个/下一个' 自动评级)**")


    # --- 关键触发按钮 (不变) ---
    if st.button("点击翻转卡片", key="flip_card_trigger"):
        st.session_state.card_flipped = not st.session_state.card_flipped
        st.session_state[RERUN_TRIGGER] = True 
        
    if st.session_state.get(RERUN_TRIGGER):
        st.session_state[RERUN_TRIGGER] = False
        st.rerun()


if __name__ == "__main__":
    main()