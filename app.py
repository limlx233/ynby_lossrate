import streamlit as st
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Serialization of dataframe.*")


# 页面配置
st.set_page_config(
    page_title="损耗率 IMR 控制图分析系统",
    layout="centered",
    initial_sidebar_state="expanded"
)
# ========== 主界面逻辑 ==========

# 保存和恢复会话状态
def save_state():
    st.session_state['saved_state'] = st.session_state.to_dict()

def restore_state():
    if 'saved_state' in st.session_state:
        for key, value in st.session_state['saved_state'].items():
            st.session_state[key] = value


pages = [
        st.Page("page1.py", title="膏体损耗率分析"),
        st.Page("page2.py", title="包材损耗率分析"),
]

pg = st.navigation(pages)

# 监听导航切换事件
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = pg.title

if pg.title != st.session_state['current_page']:
    save_state()
    st.session_state['current_page'] = pg.title

# 显示当前页面
pg.run()