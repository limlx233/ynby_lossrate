import streamlit as st
import warnings
import os
import tempfile
import pandas as pd
import atexit

# 强制指定 Streamlit 配置（适配1.41.1）
st.set_page_config(
    page_title="损耗率 IMR 控制图分析系统",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Serialization of dataframe.*")

# ========== 全局配置（适配1.41.1） ==========
try:
    st.config.set_option("client.caching", False)
    st.config.set_option("server.enableXsrfProtection", False)
except:
    # 1.41.1 部分配置项命名不同，降级处理
    st._config.set_option("client.caching", False)
    st._config.set_option("server.enableXsrfProtection", False)

# ========== 定义页面专属变量 ==========
PAGE1_VARS = [
    "org_page1", "batch_nodes_page1", "anova_results_page1", "IMR_params_page1",
    "temp_historical_file", "temp_current_month_file", "temp_historical_filtered",
    "temp_current_month_outliers", "temp_current_month_control_limits",
    "temp_historical_low_loss", "temp_historical_high_loss",
    "temp_current_month_low_loss", "temp_current_month_high_loss"
]

PAGE2_VARS = [
    "org_page2", "batch_nodes_fhg_p5", "batch_nodes_zh_p5", "batch_nodes_zx_p5",
    "IMR_params_fhg_p5", "IMR_params_zh_p5", "IMR_params_zx_p5",
    "temp_fhg_outlier_p5", "temp_zh_outlier_p5", "temp_zx_outlier_p5",
    "temp_fhg_p5", "temp_zh_p5", "temp_zx_p5",
    "temp_fhg_low_loss_rate", "temp_zh_low_loss_rate", "temp_zx_low_loss_rate"
]

GLOBAL_VARS = ["current_page"]

# ========== 工具函数：临时文件管理（兼容1.41.1） ==========
def save_df_to_tempfile(df):
    """将DataFrame保存到临时文件，返回文件路径"""
    if df is None or df.empty:
        return None
    os.makedirs("./temp", exist_ok=True)
    # 1.41.1 兼容的临时文件命名（避免特殊字符）
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".pkl", mode="wb", delete=False, dir="./temp", prefix="df_"
    )
    try:
        df.to_pickle(temp_file)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.warning(f"保存临时文件失败: {e}")
        temp_file.close()
        return None

def load_df_from_tempfile(file_path):
    """从临时文件加载DataFrame"""
    if file_path is None or not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        st.warning(f"加载临时文件失败: {e}")
        return pd.DataFrame()

def cleanup_temp_file(file_path):
    """删除单个临时文件"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass

def cleanup_all_temp_files():
    """会话结束时清理所有临时文件"""
    temp_dir = "./temp"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

atexit.register(cleanup_all_temp_files)

# ========== 会话状态初始化（适配1.41.1） ==========
def init_session_state():
    """初始化所有变量（1.41.1 兼容）"""
    all_vars = PAGE1_VARS + PAGE2_VARS + GLOBAL_VARS
    for var in all_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    # 初始化默认页面
    if st.session_state["current_page"] is None:
        st.session_state["current_page"] = "膏体损耗率分析"

def clean_non_current_page_vars(current_page_title):
    """切换页面时清理非当前页变量"""
    page_vars_map = {
        "膏体损耗率分析": PAGE1_VARS,
        "包材损耗率分析": PAGE2_VARS
    }
    keep_vars = page_vars_map.get(current_page_title, []) + GLOBAL_VARS
    
    # 1.41.1 兼容的会话状态遍历
    for key in list(st.session_state.keys()):
        if key not in keep_vars and not key.startswith("_"):
            if key.startswith("temp_"):
                cleanup_temp_file(st.session_state[key])
            del st.session_state[key]

# ========== 页面切换（替换 st.navigation 为 1.41.1 兼容版） ==========
init_session_state()

# 侧边栏页面选择（1.41.1 稳定方案）
st.sidebar.title("页面导航")
page_options = ["膏体损耗率分析", "包材损耗率分析"]
selected_page = st.sidebar.radio(
    "选择分析模块",
    page_options,
    index=page_options.index(st.session_state["current_page"])
)

# 页面切换时清理
if selected_page != st.session_state["current_page"]:
    clean_non_current_page_vars(selected_page)
    st.session_state["current_page"] = selected_page

# ========== 导入并运行对应页面 ==========
if selected_page == "膏体损耗率分析":
    # 导入 page1 并传递工具函数
    exec(open("page1.py", encoding="utf-8").read())
else:
    # 导入 page2 并传递工具函数
    exec(open("page2.py", encoding="utf-8").read())