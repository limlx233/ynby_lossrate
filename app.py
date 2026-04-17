import streamlit as st
import warnings
import os
import tempfile
import pandas as pd
import atexit

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Serialization of dataframe.*")

# ========== 全局配置：禁用不必要的通信优化 ==========
st.config.set_option("client.caching", False)
st.config.set_option("server.enableXsrfProtection", False)

# ========== 定义页面专属变量（仅保存小型变量/临时文件路径） ==========
# Page1 专属变量
PAGE1_VARS = [
    "org_page1", 
    "batch_nodes_page1", 
    "anova_results_page1",
    "IMR_params_page1",
    # 临时文件路径（存储大型DataFrame）
    "temp_historical_file",
    "temp_current_month_file",
    "temp_historical_filtered",
    "temp_current_month_outliers",
    "temp_current_month_control_limits",
    "temp_historical_low_loss",
    "temp_historical_high_loss",
    "temp_current_month_low_loss",
    "temp_current_month_high_loss"
]

# Page2 专属变量
PAGE2_VARS = [
    "org_page2",
    "batch_nodes_fhg_p5",
    "batch_nodes_zh_p5",
    "batch_nodes_zx_p5",
    "IMR_params_fhg_p5",
    "IMR_params_zh_p5",
    "IMR_params_zx_p5",
    # 临时文件路径
    "temp_fhg_outlier_p5",
    "temp_zh_outlier_p5",
    "temp_zx_outlier_p5",
    "temp_fhg_p5",
    "temp_zh_p5",
    "temp_zx_p5",
    "temp_fhg_low_loss_rate",
    "temp_zh_low_loss_rate",
    "temp_zx_low_loss_rate"
]

# 全局共享变量
GLOBAL_VARS = ["current_page"]

# ========== 工具函数：临时文件管理（核心优化） ==========
def save_df_to_tempfile(df):
    """将DataFrame保存到临时文件，返回文件路径（替代直接存session_state）"""
    if df is None or df.empty:
        return None
    # 创建temp目录
    os.makedirs("./temp", exist_ok=True)
    # 创建临时文件（后缀.pkl，避免删除）
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".pkl", mode="wb", delete=False, dir="./temp"
    )
    # 用pickle高效存储
    df.to_pickle(temp_file)
    temp_file.close()
    return temp_file.name

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

# 注册会话结束清理函数
atexit.register(cleanup_all_temp_files)

# ========== 会话状态初始化 + 页面切换清理 ==========
def init_session_state():
    """初始化所有变量（仅赋空值，不存大型对象）"""
    all_vars = PAGE1_VARS + PAGE2_VARS + GLOBAL_VARS
    for var in all_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    # 初始化当前页面
    if st.session_state["current_page"] is None:
        st.session_state["current_page"] = "膏体损耗率分析"

def clean_non_current_page_vars(current_page_title):
    """切换页面时，清理非当前页的变量和临时文件"""
    page_vars_map = {
        "膏体损耗率分析": PAGE1_VARS,
        "包材损耗率分析": PAGE2_VARS
    }
    keep_vars = page_vars_map.get(current_page_title, []) + GLOBAL_VARS
    
    # 遍历并清理非保留变量
    for key in list(st.session_state.keys()):
        if key not in keep_vars and not key.startswith("_"):
            # 若是临时文件路径，先删除文件
            if key.startswith("temp_"):
                cleanup_temp_file(st.session_state[key])
            # 删除变量
            del st.session_state[key]

# ========== 页面配置 + 初始化 ==========
st.set_page_config(
    page_title="损耗率 IMR 控制图分析系统",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 初始化会话状态（必须在导航前）
init_session_state()

# ========== 页面导航 ==========
pages = [
    st.Page("page1.py", title="膏体损耗率分析"),
    st.Page("page2.py", title="包材损耗率分析"),
]

pg = st.navigation(pages)

# 页面切换时执行清理（核心：解决通信格式错误）
if pg.title != st.session_state["current_page"]:
    clean_non_current_page_vars(pg.title)
    st.session_state["current_page"] = pg.title

# 运行当前页面
pg.run()