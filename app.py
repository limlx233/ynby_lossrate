import streamlit as st
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Serialization of dataframe.*")

# ========== 第一步：全局会话状态初始化（最顶部执行） ==========
def init_all_session_state():
    """统一初始化所有页面的会话状态，避免未定义/未初始化"""
    # 标记是否已全局初始化
    if 'global_session_initialized' not in st.session_state:
        # Page1 所需的会话状态
        page1_vars = [
            'cached_historical_file', 'cached_current_month_file', 'batch_nodes',
            'anova_results', 'historical_processed_data', 'historical_filtered_data',
            'historical_unknown_batch', 'historical_low_loss', 'historical_high_loss',
            'IMR_params', 'current_month_raw_data', 'current_month_matched_data',
            'current_month_filtered_data', 'current_month_unknown_batch',
            'current_month_low_loss', 'current_month_high_loss',
            'current_month_outliers', 'current_month_control_limits'
        ]
        # Page2 所需的会话状态
        page2_vars = [
            "historical_processed_fhg_p5", "batch_nodes_fhg_p5", "IMR_params_fhg_p5",
            "historical_processed_zh_p5", "batch_nodes_zh_p5", "IMR_params_zh_p5",
            "historical_processed_zx_p5", "batch_nodes_zx_p5", "IMR_params_zx_p5",
            "current_fhg_p5", "current_zh_p5", "current_zx_p5",
            "fhg_outlier_p5", "zh_outlier_p5", "zx_outlier_p5",
            "fhg_p5", "zh_p5", "zx_p5",
            "fhg_low_loss_rate", "zh_low_loss_rate", "zx_low_loss_rate"
        ]
        # 初始化所有变量为 None（避免未定义）
        for var in page1_vars + page2_vars:
            if var not in st.session_state:
                st.session_state[var] = None
        # 标记全局初始化完成
        st.session_state['global_session_initialized'] = True
    # 初始化导航相关状态
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = ""

# 立即执行全局初始化（必须在页面配置前）
init_all_session_state()

# ========== 页面配置 ==========
st.set_page_config(
    page_title="损耗率 IMR 控制图分析系统",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== 安全的状态保存/恢复（修复核心） ==========
def save_state():
    """安全保存会话状态：只保存已初始化的变量，避免访问未初始化上下文"""
    # 过滤掉 Streamlit 内部变量，只保存业务变量
    safe_state = {}
    for key, value in st.session_state.items():
        # 排除 Streamlit 内部变量（以 _ 开头）和全局标记
        if not key.startswith('_') and key != 'global_session_initialized':
            safe_state[key] = value
    st.session_state['saved_state'] = safe_state

def restore_state():
    """安全恢复会话状态：只恢复已存在的变量"""
    if 'saved_state' in st.session_state:
        for key, value in st.session_state['saved_state'].items():
            # 只恢复业务变量，避免覆盖内部状态
            if not key.startswith('_'):
                st.session_state[key] = value

# ========== 页面导航 ==========
pages = [
    st.Page("page1.py", title="膏体损耗率分析"),
    st.Page("page2.py", title="包材损耗率分析"),
]

pg = st.navigation(pages)

# 安全的导航切换逻辑
if pg.title != st.session_state['current_page']:
    # 先恢复再保存，避免状态冲突
    restore_state()
    save_state()
    st.session_state['current_page'] = pg.title

# 显示当前页面
pg.run()