# 标准库
import os
import io
from datetime import datetime
# 第三方库
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import norm

# 自定义库（确保 DP.py 存在）
try:
    from DP import dp3
except ImportError:
    st.error("缺少自定义库 DP.py，请确保 dp3 函数存在！")
    st.stop()

# 忽略无关警告
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置（适配1.41.1） =====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']  # 兼容1.41.1字体

# ===================== 核心函数（完全保留原有逻辑） =====================
def calculate_process_sigma_and_cpk(
    row,
    cl_col='I图中心值(CL)',
    ucl_col='I图上限(UCL)',
    lcl_col='I图下限(LCL)',
    loss_rate_col='损耗率%',
    usl=30.0,
    lsl=0.0
):
    try:
        cl = float(row[cl_col]) if pd.notna(row[cl_col]) else np.nan
        ucl = float(row[ucl_col]) if pd.notna(row[ucl_col]) else np.nan
        lcl = float(row[lcl_col]) if pd.notna(row[lcl_col]) else np.nan
        if np.isnan(ucl) or np.isnan(lcl) or (ucl - lcl) <= 0:
            return np.nan, np.nan
        process_sigma = (ucl - lcl) / 6.0
        cpu = (usl - cl) / (3 * process_sigma)
        cpl = (cl - lsl) / (3 * process_sigma)
        cpk = min(cpu, cpl)
        z_value = cpk * 3
        sigma_level = z_value + 1.5
        sigma_level = max(sigma_level, 0.0)
        return round(cpk, 4), f"{sigma_level:.1f}σ"
    except:
        return np.nan, np.nan

def judge_abnormal_and_remark(df):
    df_copy = df.copy()
    df_copy['异常值'] = False
    df_copy['超限备注'] = "正常"
    both_out_mask = (df_copy['I图超限']) & (df_copy['MR超限'])
    df_copy.loc[both_out_mask, '异常值'] = True
    df_copy.loc[both_out_mask, '超限备注'] = "单值超限且过程波动异常（异常值）"
    only_i_out_mask = (df_copy['I图超限']) & (~df_copy['MR超限'])
    df_copy.loc[only_i_out_mask, '超限备注'] = "单值超限、但过程稳定"
    only_mr_out_mask = (~df_copy['I图超限']) & (df_copy['MR超限'])
    df_copy.loc[only_mr_out_mask, '超限备注'] = "单值未超限，过程有波动"
    return df_copy

def standardize_data_columns(df, data_type):
    df_copy = df.copy()
    base_cols = [
        '年月份', '产品说明', '产品批号', '线体', '香型', '规格', '产品',
        '批次分类', '批量分类', '实际', '理论', '损耗率', '损耗率%' 
    ]
    if data_type == 'outlier':
        extra_cols = [
            '异常值', '超限备注', 
            'I图控制上限', 'I图控制下限', 'I图控制中心值',
            'MR图控制上限', 'MR图控制下限', 'MR图控制中心值',
            'I图超限', 'MR超限', '超限'
        ]
    elif data_type == 'low_loss':
        extra_cols = ['异常值', '超限备注']
        df_copy['异常值'] = True
        df_copy['超限备注'] = "负损耗（损耗率<0）"
        df_copy['Sigma水平'] = ""
        df_copy['Cpk'] = np.nan
    elif data_type == 'high_loss':
        extra_cols = ['异常值', '超限备注']
        df_copy['异常值'] = True
        df_copy['超限备注'] = "高损耗（损耗率>30%）"
        df_copy['Sigma水平'] = ""
        df_copy['Cpk'] = np.nan
    else:
        extra_cols = []
    
    all_cols = base_cols + extra_cols
    all_cols = [col for col in all_cols if col in df_copy.columns]
    
    if '损耗率%' not in df_copy.columns and '损耗率' in df_copy.columns:
        df_copy['损耗率%'] = df_copy['损耗率'] * 100
    
    df_standardized = df_copy[all_cols].copy()
    return df_standardized

def plot_imr_control_charts(df_analysis: pd.DataFrame, 
                            df_control_params: pd.DataFrame,
                            line_col: str = '线体',
                            batch_type_col: str = '批次分类',
                            batch_size_col: str = '批量分类',
                            flavor: str = '香型',
                            sku: str = '产品',
                            product: str = '产品说明',
                            product_size: str = '规格',
                            batch_id: str = '产品批号',
                            batch_id_col: str = '批号次序',
                            actual_value = '实际',
                            therotical_value = '理论',
                            loss_rate_col: str = '损耗率') -> tuple[pd.DataFrame, pd.DataFrame]:
    df_analysis_copy = df_analysis.copy()
    df_control_copy = df_control_params.copy()
    df_analysis_copy['损耗率%'] = df_analysis_copy[loss_rate_col] * 100
    
    if line_col in df_analysis_copy.columns:
        df_analysis_copy[line_col] = df_analysis_copy[line_col].apply(lambda x: f"{x}线")
    else:
        df_analysis_copy[line_col] = "未知线"
        st.warning(f"数据缺少{line_col}列，已默认填充为'未知线'")
    
    batch_type_order = ['首批', '非首批']
    batch_size_order = ['小批量', '中批量', '大批量']
    
    required_analysis_cols = [line_col, batch_type_col, batch_size_col, batch_id_col, '损耗率%']
    required_control_cols = [batch_type_col, batch_size_col, 
                            'I图上限(UCL)', 'I图下限(LCL)', 'I图中心值(CL)',
                            'MR图上限(UCL)', 'MR图下限(LCL)', 'MR图中心值(CL)']
    
    for col in required_analysis_cols:
        if col not in df_analysis_copy.columns:
            st.error(f"待分析数据缺少必要列：{col}")
            return pd.DataFrame(), pd.DataFrame()
    
    for col in required_control_cols:
        if col not in df_control_copy.columns:
            st.error(f"控制参数数据缺少必要列：{col}")
            return pd.DataFrame(), pd.DataFrame()
    
    df_control_core = df_control_copy[required_control_cols].copy()
    df_merged = pd.merge(
        df_analysis_copy,
        df_control_core,
        on=[batch_type_col, batch_size_col],
        how='left'
    )
    
    if df_merged.isnull().any().any():
        st.warning("部分数据未匹配到控制参数，已过滤")
        df_merged = df_merged.dropna(subset=required_control_cols)
    
    df_merged['I图下限(LCL)'] = df_merged['I图下限(LCL)'].apply(lambda x: max(x, 0))
    df_merged['MR图下限(LCL)'] = df_merged['MR图下限(LCL)'].apply(lambda x: max(x, 0))
    
    df_merged[['Cpk', 'Sigma水平']] = df_merged.apply(
        lambda row: pd.Series(calculate_process_sigma_and_cpk(row)),
        axis=1
    )

    df_control_limits = df_merged[[
        batch_type_col, 
        batch_size_col, 
        'I图上限(UCL)', 
        'I图下限(LCL)',
        'I图中心值(CL)',
        'MR图上限(UCL)',
        'MR图下限(LCL)',
        'MR图中心值(CL)',
        'Cpk',
        'Sigma水平'
    ]].drop_duplicates()
    
    df_control_limits.rename(columns={
        'I图上限(UCL)': 'I图控制上限',
        'I图下限(LCL)': 'I图控制下限',
        'I图中心值(CL)': 'I图控制均值',
        'MR图上限(UCL)': 'MR图控制上限',
        'MR图下限(LCL)': 'MR图控制下限',
        'MR图中心值(CL)': 'MR图控制均值',
        'Cpk': '过程能力Cpk',
        'Sigma水平': '西格玛水平'
    }, inplace=True)
    
    df_control_limits[batch_type_col] = pd.Categorical(
        df_control_limits[batch_type_col], 
        categories=batch_type_order, 
        ordered=True
    )
    df_control_limits[batch_size_col] = pd.Categorical(
        df_control_limits[batch_size_col], 
        categories=batch_size_order, 
        ordered=True
    )
    df_control_limits = df_control_limits.sort_values(
        by=[batch_type_col, batch_size_col]
    ).reset_index(drop=True)
    
    df_merged = df_merged.sort_values(by=[line_col, batch_type_col, batch_size_col, batch_id_col])
    df_merged['移动极差'] = 0.0
    df_merged['MR超限'] = False
    
    for line in df_merged[line_col].unique():
        for batch_type in batch_type_order:
            for batch_size in batch_size_order:
                mask = (df_merged[line_col] == line) & \
                        (df_merged[batch_type_col] == batch_type) & \
                        (df_merged[batch_size_col] == batch_size)
                
                group_data = df_merged[mask].copy()
                if len(group_data) < 2:
                    continue
                
                sorted_group = group_data.sort_values(by=batch_id_col)
                values = sorted_group['损耗率%'].values
                mr_values = [0]
                for i in range(1, len(values)):
                    mr_values.append(abs(values[i] - values[i-1]))
                
                df_merged.loc[sorted_group.index, '移动极差'] = mr_values
    
    df_merged['I图超限'] = (df_merged['损耗率%'] > df_merged['I图上限(UCL)']) | \
                            (df_merged['损耗率%'] < df_merged['I图下限(LCL)'])
    df_merged['MR超限'] = (df_merged['移动极差'] > df_merged['MR图上限(UCL)']) | \
                                (df_merged['移动极差'] < df_merged['MR图下限(LCL)'])
    df_merged['超限'] = df_merged['I图超限'] | df_merged['MR超限']
    
    df_merged = judge_abnormal_and_remark(df_merged)
    
    df_out_of_control = df_merged[df_merged['超限']].copy()
    df_out_of_control.rename(columns={
        'I图上限(UCL)': 'I图控制上限',
        'I图下限(LCL)': 'I图控制下限',
        'I图中心值(CL)': 'I图控制中心值',
        'MR图上限(UCL)': 'MR图控制上限',
        'MR图下限(LCL)': 'MR图控制下限',
        'MR图中心值(CL)': 'MR图控制中心值'
    }, inplace=True)
    
    df_out_of_control = standardize_data_columns(df_out_of_control, 'outlier')
    
    with st.expander(label='损耗率IMR控制图（%）'):
        valid_lines = [line for line in df_merged[line_col].unique() if len(df_merged[df_merged[line_col]==line])>0]
        
        if not valid_lines:
            st.warning("无有效线体数据，无法绘制控制图")
            return df_out_of_control, df_control_limits
        
        # 1.41.1 兼容的 tabs 用法
        tabs = st.tabs(valid_lines)
        for tab_idx, line in enumerate(valid_lines):
            with tabs[tab_idx]:
                st.subheader(f"{line} - 损耗率IMR控制图（%）")
                
                line_data = df_merged[df_merged[line_col] == line].copy()
                line_groups = []
                for bt in batch_type_order:
                    for bs in batch_size_order:
                        mask = (line_data[batch_type_col] == bt) & (line_data[batch_size_col] == bs)
                        if mask.any():
                            line_groups.append((bt, bs))
                
                if not line_groups:
                    st.warning(f"{line} 无有效批次/批量组合数据")
                    continue
                
                for bt, bs in line_groups:
                    st.markdown(f"#### {line}-{bt}-{bs}")
                    
                    group_data = line_data[
                        (line_data[batch_type_col] == bt) &
                        (line_data[batch_size_col] == bs)
                    ].sort_values(by=batch_id_col).reset_index(drop=True)
                    
                    if len(group_data) < 1:
                        st.warning(f"{line}-{bt}-{bs} 无数据")
                        continue
                    
                    cl_x = group_data['I图中心值(CL)'].iloc[0]
                    ucl_x = group_data['I图上限(UCL)'].iloc[0]
                    lcl_x = group_data['I图下限(LCL)'].iloc[0]
                    cl_mr = group_data['MR图中心值(CL)'].iloc[0]
                    ucl_mr = group_data['MR图上限(UCL)'].iloc[0]
                    lcl_mr = group_data['MR图下限(LCL)'].iloc[0]
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                    title = f"{line}-{bt}-{bs} 损耗率IMR控制图（%）"
                    fig.suptitle(title, fontsize=14, fontweight='bold')
                    
                    ax1.set_title('单值图 (I图)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('损耗率（%）', fontsize=12)
                    
                    ax1.axhline(y=cl_x, color='green', linestyle='-', linewidth=1.5, label=f'中心线(CL={cl_x:.2f}%)')
                    ax1.axhline(y=ucl_x, color='red', linestyle='--', linewidth=1.5, label=f'上控制限(UCL={ucl_x:.2f}%)')
                    ax1.axhline(y=lcl_x, color='red', linestyle='--', linewidth=1.5, label=f'下控制限(LCL={lcl_x:.2f}%)')
                    
                    x_vals = group_data[batch_id_col]
                    y_vals = group_data['损耗率%']
                    ax1.plot(x_vals, y_vals, 'o-', color='blue', markersize=4, linewidth=1, label='正常数据')
                    
                    only_i_out = group_data[(group_data['I图超限']) & (~group_data['MR超限'])]
                    both_out = group_data[(group_data['I图超限']) & (group_data['MR超限'])]
                    
                    if not only_i_out.empty:
                        ax1.scatter(only_i_out[batch_id_col], only_i_out['损耗率%'], 
                                    color='orange', marker='^', s=100, label='仅I图超限（过程稳定）', zorder=5)
                    
                    if not both_out.empty:
                        ax1.scatter(both_out[batch_id_col], both_out['损耗率%'], 
                                    color='red', marker='x', s=100, label='I+MR图超限（异常值）', zorder=6)
                    
                    y_min = 0
                    y_max = max(ucl_x + 0.5, y_vals.max() + 0.5)
                    ax1.set_ylim(y_min, y_max)
                    ax1.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
                    
                    unique_batch_ids = sorted(group_data[batch_id_col].unique())
                    tick_interval = max(1, len(unique_batch_ids) // 10)
                    ax1.set_xticks(unique_batch_ids[::tick_interval])
                    ax1.tick_params(axis='x', rotation=45, labelsize=10)
                    
                    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax1.legend(loc='upper right', fontsize=10)
                    
                    ax2.set_title('移动极差图 (MR图)', fontsize=12, fontweight='bold')
                    ax2.set_xlabel('批次序号', fontsize=12)
                    ax2.set_ylabel('移动极差（%）', fontsize=12)
                    
                    ax2.axhline(y=cl_mr, color='green', linestyle='-', linewidth=1.5, label=f'中心线(CL={cl_mr:.2f}%)')
                    ax2.axhline(y=ucl_mr, color='red', linestyle='--', linewidth=1.5, label=f'上控制限(UCL={ucl_mr:.2f}%)')
                    ax2.axhline(y=lcl_mr, color='red', linestyle='--', linewidth=1.5, label=f'下控制限(LCL={lcl_mr:.2f}%)')
                    
                    mr_vals = group_data['移动极差']
                    ax2.plot(x_vals, mr_vals, 's-', color='purple', markersize=4, linewidth=1, label='移动极差')
                    
                    only_mr_out = group_data[(~group_data['I图超限']) & (group_data['MR超限'])]
                    
                    if not only_mr_out.empty:
                        ax2.scatter(only_mr_out[batch_id_col], only_mr_out['移动极差'], 
                                    color='orange', marker='^', s=100, label='仅MR图超限（单值正常）', zorder=5)
                    
                    if not both_out.empty:
                        ax2.scatter(both_out[batch_id_col], both_out['移动极差'], 
                                    color='red', marker='x', s=100, label='I+MR图超限（异常值）', zorder=6)
                    
                    y_min_mr = 0
                    y_max_mr = max(ucl_mr + 0.5, mr_vals.max() + 0.5)
                    ax2.set_ylim(y_min_mr, y_max_mr)
                    ax2.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
                    
                    ax2.set_xticks(unique_batch_ids[::tick_interval])
                    ax2.tick_params(axis='x', rotation=45, labelsize=10)
                    
                    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax2.legend(loc='upper right', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)  # 1.41.1 兼容的绘图方式
                    plt.close(fig)
    
    st.subheader("超出控制限损耗数据（%）")
    st.dataframe(df_out_of_control, use_container_width=True)
    
    return df_out_of_control, df_control_limits

# ===================== 页面逻辑（适配1.41.1） =====================
st.header("膏体损耗率分析", divider="rainbow")
with st.expander(label='说明'):
    st.markdown('''
                1. 先在「历史数据汇总」处上传历史数据
                2. 在「月度损耗率数据分析」处上传月度数据并生成控制图
                3. 点击下载按钮，下载结果文件
                ---
                📊 损耗率计算公式：
                - 收率 = 理论 / 实际
                - 损耗率 = 1 - 收率
                ---
                ℹ️ 字段说明：
                - 批次分类
                    - 首批：同一月份相同线体下的批次与前一批次的香型不同或当前月份的首批与上月末批的香型不同
                    - 非首批：与首批条件下相反
                - 批量分类
                    - 使用K均值聚类算法, 对灌装生产批次的膏体的实际使用量进行批量分类
                ''')  

with st.container(border=True):
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col1:
        st.markdown('''
        ##### 1. 选择组织:
        ''')
    with col2:
        Org = st.selectbox(label=" ",options=["口腔-JKC","口腔-JKY"], key="org_select_page1")  # 1.41.1 必须指定唯一key
        if Org == "口腔-JKC":
            org = 'JKC'
        else:
            org = 'JKY'
        st.session_state.org_page1 = org

    col4, col5, clo6 = st.columns([1, 2.5, 1])
    with col4:
        st.markdown('''
        ##### 2. 上传文件:
        ''')
    with col5:
        uploaded_file1 = st.file_uploader(
            "历史耗用数据",
            type=["xlsx"],
            accept_multiple_files=False,
            key="file1_page1"  # 1.41.1 唯一key
        )
        uploaded_file2 = st.file_uploader(
            "月度耗用数据",
            type=["xlsx"],
            accept_multiple_files=False,
            key="file2_page1"  # 1.41.1 唯一key
        )
    
    if uploaded_file1 is not None and uploaded_file2  is not None :
        st.success("文件已上传。", icon="✅")
        try:
            # 读取历史数据
            df = pd.read_excel(uploaded_file1, engine="openpyxl", sheet_name='膏体', header=0)
            df1 = dp3.get_raw_data(df)
            st.session_state.temp_historical_file = save_df_to_tempfile(df1)
            
            # 处理历史数据
            res, batch_nodes, anova_results = dp3.batch_kmeans_clustering(df1)
            res1, df_unknownbatch, df_lowLossRate, df_highLossRate = dp3.filter_raw_data(res)
            
            st.session_state.batch_nodes_page1 = batch_nodes
            st.session_state.anova_results_page1 = anova_results
            st.session_state.temp_historical_low_loss = save_df_to_tempfile(df_lowLossRate)
            st.session_state.temp_historical_high_loss = save_df_to_tempfile(df_highLossRate)

            with st.expander("历史数据正态性检验", expanded=False):
                st.markdown('---')
                st.markdown('#### 复合管数据正态性检验')
                res1 = dp3.group_normality_test(res1, group_cols=['批次分类','批量分类'])
            
            # 计算控制图参数
            imr_params, clean_histdf = dp3.calculate_imr_control_chart_params(res1)
            imr_params = imr_params.merge(
                batch_nodes[['批量分类','区间范围']],
                on='批量分类',
                how='left'
            )
            pl = "区间范围"
            col_data = imr_params.pop(pl)
            imr_params.insert(loc=2, column=pl, value=col_data)
            
            st.session_state.IMR_params_page1 = imr_params
            st.session_state.temp_historical_filtered = save_df_to_tempfile(clean_histdf)
            
            st.success(f"✅ 历史数据处理完成：{uploaded_file1.name}")

            # 读取月度数据
            df2 = pd.read_excel(uploaded_file2, sheet_name='膏体',header=0)
            st.session_state.temp_current_month_file = save_df_to_tempfile(df2)
            
            with st.expander(label="SPC八大法则(用于观察控制图)"):
                st.markdown("""
                    - 法则 1：**单点出界**
                        - 任意 1 个点，超出 UCL 上控制限 或 LCL 下控制限
                    - 法则 2：**连续 9 点同侧**
                        - 连续 9 个点，全部在中心线的同一侧（全在上 / 全在下）
                    - 法则 3：**连续 6 点单调趋势**
                        - 连续 6 个点 持续上升 或 持续下降
                    - 法则 4：**连续 14 点交替波动**
                        - 连续 14 个点 呈现「一上一下、来回交错」规律性震荡
                    - 法则 5：**连续 3 点 2 点临近控制线**
                        - 连续 3 个点里，有 2 个点极度靠近上限 / 下限（同侧），虽未超界
                    - 法则 6：**连续 5 点 4 点远离中心线**
                        - 连续 5 个点中，4 个点明显远离中心线、偏向一侧控制限
                    - 法则 7：**连续 15 点过度集中**
                        - 连续 15 个点 全部紧贴中心线上下，波动极小、过度平稳
                    - 法则 8：**连续 8 点无集中、两侧分散**
                        - 连续 8 个点 分散在中心线两侧，没有点靠近中心线
                """)

            # 处理当月数据
            df2_processed = dp3.get_raw_data(df2)
            df_hist = load_df_from_tempfile(st.session_state.temp_historical_file)
            df_res = pd.concat([df_hist, df2_processed],axis=0, ignore_index=True)
            df_res['年月份'] = df_res['年月'].apply(dp3.convert_chinese_year_month)
            first_col = df_res.columns[0]
            new_columns = [first_col, '年月份'] + [col for col in df_res.columns if col not in [first_col, '年月份']]
            df_res = df_res[new_columns]

            # 匹配批量分类
            df2_matched = dp3.match_batch_category(df2_processed, batch_nodes)
            
            # 过滤数据
            res2, df_unknownbatch, df_lowLossRate, df_highLossRate = dp3.filter_raw_data(df2_matched)
            
            # 标准化负损耗/高损耗数据
            df_lowLossRate_standard = standardize_data_columns(df_lowLossRate, 'low_loss')
            df_highLossRate_standard = standardize_data_columns(df_highLossRate, 'high_loss')
            
            st.session_state.temp_current_month_low_loss = save_df_to_tempfile(df_lowLossRate_standard)
            st.session_state.temp_current_month_high_loss = save_df_to_tempfile(df_highLossRate_standard)
            
            # 生成控制图
            outliers, control_limits = plot_imr_control_charts(res2, imr_params)
            st.session_state.temp_current_month_outliers = save_df_to_tempfile(outliers)
            st.session_state.temp_current_month_control_limits = save_df_to_tempfile(control_limits)
            
            st.success(f"✅ 月度数据处理完成：{uploaded_file2.name}")
        except Exception as e:
            st.error(f"数据处理失败：{str(e)}")
            # 1.41.1 兼容的异常回溯
            import traceback
            st.code(traceback.format_exc(), language="python")
        
        # 下载功能（适配1.41.1）
        st.markdown('''
        ##### 3. 结果下载:
        ''')
        def create_excel():
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            
            # 从临时文件加载数据
            current_month_outliers = load_df_from_tempfile(st.session_state.temp_current_month_outliers)
            current_month_low_loss = load_df_from_tempfile(st.session_state.temp_current_month_low_loss)
            current_month_high_loss = load_df_from_tempfile(st.session_state.temp_current_month_high_loss)
            cached_historical_file = load_df_from_tempfile(st.session_state.temp_historical_file)
            batch_nodes = st.session_state.batch_nodes_page1
            IMR_params = st.session_state.IMR_params_page1
            
            # 写入异常损耗数据
            if not current_month_outliers.empty:
                df_outliers = current_month_outliers[current_month_outliers['异常值'] == True]
                df_outliers.to_excel(writer, sheet_name='月度异常损耗数据', index=False)
            
            # 写入负损耗数据
            if not current_month_low_loss.empty:
                current_month_low_loss.to_excel(writer, sheet_name='月度负损耗数据', index=False)
            
            # 写入高损耗数据
            if not current_month_high_loss.empty:
                current_month_high_loss.to_excel(writer, sheet_name='月度高损耗数据', index=False)
            
            # 写入历史数据汇总
            if 'df_res' in globals() and not df_res.empty:
                df_res.to_excel(writer, sheet_name='历史数据汇总(含当月)', index=False)
            elif not cached_historical_file.empty:
                cached_historical_file.to_excel(writer, sheet_name='历史数据汇总(含当月)', index=False)
            
            # 写入控制图参数
            if IMR_params is not None and not IMR_params.empty:
                IMR_params.to_excel(writer, sheet_name='基于历史数据的控制图参数', index=False)

            # 写入批量分类
            if batch_nodes is not None and not batch_nodes.empty:
                batch_nodes.to_excel(writer, sheet_name='批量分类-分组明细', index=False)
            
            writer.close()
            output.seek(0)
            return output.getvalue()
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 1.41.1 兼容的下载按钮
        st.download_button(
            label="下载结果",
            type="primary",
            data=create_excel(),
            file_name=f"{Org}-膏体物耗分析_{current_time}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="download_page1"  # 唯一key
        )
        
    elif uploaded_file1 is None and uploaded_file2  is not None :
        st.warning("请上传历史耗用数据！", icon="⚠️")
    elif uploaded_file2 is None and uploaded_file1  is not None :
        st.warning("请上传月度耗用数据！", icon="⚠️")
    else :
        st.info('请上传历史和月度耗用数据！', icon="ℹ️")