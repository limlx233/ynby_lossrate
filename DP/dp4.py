import re
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告


def filter1(df):
    cols  = df.columns
    if '批号' in cols:
        df = df.copy()
        df = df.dropna(subset=['批号'])
    else:
        print("数据集中不存在'批号'列")
    return df

def generate_batch_sequence(df, batch_col='批次', unit_col='生产单元', new_seq_col='批次序号'):
    """
    从批次列中提取前8位和后2位字符，组合成新批次标识，
    并在每个生产单元内按此标识排序，生成从1开始的整数序号。

    参数:
        df (pd.DataFrame): 输入数据框
        batch_col (str): 批次列名，默认 '批次'
        unit_col (str): 生产单元列名，默认 '生产单元'
        new_seq_col (str): 新生成的序号列名，默认 '批次序号'

    返回:
        pd.DataFrame: 添加了 new_seq_col 列的新 DataFrame（不修改原 df）
    """
    # 复制避免修改原数据
    df = df.copy()
    # 检查列是否存在
    if batch_col not in df.columns or unit_col not in df.columns:
        raise ValueError(f"输入DataFrame必须包含列: '{batch_col}' 和 '{unit_col}'")
    # 提取前8位 + 后2位（共10位），要求批次字符串长度 >= 10
    def extract_key(batch_str):
        if pd.isna(batch_str) or len(str(batch_str)) < 10:
            return None  # 或可抛出异常，根据业务决定
        s = str(batch_str)
        return s[:8] + s[-2:]
    df['_batch_key'] = df[batch_col].apply(extract_key)
    # 按生产单元分组，在组内按 _batch_key 排序后分配序号
    df = df.sort_values(by=[unit_col, '_batch_key'], na_position='last')
    df[new_seq_col] = df.groupby(unit_col).cumcount() + 1
    # 删除临时列
    df = df.drop(columns=['_batch_key'])
    return df

def format_month(col_value):
    # 如果是Pandas的时间戳格式
    if isinstance(col_value, pd.Timestamp):
        return col_value.strftime('%Y年%m月')
    # 如果是Python标准库的datetime格式（修复pd.datetime为datetime.datetime）
    elif isinstance(col_value, datetime.datetime):
        return col_value.strftime('%Y年%m月')
    # 如果是带时间的字符串（如"2026-01-01 00:00:00"）
    elif isinstance(col_value, str) and '-' in col_value and ':' in col_value:
        dt = pd.to_datetime(col_value)
        return dt.strftime('%Y年%m月')
    # 如果已经是"xxxx年xx月"格式，直接返回
    elif isinstance(col_value, str) and '年' in col_value and '月' in col_value:
        return col_value
    # 其他异常情况返回空值
    else:
        return None

    # ------------------- 步骤3：标记批次类型 -------------------
    # df_copy['批次类型'] = ''  # 初始化批次类型列
    
    # # 遍历每个生产线
    # for line, sorted_original_months in line_month_map.items():
    #     line_data = df_copy[df_copy['生产线'] == line].copy()
        
    #     # 遍历该生产线的每个原始年月
    #     for idx, original_month in enumerate(sorted_original_months):
    #         # 筛选当前原始年月的数据（通过标准化年月关联）
    #         std_month = line_data[line_data['年月'] == original_month]['年月_标准化'].iloc[0]
    #         month_data = line_data[line_data['年月_标准化'] == std_month].copy()
    #         month_indices = month_data.index  # 获取原df中的索引
            
    #         # 遍历该年月的每个批次
    #         for batch_idx, (row_idx, row) in enumerate(month_data.iterrows()):
    #             current_flavor = row['香型']
                
    #             # 情况1：当前是该年月的第一个批次
    #             if batch_idx == 0:
    #                 # 子情况1.1：当前是该生产线的第一个年月（无前置月份）
    #                 if idx == 0:
    #                     df_copy.loc[row_idx, '批次类型'] = '换型'
    #                 # 子情况1.2：有前置月份，取前置月份最后一个批次的香型
    #                 else:
    #                     prev_original_month = sorted_original_months[idx-1]
    #                     prev_std_month = line_data[line_data['年月'] == prev_original_month]['年月_标准化'].iloc[0]
    #                     prev_month_data = line_data[line_data['年月_标准化'] == prev_std_month]
    #                     if not prev_month_data.empty:
    #                         # 取前置月份最后一个批次的香型
    #                         prev_flavor = prev_month_data.iloc[-1]['香型']
    #                         if current_flavor != prev_flavor:
    #                             df_copy.loc[row_idx, '批次类型'] = '换型'
    #                         else:
    #                             df_copy.loc[row_idx, '批次类型'] = '连续生产'
    #                     else:
    #                         df_copy.loc[row_idx, '批次类型'] = '换型'
                
    #             # 情况2：当前是该年月的非第一个批次
    #             else:
    #                 # 取同月份前一个批次的香型
    #                 prev_batch_flavor = month_data.iloc[batch_idx-1]['香型']
    #                 if current_flavor != prev_batch_flavor:
    #                     df_copy.loc[row_idx, '批次类型'] = '换型'
    #                 else:
    #                     df_copy.loc[row_idx, '批次类型'] = '连续生产'
    
    # 删除临时的标准化年月列（仅保留原始字段）
    df_copy = df_copy.drop(columns=['年月_标准化'])
    return df_copy


def batch_kmeans_clustering(df, value_col='入库数量', max_k=5, method='elbow_silhouette'):
    """
    优化版：对DataFrame中的数值列进行K均值聚类，生成批量分类和范围区间（区间为连续整数）
    
    参数：
        df: pd.DataFrame - 输入的原始数据
        value_col: str - 用于聚类的数值列名（默认'实际'）
        max_k: int - 肘部法选K的最大K值（默认5）
        method: str - K值选择方法，'elbow_silhouette'或'elbow_only'
    
    返回：
        result_df: pd.DataFrame - 原始数据新增「批量分类」「批量范围」列
        batch_nodes: pd.DataFrame - 分类节点数据
        analysis_result: dict - 综合分析结果
    """
    # 1. 数据校验与预处理
    cluster_df = df.copy()
    if value_col not in cluster_df.columns:
        raise ValueError(f"无「{value_col}」字段！当前字段：{cluster_df.columns.tolist()}")
    
    # 转换数值并剔除空值行
    cluster_df[value_col] = pd.to_numeric(cluster_df[value_col], errors='coerce')
    cluster_df = cluster_df.dropna(subset=[value_col])
    if len(cluster_df) == 0:
        raise ValueError(f"「{value_col}」字段无有效数值！")
    if len(cluster_df) < 3:
        raise ValueError(f"数据量过少，至少需要3个样本进行聚类")
    
    # 2. 优化的K值选择算法
    def get_optimal_k(data, max_k=5, method='elbow_silhouette'):
        if len(data) < 3:
            return 2
        
        # 计算不同K值的评估指标
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(data)))  # 避免K值过大
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            
            # 计算轮廓系数
            labels = kmeans.labels_
            if len(set(labels)) > 1:  # 至少有2个不同标签才能计算轮廓系数
                silhouette_avg = silhouette_score(data, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        if method == 'elbow_silhouette':
            # 结合肘部法和轮廓系数
            # 找到轮廓系数最大值对应的K
            optimal_k_idx = np.argmax(silhouette_scores)
            optimal_k = k_range[optimal_k_idx]
            
            # 如果轮廓系数最高的K值不是2，但下降幅度较大，可以考虑更小的K
            if optimal_k > 2:
                # 计算惯性下降率
                inertia_ratios = []
                for i in range(1, len(inertias)):
                    if inertias[i-1] != 0:
                        ratio = (inertias[i-1] - inertias[i]) / inertias[i-1]
                        inertia_ratios.append(ratio)
                
                # 平衡考虑：选择轮廓系数相对较高且K值较小的
                best_k = optimal_k
            else:
                best_k = 2
        else:  # elbow_only
            # 传统肘部法
            if len(inertias) < 2:
                return 2
            
            # 计算二阶导数找肘部点
            first_diff = np.diff(inertias)
            second_diff = np.diff(first_diff)
            
            # 寻找二阶导数最大的点（肘部）
            if len(second_diff) > 0:
                elbow_idx = np.argmax(second_diff) + 2  # +2因为二阶导数索引偏移
                best_k = min(elbow_idx, len(k_range))
            else:
                best_k = 2
        
        return min(best_k, max_k)
    
    # 数据标准化
    scaler = StandardScaler()
    cluster_data = scaler.fit_transform(cluster_df[[value_col]].values)
    
    # 获取最优K值并执行聚类
    optimal_k = get_optimal_k(cluster_data, max_k=max_k, method=method)
    
    # 重新进行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data)
    cluster_df['聚类标签'] = cluster_labels
    
    # 3. 聚类结果统计与排序
    cluster_batch_stats = cluster_df.groupby('聚类标签')[value_col].agg([
        'count', 'min', 'max', 'mean', 'std', 'median'
    ]).reset_index()
    
    # 按均值排序，确保分组的逻辑顺序
    cluster_batch_stats = cluster_batch_stats.sort_values('mean').reset_index(drop=True)
    
    # 根据最优K值分配分组名称
    if optimal_k == 2:
        size_names = ['小批量', '大批量']
    elif optimal_k == 3:
        size_names = ['小批量', '中批量', '大批量']
    elif optimal_k == 4:
        size_names = ['小批量', '中批量', '中大批量', '大批量']
    elif optimal_k == 5:
        size_names = ['小批量', '中小批量', '中批量', '中大批量', '大批量']
    else:
        # 如果K值不是2-5之间的值，使用默认命名
        size_names = ['小批量', '中小批量', '中批量', '中大批量', '大批量']
    
    cluster_batch_stats['分组名称'] = [
        size_names[i] if i < len(size_names) else f'第{i+1}批' 
        for i in range(len(cluster_batch_stats))
    ]
    
    # 4. 生成连续整数的批量范围区间（无重叠、无缝衔接）
    sorted_stats = cluster_batch_stats.copy()
    range_list = []
    lower_bounds = []
    upper_bounds = []
    
    for idx, row in sorted_stats.iterrows():
        if idx == 0:
            # 第一组：从0开始（取整）
            current_min = int(np.floor(0.0))
        else:
            # 后续组：从上一组的上限 + 1 开始（确保连续整数，无重叠）
            prev_max = upper_bounds[idx-1]
            current_min = prev_max + 1
        
        # 最后一组设置为无穷大（整数形式）
        if idx == len(sorted_stats) - 1:
            current_max = float('inf')
            range_str = f"[{current_min}, ∞)"
        else:
            # 其他组：取当前组max的上取整，作为上限（确保覆盖当前组所有值）
            current_max = int(np.ceil(row['max']))
            # 修正：如果下一组min小于当前max，用下一组min的下取整-1作为上限（避免覆盖下一组）
            next_min = sorted_stats.iloc[idx+1]['min']
            if next_min <= current_max:
                current_max = int(np.floor(next_min)) - 1
            range_str = f"[{current_min}, {current_max})"
        
        range_list.append(range_str)
        lower_bounds.append(current_min)
        upper_bounds.append(current_max if idx != len(sorted_stats)-1 else float('inf'))
    
    cluster_batch_stats['区间范围'] = range_list
    cluster_batch_stats['区间下限'] = lower_bounds
    cluster_batch_stats['区间上限'] = upper_bounds
    cluster_batch_stats['分组序号'] = range(1, len(cluster_batch_stats)+1)
    
    # 5. 映射结果到原始数据
    result_df = df.copy()
    result_df['批量分类'] = "未分组"
    result_df['批量范围'] = ""
    
    # 建立标签→名称/区间的映射
    label_to_name = dict(zip(cluster_batch_stats['聚类标签'], cluster_batch_stats['分组名称']))
    label_to_range = dict(zip(cluster_batch_stats['聚类标签'], cluster_batch_stats['区间范围']))
    
    # 仅更新有有效聚类结果的行
    valid_indices = cluster_df.index
    result_df.loc[valid_indices, '批量分类'] = cluster_df['聚类标签'].map(label_to_name)
    result_df.loc[valid_indices, '批量范围'] = cluster_df['聚类标签'].map(label_to_range)
    
    # 6. 综合质量评估
    valid_cluster_df = cluster_df[cluster_df['聚类标签'].isin(cluster_batch_stats['聚类标签'])].copy()
    
    # 方差分析
    group_data = [
        valid_cluster_df[valid_cluster_df['聚类标签'] == label][value_col].values 
        for label in cluster_batch_stats['聚类标签']
    ]
    
    if len(group_data) > 1 and all(len(g) > 0 for g in group_data):
        try:
            f_stat, anova_p_value = stats.f_oneway(*group_data)
        except:
            f_stat, anova_p_value = 0, 1  # 如果方差分析失败，设为默认值
    else:
        f_stat, anova_p_value = 0, 1
    
    # 轮廓系数
    labels = cluster_df['聚类标签'].values
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(cluster_data, labels)
    else:
        silhouette_avg = 0
    
    # 聚类内距离
    cluster_inertias = []
    for label in cluster_batch_stats['聚类标签']:
        cluster_points = cluster_data[cluster_df['聚类标签'] == label]
        if len(cluster_points) > 0:
            center = kmeans.cluster_centers_[label].reshape(1, -1)
            distances = np.linalg.norm(cluster_points - center, axis=1)
            cluster_inertias.append(np.mean(distances))
        else:
            cluster_inertias.append(0)
    analysis_result = {
        'F统计量': round(f_stat, 4),
        'ANOVA_P值': round(anova_p_value, 6),
        '轮廓系数': round(silhouette_avg, 4),
        '最优K值': optimal_k,
        '聚类内平均距离': round(np.mean(cluster_inertias), 4) if cluster_inertias else 0
    }
    # 整理分类节点数据（包含区间分组信息）
    batch_nodes = cluster_batch_stats[['分组序号', '分组名称', '区间范围', '区间下限', '区间上限', 'count', 'min', 'max', 'mean', 'std', 'median']].copy()
    batch_nodes.columns = ['分组序号', '批量分类', '区间范围', '区间下限', '区间上限', '样本数', '最小值', '最大值', '均值', '标准差', '中位数']
    
    # 保留两位小数（数值类型保留，仅处理无穷大的数值表示）
    for col in ['最小值', '最大值', '均值', '标准差', '中位数', '区间下限', '区间上限']:
        if col in batch_nodes.columns:
            if col in ['区间下限', '区间上限']:
                # 区间上下限是整数/inf，无需额外处理
                batch_nodes[col] = batch_nodes[col].apply(
                    lambda x: x if pd.isna(x) or x == float('inf') or x == -float('inf') else int(x)
                )
            else:
                # 普通数值列直接保留两位小数（数值类型）
                batch_nodes[col] = batch_nodes[col].round(2)
    return result_df, batch_nodes, analysis_result

def group_normality_test(
    df: pd.DataFrame,
    group_cols: list = ['生产线', '批量分类'],
    value_col: str = '损耗率',
    min_sample_size: int = 8,  # 最小样本量（低于则跳过检验）
    figsize: tuple = (10, 8)  # 单个分组图表尺寸
) -> pd.DataFrame:
    """
    【Streamlit适配版】按指定分组列做正态性检验
    核心调整：
    1. 移除所有逐组的提示/消息打印，仅通过结果表统一展示
    2. 结构：单个 st.expander → 嵌套 st.tabs（所有分组名称）
    3. 正态性检验为分组级，结果批量赋值给行，优先展示分组汇总表
    """
    # ===================== 1. 数据校验 =====================
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集中缺少必要列：{missing_cols}")

    df_result = df.copy()
    # 创建分组名称（生产线-批量分类）
    df_result['分组名称'] = df_result[group_cols[0]].astype(str) + '-' + df_result[group_cols[1]].astype(str)
    df_clean = df_result.dropna(subset=[value_col]).copy()

    if len(df_clean) == 0:
        raise ValueError(f"数值列{value_col}无有效数据（全为空）")

    # ===================== 2. 初始化新增列 =====================
    df_result['是否正态'] = np.nan

    # ===================== 3. 批量计算所有分组的正态性检验结果（无实时提示） =====================
    normality_mapping = {}
    all_valid_group_info = []
    # 存储分组级详细结果（用于最终汇总表）
    group_detail_results = []

    # 按分组名称聚合处理
    for group_name, group_data in df_clean.groupby('分组名称'):
        batch_size = len(group_data)
        ad_stat = np.nan
        ad_crit_5pct = np.nan
        test_result = ''

        # 样本量不足的情况
        if batch_size < min_sample_size:
            test_result = '样本量不足'
            normality_mapping[group_name] = test_result
        else:
            # 正态性检验（Anderson-Darling）
            ad_stat, ad_critical_values, _ = stats.anderson(group_data[value_col], dist='norm')
            ad_crit_5pct = ad_critical_values[2]
            is_normal = ad_stat < ad_crit_5pct
            test_result = '正态' if is_normal else '非正态'
            normality_mapping[group_name] = test_result

            # 存储有效分组信息（用于绘图）
            all_valid_group_info.append({
                '分组名称': group_name,
                '数据': group_data[value_col],
                '样本量': batch_size,
                'AD统计量': ad_stat,
                '5%临界值': ad_crit_5pct,
                '是否正态': is_normal
            })

        # 存储分组级详细结果（用于最终汇总表）
        group_detail_results.append({
            '分组名称': group_name,
            '分组样本量': batch_size,
            'AD统计量': round(ad_stat, 4) if not np.isnan(ad_stat) else np.nan,
            '5%临界值': round(ad_crit_5pct, 4) if not np.isnan(ad_crit_5pct) else np.nan,
            '是否正态': test_result
        })

    # ===================== 4. 单个Expander + 全部分组Tab 绘图 =====================
    st.markdown("📊 分组正态性检验")

    if all_valid_group_info:
        tab_titles = [info['分组名称'] for info in all_valid_group_info]
        tabs = st.tabs(tab_titles)

        for tab, group_info in zip(tabs, all_valid_group_info):
            with tab:
                group_name = group_info['分组名称']
                data = group_info['数据']
                batch_size = group_info['样本量']
                is_normal = group_info['是否正态']
                ad_stat = group_info['AD统计量']
                ad_crit_5pct = group_info['5%临界值']

                # 创建单个分组的图表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
                fig.suptitle(f'分组：{group_name} 正态性检验（样本量：{batch_size}）', fontsize=14, fontweight='bold')

                # 子图1：正态概率图
                stats.probplot(data, plot=ax1)
                ax1.set_title(f'正态概率图 | 正态：{"是" if is_normal else "否"}', fontsize=12)
                ax1.set_xlabel('理论分位数')
                ax1.set_ylabel('实际分位数')
                ax1.grid(alpha=0.3)

                # 子图2：直方图 + 核密度曲线
                ax2.hist(data, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
                kde_x = np.linspace(data.min(), data.max(), 100)
                kde_y = stats.gaussian_kde(data)(kde_x)
                kde_y_scaled = kde_y * len(data) * (kde_x[1] - kde_x[0])
                ax2.plot(kde_x, kde_y_scaled, color='red', linewidth=1.5, label='核密度曲线')
                ax2.set_title(f'直方图 | AD：{ad_stat:.4f} | 5%临界值：{ad_crit_5pct:.4f}', fontsize=12)
                ax2.set_xlabel(value_col)
                ax2.set_ylabel('频数')
                ax2.legend()
                ax2.grid(alpha=0.3)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("ℹ️ 暂无有效分组数据（样本量均不足），跳过图表绘制")

    # ===================== 5. 批量赋值正态性结果（分组级 → 行级） =====================
    for group_name, normality in normality_mapping.items():
        df_result.loc[df_result['分组名称'] == group_name, '是否正态'] = normality

    # 处理空值行
    df_result.loc[df_result[value_col].isna(), '是否正态'] = '无有效数据'

    group_detail_df = pd.DataFrame(group_detail_results)
    # 按分组名称排序，方便查看
    group_detail_df = group_detail_df.sort_values('分组名称').reset_index(drop=True)
    st.dataframe(group_detail_df, use_container_width=True, hide_index=True)
    return df_result



def calculate_imr_control_chart_params(
    df,
    group_name_col='分组名称',
    normality_col='是否正态',
    loss_rate_col='损耗率'
):
    """
    两步法版：按分组计算IMR控制图参数（严格遵循AIAG/ASQ行业标准）
    两步法逻辑：原始数据算临时控制限 → 剔除异常数据 → 用清洁数据算最终控制限
    - 正态：均值 + 标准E2/D4常数
    - 非正态：中位数 + 稳健E2/D4常数（适配非正态数据）
    - MR图下限强制为0（符合移动极差定义）
    - 损耗率下限截断到0（损耗率非负）
    返回值：
        1. out：控制图参数DF（剔除异常后的最终结果）
        2. low_loss_rate_df：损耗率<0的记录DF（列名与out一致）
    """
    # ===================== 1. 输入校验（保留原有逻辑） =====================
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"输入必须是pandas.DataFrame，当前类型：{type(df)}")
    
    required_cols = [group_name_col, normality_col, loss_rate_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必填字段：{missing_cols}")
    
    if df.empty:
        raise ValueError("输入的DataFrame为空，无法计算IMR参数")

    # ===================== 2. 数据预处理（新增：筛选损耗率<0的数据） =====================
    df = df.copy()
    # 转百分比并处理数值类型
    df['损耗率%'] = pd.to_numeric(df[loss_rate_col], errors='coerce') * 100
    
    # ------------ 原有清洗逻辑：仅保留损耗率≥0的有效数据 ------------
    df = df[(df['损耗率%'] >= 0)].dropna(subset=required_cols + ['损耗率%'])

    use_cols = required_cols + ['损耗率%']
    imr_df = df.dropna(subset=use_cols).copy()  # 剔除空值
    
    if imr_df.empty:
        raise ValueError("数据清洗后无有效损耗率数值（损耗率≥0），请检查字段格式与空值")
    
    imr_df[group_name_col] = imr_df[group_name_col].astype(str).str.strip()
    imr_df[normality_col] = imr_df[normality_col].astype(str).str.strip().str.lower()

    # ===================== 3. 行业标准常数定义（保留原有逻辑） =====================
    NORMAL_CONST = {
        'E2': 2.66,    # I图常数（UCL/LCL = 均值 ± E2*MR均值）
        'D4': 3.267,   # MR图上限常数（UCL = D4*MR均值）
        'D3': 0.0,     # MR图下限常数（始终为0）
        'd2': 1.128    # 标准差估算常数（参考用）
    }
    NON_NORMAL_CONST = {
        'E2': 3.145,    # 中位数法I图常数
        'D4': 3.865,    # 中位数法MR图上限常数
        'D3': 0.0       # MR图下限始终为0
    }

    # ===================== 4. 核心函数：单分组计算控制限（保留原有逻辑） =====================
    def calc_group_limits(g_df):
        """单分组计算控制限（适配临时/最终计算）"""
        x = g_df['损耗率%'].values
        n = len(x)
        
        result = {
            '样本数(n)': n,
            'I图中心值(CL)': np.nan,
            'MR图中心值(CL)': np.nan,
            'I图上限(UCL)': np.nan,
            'I图下限(LCL)': np.nan,
            'MR图上限(UCL)': np.nan,
            'MR图下限(LCL)': 0.0,
            'X最小值': np.min(x) if n > 0 else np.nan,
            'X最大值': np.max(x) if n > 0 else np.nan,
            'MR最大值': np.nan,
            '是否正态': False
        }
        
        if n < 2:  # 样本数<2无法计算移动极差
            return pd.Series(list(result.values()), index=list(result.keys())), g_df
        
        # 计算移动极差（MR）
        mr = np.abs(x[1:] - x[:-1])
        mr_mean = np.mean(mr)  # MR图中心线
        mr_max = np.max(mr) if len(mr) > 0 else 0.0
        result['MR最大值'] = mr_max
        result['MR图中心值(CL)'] = mr_mean

        # 判断正态性
        norm_flag = g_df[normality_col].iloc[0]
        is_normal = norm_flag == '正态'
        result['是否正态'] = is_normal

        if is_normal:
            # 正态：均值 + 标准常数
            x_mean = np.mean(x)
            result['I图中心值(CL)'] = x_mean
            result['I图上限(UCL)'] = x_mean + NORMAL_CONST['E2'] * mr_mean
            result['I图下限(LCL)'] = x_mean - NORMAL_CONST['E2'] * mr_mean
            result['MR图上限(UCL)'] = NORMAL_CONST['D4'] * mr_mean
        else:
            # 非正态：中位数 + 稳健常数
            x_median = np.median(x)
            result['I图中心值(CL)'] = x_median
            result['I图上限(UCL)'] = x_median + NON_NORMAL_CONST['E2'] * mr_mean
            result['I图下限(LCL)'] = x_median - NON_NORMAL_CONST['E2'] * mr_mean
            result['MR图上限(UCL)'] = NON_NORMAL_CONST['D4'] * mr_mean

        # 损耗率下限截断到0（损耗率不能为负）
        result['I图下限(LCL)'] = max(result['I图下限(LCL)'], 0.0)
        
        return pd.Series(list(result.values()), index=list(result.keys())), g_df

    # ===================== 5. 两步法核心逻辑（保留原有逻辑） =====================
    final_result_list = []
    # 按分组遍历处理
    for group_name, group_df in imr_df.groupby(group_name_col, sort=False):
        # -------- 第一步：用原始数据计算临时控制限 --------
        temp_limits, _ = calc_group_limits(group_df)
        
        # 跳过样本数不足的分组
        if group_df.shape[0] < 2:
            final_result = pd.Series({
                group_name_col: group_name,
                '样本数(n)': group_df.shape[0],
                'I图中心值(CL)': np.nan,
                'MR图中心值(CL)': np.nan,
                'I图上限(UCL)': np.nan,
                'I图下限(LCL)': 0.0,
                'MR图上限(UCL)': np.nan,
                'MR图下限(LCL)': 0.0,
                'X最小值': np.nan,
                'X最大值': np.nan,
                'MR最大值': np.nan,
                '是否正态': group_df[normality_col].iloc[0] == '正态' if len(group_df) > 0 else False,
                '剔除异常数': 0
            })
            final_result_list.append(final_result)
            continue
        
        # 识别异常值：超出临时I图上下限的样本
        temp_ucl = temp_limits['I图上限(UCL)']
        temp_lcl = temp_limits['I图下限(LCL)']
        # 标记异常行
        group_df['是否异常'] = ~group_df['损耗率%'].between(temp_lcl, temp_ucl)
        # 剔除异常数据
        clean_group_df = group_df[~group_df['是否异常']].copy()
        # 记录剔除的异常数
        outlier_count = group_df['是否异常'].sum()
        
        # 处理剔除后样本数不足的情况（至少保留2个样本）
        if len(clean_group_df) < 2:
            # 若剔除后样本不足，使用原始数据（避免无法计算）
            clean_group_df = group_df.copy()
            outlier_count = 0  # 标记为未剔除
        
        # -------- 第二步：用清洁数据计算最终控制限 --------
        final_limits, _ = calc_group_limits(clean_group_df)
        
        # 整理最终结果（补充分组名称、剔除异常数）
        final_result = final_limits.to_dict()
        final_result[group_name_col] = group_name
        final_result['剔除异常数'] = outlier_count
        final_result_list.append(pd.Series(final_result))

    # ===================== 6. 结果整理与格式化（保留原有逻辑） =====================
    out = pd.DataFrame(final_result_list)
    # 调整列顺序（提升可读性）
    col_order = [
        group_name_col, '样本数(n)',
        'I图中心值(CL)', 'I图下限(LCL)', 'I图上限(UCL)',
        'MR图中心值(CL)', 'MR图下限(LCL)', 'MR图上限(UCL)',
        'X最小值', 'X最大值', 'MR最大值', '剔除异常数', '是否正态',
    ]
    out = out[col_order]
    
    # 数值格式化（保留4位小数）
    float_cols = [c for c in out.columns if c not in [group_name_col, '是否正态']]
    out[float_cols] = out[float_cols].round(4)

    # 强制MR图下限为0（最终校验）
    out['MR图下限(LCL)'] = 0.0
    
    # ===================== . 返回两个结果 =====================
    return out


def match_batch_category_by_quantity(
    monthly_data: pd.DataFrame,
    batch_category_rules: pd.DataFrame,
    quantity_col: str = "入库数量",
    category_col: str = "批量分类",
    lower_limit_col: str = "区间下限",
    upper_limit_col: str = "区间上限"
) -> pd.DataFrame:
    """
    仅为月度数据匹配批量分类（核心功能）
    适配场景：除批量分类外，其他字段均为数字
    
    参数说明：
    ----------
    monthly_data : pd.DataFrame
        月度数据（需要匹配批量分类的原始数据）
    batch_category_rules : pd.DataFrame
        历史生成的批量分类规则（st.session_state.batch_nodes_fhg_p5）
    quantity_col : str, optional
        月度数据中入库数量的列名（数值型），默认"入库数量"
    category_col : str, optional
        新增的批量分类列名，默认"批量分类"
    lower_limit_col : str, optional
        批量分类规则中区间下限列名，默认"区间下限"
    upper_limit_col : str, optional
        批量分类规则中区间上限列名，默认"区间上限"
    
    返回值：
    ----------
    pd.DataFrame
        新增「批量分类」列的月度数据（未匹配标记为"未分类"）
    """
    # 防御性拷贝，不修改原数据
    df = monthly_data.copy()
    
    # 空值处理
    if df.empty or batch_category_rules is None or batch_category_rules.empty:
        df[category_col] = "未分类"
        return df
    
    # 关键列检查
    if quantity_col not in df.columns:
        df[category_col] = "未分类"
        return df
    if not all(col in batch_category_rules.columns for col in [category_col, lower_limit_col, upper_limit_col]):
        df[category_col] = "未分类"
        return df
    
    # 入库数量转数值型（适配数字字段特性）
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')
    
    # 初始化批量分类列
    df[category_col] = "未分类"
    
    # 核心匹配逻辑：遍历规则匹配区间
    for _, rule in batch_category_rules.iterrows():
        lower = float(rule[lower_limit_col]) if pd.notna(rule[lower_limit_col]) else -np.inf
        upper = float(rule[upper_limit_col]) if pd.notna(rule[upper_limit_col]) else np.inf
        
        # 匹配入库数量区间
        match_mask = (df[quantity_col] >= lower) & (df[quantity_col] <= upper) & (df[category_col] == "未分类")
        df.loc[match_mask, category_col] = rule[category_col]
    return df



# 依赖函数（需保留）
def judge_abnormal_and_remark(df):
    """判定异常值：I+MR都超限=异常"""
    df_copy = df.copy()
    df_copy['异常值'] = False
    df_copy['超限备注'] = "正常"
    
    both_out_mask = (df_copy['I图超限']) & (df_copy['MR超限'])
    df_copy.loc[both_out_mask, '异常值'] = True
    df_copy.loc[both_out_mask, '超限备注'] = "I+MR图超限（异常值）"
    
    only_i_out_mask = (df_copy['I图超限']) & (~df_copy['MR超限'])
    df_copy.loc[only_i_out_mask, '超限备注'] = "仅I图超限（过程稳定）"
    
    only_mr_out_mask = (~df_copy['I图超限']) & (df_copy['MR超限'])
    df_copy.loc[only_mr_out_mask, '超限备注'] = "仅MR图超限（单值正常）"
    
    return df_copy

def standardize_data_columns(df, data_type):
    """标准化输出列格式"""
    df_copy = df.copy()
    base_cols = [
        '年月份', '产品说明', '产品批号', '线体', '香型', '规格', '产品',
        '批次分类', '批量分类', '分组名称', '实际', '理论', '损耗率', '损耗率%',
        '异常值', '超限备注'
    ]
    
    if data_type == 'outlier':
        extra_cols = [
            'I图控制上限', 'I图控制下限', 'I图控制中心值',
            'MR图控制上限', 'MR图控制下限', 'MR图控制中心值'
        ]
    elif data_type == 'negative_loss':
        extra_cols = []
    else:
        extra_cols = []
    
    all_cols = [col for col in base_cols + extra_cols if col in df_copy.columns]
    return df_copy[all_cols]



def calculate_sigma_level(row,
                        cl_col='I图中心值(CL)', 
                        ucl_col='I图上限(UCL)', 
                        lcl_col='I图下限(LCL)', 
                        loss_rate_col='损耗率'):
    """
    计算「单行损耗率对应的实际西格玛水平」（过程能力Z值）
    核心：基于整过程的标准差，计算当前行损耗率满足业务规格限的能力
    输入：
        row: DataFrame行
        usl: 业务规格上限（必填，如损耗率最大允许5%则传5）
        lsl: 业务规格下限（默认0，如损耗率最小允许0%）
        cl_col: 过程均值列名（控制图CL）
        ucl_col: 过程上限列名（控制图UCL）
        lcl_col: 过程下限列名（控制图LCL）
        loss_rate_col: 单行损耗率列名
    返回：
        西格玛水平（保留1位小数，如"4.5σ"），异常返回np.nan
    """
    try:
        
        # 1. 提取整过程的控制图参数（计算过程标准差，整过程唯一）
        # 清理控制图参数中的非数值字符（如'-'）并转为浮点数
        cl = float(str(row[cl_col]).replace('-', '')) if pd.notna(row[cl_col]) else np.nan
        ucl = float(str(row[ucl_col]).replace('-', '')) if pd.notna(row[ucl_col]) else np.nan
        lcl = float(str(row[lcl_col]).replace('-', '')) if pd.notna(row[lcl_col]) else np.nan
        usl = ucl
        lsl = 0
        # 2. 提取当前行的损耗率值（核心修改点）
        current_loss_rate = float(str(row[loss_rate_col]).replace('-', '')) if pd.notna(row[loss_rate_col]) else np.nan
        
        # 3. 校验必要参数是否有效
        if pd.isna(ucl) or pd.isna(lcl) or pd.isna(current_loss_rate):
            return np.nan
        
        # 4. 计算过程标准差（控制图核心：UCL-LCL=6σ，整过程稳定值）
        control_limit_width = ucl - lcl
        if np.isclose(control_limit_width, 0):  # 避免除以0
            return np.nan
        process_sigma = control_limit_width / 6  # 过程真实标准差
        
        # 5. 计算当前损耗率的Z值（偏离业务规格限的程度）
        # Z值定义：(规格限 - 实际值) / 过程标准差
        z_upper = (usl - current_loss_rate) / process_sigma  # 距离上规格限的西格玛数
        z_lower = (current_loss_rate - lsl) / process_sigma  # 距离下规格限的西格玛数
        
        # 6. 取最小Z值（更严格的一侧，代表当前损耗率的过程能力）
        z_min = min(z_upper, z_lower)
        
        # 7. 计算最终西格玛水平（加1.5σ偏移，六西格玛标准）
        actual_sigma_level = z_min + 1.5
        actual_sigma_level = max(actual_sigma_level, 0)  # 确保非负
        
        # 8. 格式化为标准展示形式
        return f"{actual_sigma_level:.1f}σ"
    
    # 处理所有异常情况（类型错误、键错误、数值转换失败等）
    except (ValueError, TypeError, KeyError):
        return np.nan

def judge_abnormal_and_remark(df):
    """异常判定+备注+Sigma（逐行判断，彻底消除数组歧义）"""
    df_copy = df.copy()
    df_copy['异常值'] = False
    df_copy['超限备注'] = "正常"
    # df_copy['sigma水平'] = np.nan

    for idx in df_copy.index:
        # 布尔状态强转标量
        i_out = bool(df_copy.loc[idx, 'I图超限'])
        mr_out = bool(df_copy.loc[idx, 'MR超限'])

        if i_out and mr_out:
            df_copy.loc[idx, '异常值'] = True
            df_copy.loc[idx, '超限备注'] = "单值超限且过程波动异常（异常值）"
        elif i_out and not mr_out:
            df_copy.loc[idx, '超限备注'] = "单值超限、但过程稳定"
        elif not i_out and mr_out:
            df_copy.loc[idx, '超限备注'] = "单值未超限，过程有波动"

    return df_copy

def standardize_data_columns(df, data_type,):
    """输出字段标准化（完全匹配你原始字段）"""
    df_copy = df.copy()

    # 你真实原始字段顺序
    base_cols = [
        '年月', '任务单', '批号', '生产线', '物料编码', '名称',
        '入库数量', '耗用数', '损耗率', '损耗率%', '批量分类',
        '分组名称', '批号次序'
    ]

    if data_type == 'outlier':
        extra_cols = [
            '异常值', '超限备注',
            'I图控制上限', 'I图控制下限', 'I图控制中心值',
            'MR图控制上限', 'MR图控制下限', 'MR图控制中心值',
            'I图超限', 'MR超限', '超限',
            '样本数(n)', 'X最小值', 'X最大值', 'MR最大值', '是否正态'
        ]
    elif data_type == 'low_loss':
        extra_cols = ['异常值', '超限备注']
        df_copy['异常值'] = True
        df_copy['超限备注'] = "负损耗（损耗率<0）"
        df_copy['sigma水平'] = ""
    elif data_type == 'high_loss':
        extra_cols = ['异常值', '超限备注']
        df_copy['异常值'] = True
        df_copy['超限备注'] = "高损耗（损耗率>30%）"
        df_copy['sigma水平'] = ""
    else:
        extra_cols = []

    # 只保留存在的列
    base_cols = [c for c in base_cols if c in df_copy.columns]
    extra_cols = [c for c in extra_cols if c in df_copy.columns]
    all_cols = base_cols + extra_cols

    if '损耗率%' not in df_copy.columns and '损耗率' in df_copy.columns:
        df_copy['损耗率%'] = df_copy['损耗率'].astype(float) * 100
    df_res = df_copy[all_cols].copy()
    df_res['年月'] = df_res['年月'].apply(format_month)
    return df_res

def plot_imr_control_charts(
    df_analysis: pd.DataFrame,
    df_control_params: pd.DataFrame,
    category: str
):
    """
    适配真实字段的IMR控制图主函数（使用数据源自带批次序号）
    原始字段：年月,任务单,批号,生产线,物料编码,名称,入库数量,耗用数,损耗率,批量分类,批次序号
    参数字段：分组名称,样本数(n),I图中心值(CL),MR图中心值(CL),I图上限(UCL),I图下限(LCL),MR图上限(UCL),MR图下限(LCL),X最小值,X最大值,MR最大值,是否正态
    """
    df_raw = df_analysis.copy(deep=True)
    df_param = df_control_params.copy(deep=True)

    # ===================== 固定映射真实字段 =====================
    line_col        = '生产线'
    batch_size_col  = '批量分类'
    loss_rate_col   = '损耗率'
    batch_id_col    = '批次序号' 

    # 初始化负损耗DF（避免未定义）
    df_low_loss_raw = pd.DataFrame()

    # ===================== 1. 原始数据预处理 =====================
    # 必须字段检查（新增批次序号检查）
    raw_required = [line_col, batch_size_col, loss_rate_col, batch_id_col]
    missing_raw = [c for c in raw_required if c not in df_raw.columns]
    if missing_raw:
        st.error(f"原始数据缺少字段：{missing_raw}（需包含批次序号）")
        # 异常分支：返回3个空DF
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    param_required = [
        '分组名称', 'I图中心值(CL)', 'MR图中心值(CL)',
        'I图上限(UCL)', 'I图下限(LCL)', 'MR图上限(UCL)', 'MR图下限(LCL)'
    ]
    missing_param = [c for c in param_required if c not in df_param.columns]
    if missing_param:
        st.error(f"参数表缺少字段：{missing_param}")
        # 异常分支：返回3个空DF
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 损耗率%转换
    df_raw['损耗率%'] = df_raw[loss_rate_col].astype(float) * 100
    # 生产线格式处理
    df_raw[line_col] = df_raw[line_col].astype(str).str.strip()
    # 生成分组名称：生产线+批量分类
    df_raw['分组名称'] = df_raw[line_col].astype(str) + "-" + df_raw[batch_size_col].astype(str)

    # 关键修改：不再生成批号次序，直接使用自带的批次序号
    df_raw[batch_id_col] = pd.to_numeric(df_raw[batch_id_col], errors='coerce')
    # 按生产线+分组+批次序号排序
    df_raw = df_raw.sort_values(by=[line_col, '分组名称', batch_id_col]).reset_index(drop=True)

    #筛选负损耗数据
    df_low_loss_raw = df_raw[df_raw['损耗率%']<0].copy()
    df_low_loss_raw = standardize_data_columns(df_low_loss_raw, 'low_loss')
    
    df_raw = df_raw[df_raw['损耗率%']>=0].copy()

    # ===================== 2. 合并参数表（按分组名称匹配） =====================
    df_merged = pd.merge(
        df_raw,
        df_param,
        on='分组名称',
        how='left'
    )

    # 过滤无参数行
    before = len(df_merged)
    df_merged = df_merged.dropna(subset=param_required)
    if len(df_merged) < before:
        st.warning(f"过滤无参数数据：{before - len(df_merged)} 行")
    if len(df_merged) == 0:
        st.warning("无匹配参数数据")
        # 异常分支：返回3个DF（前两个空，第三个是负损耗数据）
        return pd.DataFrame(), pd.DataFrame(), df_low_loss_raw

    # 控制限下限非负
    df_merged['I图下限(LCL)'] = df_merged['I图下限(LCL)'].astype(float).clip(lower=0)
    df_merged['MR图下限(LCL)'] = df_merged['MR图下限(LCL)'].astype(float).clip(lower=0)

    # ===================== 3. 计算移动极差（基于真实批次序号排序） =====================
    df_merged = df_merged.sort_values(by=[line_col, '分组名称', batch_id_col])
    df_merged['移动极差'] = 0.0

    for (line, gname), gdf in df_merged.groupby([line_col, '分组名称']):
        gdf_sorted = gdf.sort_values(batch_id_col) 
        if len(gdf_sorted) < 2:
            continue
        vals = gdf_sorted['损耗率%'].astype(float).values
        mr = [0.0] + [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        df_merged.loc[gdf_sorted.index, '移动极差'] = mr

    # ===================== 4. 超限判定 =====================
    df_merged['I图超限'] = (df_merged['损耗率%'] > df_merged['I图上限(UCL)']) | (df_merged['损耗率%'] < df_merged['I图下限(LCL)'])
    df_merged['MR超限'] = (df_merged['移动极差'] > df_merged['MR图上限(UCL)']) | (df_merged['移动极差'] < df_merged['MR图下限(LCL)'])
    df_merged['超限'] = df_merged['I图超限'] | df_merged['MR超限']
    
    # ===================== 5. 异常判定 & Sigma =====================
    df_merged = judge_abnormal_and_remark(df_merged)
    
    # ===================== 6. 整理输出 =====================
    rename_map = {
        'I图上限(UCL)': 'I图控制上限', 'I图下限(LCL)': 'I图控制下限', 'I图中心值(CL)': 'I图控制中心值',
        'MR图上限(UCL)': 'MR图控制上限', 'MR图下限(LCL)': 'MR图控制下限', 'MR图中心值(CL)': 'MR图控制中心值'
    }
    df_merged = df_merged.rename(columns=rename_map)
    df_out = df_merged[df_merged['超限']].copy()
    df_out = standardize_data_columns(df_out, 'outlier')
    
    # 控制限汇总表
    df_ctrl_summary = df_merged[[
        '分组名称', '样本数(n)', 'I图控制中心值', 'I图控制上限', 'I图控制下限',
        'MR图控制中心值', 'MR图控制上限', 'MR图控制下限',
        'X最小值', 'X最大值', 'MR最大值', '是否正态'
    ]].drop_duplicates().reset_index(drop=True)

    # ===================== 7. 绘图 =====================
    st.markdown(f"#### {category}-损耗率IMR控制图(%)")
    valid_lines = [ln for ln in df_merged[line_col].unique() if len(df_merged[df_merged[line_col]==ln])>0]
    if not valid_lines:
        st.warning("无有效生产线数据")
        # 异常分支：返回3个DF
        return df_out, df_ctrl_summary, df_low_loss_raw
    
    tabs = st.tabs(valid_lines)
    for i, line in enumerate(valid_lines):
        with tabs[i]:
            st.subheader(f"{line}")
            line_df = df_merged[df_merged[line_col]==line].copy()
            groups = line_df['分组名称'].unique()
            for gname in groups:
                gdf = line_df[line_df['分组名称']==gname].sort_values(batch_id_col).reset_index(drop=True)
                if len(gdf) < 1:
                    continue
                st.markdown(f"#### {category}-{gname}")
                cl_x   = float(gdf['I图控制中心值'].iloc[0])
                ucl_x  = float(gdf['I图控制上限'].iloc[0])
                lcl_x  = float(gdf['I图控制下限'].iloc[0])
                cl_mr  = float(gdf['MR图控制中心值'].iloc[0])
                ucl_mr = float(gdf['MR图控制上限'].iloc[0])
                lcl_mr = float(gdf['MR图控制下限'].iloc[0])
                fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,12))
                fig.suptitle(f"{category}-{gname} I-MR控制图", fontweight='bold')
                # I图（X轴使用真实批次序号）
                ax1.set_title("单值图(I图)")
                ax1.set_ylabel("损耗率(%)")
                ax1.axhline(cl_x, c='g', label=f'CL={cl_x:.2f}')
                ax1.axhline(ucl_x, c='r', ls='--', label=f'UCL={ucl_x:.2f}')
                ax1.axhline(lcl_x, c='r', ls='--', label=f'LCL={lcl_x:.2f}')
                x = gdf[batch_id_col].values  
                y = gdf['损耗率%'].values
                ax1.plot(x, y, 'o-', c='blue')
                only_i = gdf[gdf['I图超限'] & ~gdf['MR超限']]
                both   = gdf[gdf['I图超限'] & gdf['MR超限']]
                if not only_i.empty:
                    ax1.scatter(only_i[batch_id_col], only_i['损耗率%'], c='orange', marker='^', s=100, label='仅I超限')
                if not both.empty:
                    ax1.scatter(both[batch_id_col], both['损耗率%'], c='red', marker='x', s=100, label='异常值')
                y_max = max(ucl_x + 0.5, y.max() + 0.5) if len(y) > 0 else ucl_x + 0.5
                ax1.set_ylim(0, y_max)
                ax1.legend()
                ax1.grid(alpha=0.3)
                # MR图
                ax2.set_title("移动极差图(MR图)")
                ax2.set_xlabel("批次序号")  
                ax2.set_ylabel("MR")
                ax2.axhline(cl_mr, c='g', label=f'CL={cl_mr:.2f}')
                ax2.axhline(ucl_mr, c='r', ls='--', label=f'UCL={ucl_mr:.2f}')
                ax2.axhline(lcl_mr, c='r', ls='--', label=f'LCL={lcl_mr:.2f}')
                mr = gdf['移动极差'].values
                ax2.plot(x, mr, 's-', c='purple')
                only_mr = gdf[~gdf['I图超限'] & gdf['MR超限']]
                if not only_mr.empty:
                    ax2.scatter(only_mr[batch_id_col], only_mr['移动极差'], c='orange', marker='^', s=100)
                if not both.empty:
                    ax2.scatter(both[batch_id_col], both['移动极差'], c='red', marker='x', s=100)
                mr_max = max(ucl_mr + 0.5, mr.max() + 0.5) if len(mr) > 0 else ucl_mr + 0.5
                ax2.set_ylim(0, mr_max)
                ax2.legend()
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    st.write(f"{category}-控制图超限数据明细")
    st.dataframe(df_out, use_container_width=True, hide_index=True)
    st.write(f"{category}-负损耗数据明细")
    st.dataframe(df_low_loss_raw, use_container_width=True, hide_index=True)
    
    # 正常分支：返回3个DF
    return df_out, df_ctrl_summary, df_low_loss_raw