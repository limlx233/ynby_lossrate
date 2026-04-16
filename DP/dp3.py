# ==== 膏体损耗率相关函数 =====
import re
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



# 提取香型
def standardize_flavor(product_desc, original_flavor):
    extract_flavor = ""
    if not pd.isna(product_desc):
        product_desc = str(product_desc)
        match = re.search(r'牙膏(.*?)香型', product_desc)
        if match:
            extract_flavor = match.group(1).strip()
    
    if extract_flavor == "薄荷清爽":
        extract_flavor = "薄荷"
    if "抗敏" in extract_flavor:
        extract_flavor = "抗敏 水润薄荷"
    
    if extract_flavor != "":
        return extract_flavor
    elif not pd.isna(original_flavor) and str(original_flavor).strip() != "":
        return str(original_flavor).strip()
    else:
        return "-"
    

# 提取规格
def extract_specification(df):
    """
    从DataFrame的"产品说明"列提取数字，填充到"规格"列
    参数:
        df: 包含"产品说明"列的DataFrame
    返回:
        新增/更新"规格"列后的DataFrame
    """
    # 复制原DataFrame，避免修改原始数据
    df_copy = df.copy()
    
    def _extract_num(product_desc):
        """内部函数：提取单个产品说明中的数字"""
        # 处理空值/非字符串情况
        if pd.isna(product_desc) or product_desc is None:
            return "-"
        
        # 强制转换为字符串并去除空白
        desc_str = str(product_desc).strip()
        
        # 正则匹配所有数字，取第一个匹配到的（优先提取连续数字）
        match = re.search(r'\d+', desc_str)
        
        # 找到数字则返回，否则返回"-"
        return match.group() if match else "-"
    
    # 应用提取逻辑到"产品说明"列，结果填充到"规格"列
    df_copy['规格'] = df_copy['产品说明'].apply(_extract_num)
    
    return df_copy

# 提取线体
def extract_line(product_batch):
    # 先统一处理空值/非字符串情况
    if pd.isna(product_batch) or product_batch is None:
        return "未知线体"
    
    # 强制转换为字符串，并去除首尾空白（避免特殊字符干扰）
    batch_str = str(product_batch).strip()
    
    # 只匹配第一个字母（大小写均可）
    match = re.search(r'[A-Za-z]', batch_str)
    
    # 找到则返回字母，否则返回未知线体
    return match.group() if match else "未知线体"

# 将原数据中的年月6位数字 转化为 日期
def convert_year_month(year_month_str):
        if pd.isna(year_month_str) or not str(year_month_str).isdigit() or len(str(year_month_str)) != 6:
            return "未知年月"
        try:
            dt = datetime.strptime(str(year_month_str), '%Y%m')
            return dt.strftime('%Y年%m月')
        except:
            return "未知年月"

def convert_chinese_year_month(year_month_chinese):
    """
    将中文年月格式（如"2022年03月"）反向转换为6位数字格式（如202203）
    与 convert_year_month 函数功能互逆
    
    参数:
        year_month_chinese: 中文年月字符串（如"2022年03月"、"2023年1月"）
    
    返回:
        6位数字格式的年月（int类型，如202203）；转换失败返回"未知年月"
    """
    # 处理空值
    if pd.isna(year_month_chinese):
        return "未知年月"
    
    # 转为字符串并去除首尾空白
    year_month_str = str(year_month_chinese).strip()
    
    # 正则匹配中文年月格式（支持 2022年03月 / 2022年3月 两种格式）
    # 匹配组1：4位年份，组2：1-2位月份
    pattern = r'^(\d{4})年(\d{1,2})月$'
    match = re.match(pattern, year_month_str)
    
    if not match:
        return "未知年月"
    
    try:
        # 提取年份和月份
        year = match.group(1)
        month = match.group(2)
        
        # 校验月份是否合法（1-12）
        month_int = int(month)
        if month_int < 1 or month_int > 12:
            return "未知年月"
        
        # 月份补零（确保2位数），拼接为6位数字并转为int
        month_padded = month.zfill(2)  # 3 → 03，12 → 12
        result = int(f"{year}{month_padded}")
        
        return result
    except:
        return "未知年月"
        
# 提取批次序号
def extract_batch_order(product_batch):
    if pd.isna(product_batch) or product_batch is None:
        return 1
    
    batch_str = str(product_batch).strip()
    # 匹配第一个字母 + 后续1-2位数字（兼容1位/2位数字场景）
    match = re.search(r'[A-Za-z](\d{1,2})', batch_str)
    
    if match:
        try:
            # 转换为整数（自动去前导0），并校验是否大于0
            num = int(match.group(1))
            return num if num > 0 else 1
        except (ValueError, TypeError):
            # 极端情况：数字转换失败，返回1
            return 1
    # 无匹配时返回1
    return 1


def batch_kmeans_clustering(df, value_col='实际', max_k=5, method='elbow_silhouette'):
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


def get_raw_data(df):
    '''
    Docstring for get_raw_data
    用于处理dataframe 获取香型、线体、年月 并对批次分类进行标识
    :param df: pd.dataframe
    '''
    df1 = df.copy()
    df1['原始行索引'] = df1.index
    df1['原始香型'] = df1['香型'].copy()
    df1['香型'] = df1.apply(lambda row: standardize_flavor(row['产品说明'], row['原始香型']), axis=1)
    df1 = df1.drop('原始香型', axis=1)
    df1[['收率']] =  (df1['理论'] / df1['实际']).to_frame() #
    df1.loc[:,'损耗率'] = 1 - df1['收率'] #
    df1['年月'] = df1['年月份'].apply(convert_year_month)
    df1['线体'] = df1['产品批号'].apply(extract_line)
    df1 = extract_specification(df1)
    df1['批号次序'] = df1['产品批号'].apply(extract_batch_order)
    df1['批次分类'] = "非首批"
    # 全局排序（线体→年月→批号次序→原始行索引）
    df1 = df1.sort_values(by=['线体', '年月', '批号次序', '原始行索引']).reset_index(drop=True)
    # 遍历每个线体
    for line in df1['线体'].unique():
        line_df = df1[df1['线体'] == line].copy()
        months = sorted(line_df['年月'].unique())
        # 遍历每个年月
        for month_idx, month in enumerate(months):
            month_df = line_df[line_df['年月'] == month].copy()
            # 按“批号次序+原始行索引”排序（解决批号次序重复的情况）
            month_df = month_df.sort_values(by=['批号次序', '原始行索引']).reset_index(drop=True)
            # 获取该年月下的所有批次（含重复批号次序）
            batches = month_df.to_dict('records')
            # 遍历每个批次（按排序后的顺序）
            for i, current_batch in enumerate(batches):
                current_flavor = current_batch['香型']
                # 取当前批次在原始df中的索引（用于赋值）
                current_idx_in_df = current_batch['原始行索引']
                batch_order = current_batch['批号次序']
                # ---------------------- 首批判定逻辑（满足任一即可） ----------------------
                is_first = False
                # 条件1：当前批次与同月份上一个相邻批次的香型不一致
                if i > 0:
                    prev_flavor = batches[i-1]['香型']
                    if prev_flavor != current_flavor:
                        is_first = True
                # 条件2：当前是本月第一个批次，且与同线体上月最后一个批次的香型不同
                else:
                    # 确认当前是本月第一个批次（i=0）
                    # 检查是否有上月数据
                    if month_idx > 0:
                        prev_month = months[month_idx-1]
                        prev_month_df = line_df[line_df['年月'] == prev_month]
                        if not prev_month_df.empty:
                            # 获取上月最后一个批次（按排序后的最后一条）
                            prev_last_row = prev_month_df.sort_values(by=['批号次序', '原始行索引']).iloc[-1]
                            prev_last_flavor = prev_last_row['香型']
                            if prev_last_flavor != current_flavor:
                                is_first = True
                        else:
                            # 上月无数据，本月第一个批次直接判定为首批
                            is_first = True
                    else:
                        # 该线体第一个年月，本月第一个批次直接判定为首批
                        is_first = True
                # ---------------------- 仅赋值“首批”，默认已为“非首批” ----------------------
                if is_first:
                    df1.loc[df1['原始行索引'] == current_idx_in_df, '批次分类'] = "首批"
    # 清理临时列
    df1 = df1.drop(['原始行索引'], axis=1)
    df1 = df1.drop(['年月份'], axis=1)
    res = df1
    return res


def filter_raw_data(df):
    df1 = df.copy()
    # between的第三个参数inclusive='left' 表示包含左边界（0）、不包含右边界（0.3）
    df_unknownbatch = df1[df1['线体'] == '未知线体']
    df_lowLossRate = df1[df1['损耗率'] < 0]
    df_highLossRate = df1[df1['损耗率'] >= 0.3]
    df1 = df1[df1['损耗率'].between(0, 0.3, inclusive='left') & (df1['线体'] != '未知线体')]
    return df1, df_unknownbatch, df_lowLossRate, df_highLossRate


def calculate_imr_control_chart_params(
    df, 
    batch_category_col='批次分类', 
    volume_category_col='批量分类',
    line_col='线体',
    loss_rate_col='损耗率',
    normality_col='是否正态',  # 正态 / 非正态
    batch_order_col='批号次序'
):
    """
    工业标准 IMR 控制图（正态=均值法，非正态=中位数法）
    权威常数来源：AIAG / ASQ 稳健控制图标准
    返回：控制图参数DF, 剔除异常后的干净数据DF
    """
    required_cols = [batch_category_col, volume_category_col, loss_rate_col, normality_col, batch_order_col]
    df = df.copy()

    # 缺失值处理
    if batch_order_col not in df.columns:
        df[batch_order_col] = range(len(df))

    # 转为百分比
    df['损耗率%'] = df[loss_rate_col] * 100

    df = df[(df['损耗率%'] >= 0)].dropna(subset=required_cols + ['损耗率%'])
    df['损耗率%'] = pd.to_numeric(df['损耗率%'], errors='coerce')
    df = df.dropna(subset=['损耗率%'])
    
    if len(df) == 0:
        raise ValueError("无有效数据（业务规则过滤后为空）")

    # 正态（均值法）
    NORMAL_CONST = {
        'E2': 2.66,
        'D4': 3.267,
        'D3': 0.0
    }
    # 非正态（中位数法，稳健常数）
    NON_NORMAL_CONST = {
        'E2': 3.145,
        'D4': 3.865,
        'D3': 0.0
    }

    # -------------------------------------------------------------------------
    # 核心计算：正态=均值 | 非正态=中位数
    # -------------------------------------------------------------------------
    def calculate_single_imr(data, normality="正态"):
        data = data.sort_values(batch_order_col).reset_index(drop=True)
        n = len(data)
        if n < 2:
            return n, [np.nan]*8
        x = data['损耗率%'].values
        mr = np.abs(x[1:] - x[:-1])
        # 根据正态性选择参数
        if normality == "正态":
            CL = np.mean(x)           
            MR_CL = np.mean(mr) if len(mr) > 0 else np.nan
            c = NORMAL_CONST
        else:
            CL = np.median(x)        
            MR_CL = np.median(mr) if len(mr) > 0 else np.nan
            c = NON_NORMAL_CONST

        I_UCL = CL + c['E2'] * MR_CL
        I_LCL = CL - c['E2'] * MR_CL
        MR_UCL = c['D4'] * MR_CL
        MR_LCL = c['D3'] * MR_CL

        return n, CL, MR_CL, I_UCL, I_LCL, MR_UCL, MR_LCL, x, mr

    # -------------------------------------------------------------------------
    # 分组计算 + 返回 clean_data（关键修改）
    # -------------------------------------------------------------------------
    def final_group_calc(group):
        normality = group[normality_col].iloc[0]
        n1, cl1, mr1, u1, l1, mru1, mrl1, x1, _ = calculate_single_imr(group, normality)
        
        if n1 < 2:
            # 数据不足时，返回空值 + 空df
            return pd.Series([
                n1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, pd.DataFrame()
            ])
        
        # 剔除异常点
        clean_data = group[(x1 >= l1) & (x1 <= u1)].copy()
        
        # 第二次最终计算
        n_fin, cl_fin, mr_fin, u_fin, l_fin, mru_fin, mrl_fin, _, _ = calculate_single_imr(clean_data, normality)

        # ===================== 这里增加返回 clean_data =====================
        return pd.Series([
            n_fin, round(cl_fin,4), round(mr_fin,4),
            round(u_fin,4), round(l_fin,4),
            round(mru_fin,4), round(mrl_fin,4),
            clean_data  # 把干净数据一起返回
        ])

    # -------------------------------------------------------------------------
    # 执行分组计算
    # -------------------------------------------------------------------------
    result = df.groupby([batch_category_col, volume_category_col], group_keys=False).apply(final_group_calc)

    # 给列命名（最后一列是干净数据）
    result.columns = [
        '样本数(n)', 'I图中心值(CL)', 'MR图中心值(CL)',
        'I图上限(UCL)', 'I图下限(LCL)', 'MR图上限(UCL)', 'MR图下限(LCL)',
        'clean_data'  # 新增列
    ]

    # 拆分：1.控制图参数  2.干净数据集
    params_df = result.drop('clean_data', axis=1).reset_index()
    clean_df = pd.concat(result['clean_data'].tolist(), ignore_index=True)

    # ===================== 返回两个结果 =====================
    return params_df, clean_df


def match_batch_category(df_target: pd.DataFrame, 
                            df_rules: pd.DataFrame,
                            value_col: str = '实际',
                            lower_col: str = '区间下限',
                            upper_col: str = '区间上限',
                            category_col: str = '批量分类',
                            range_col: str = '区间范围') -> pd.DataFrame:
    """
    根据区间规则匹配批量分类和区间范围
    
    参数:
        df_target: 待匹配的目标DataFrame，需包含指定的数值列（如'实际'）
        df_rules: 区间规则DataFrame，包含区间上下限、批量分类、区间范围等字段
        value_col: 目标DataFrame中用于匹配的数值列名，默认'实际'
        lower_col: 规则DataFrame中的区间下限列名，默认'区间下限'
        upper_col: 规则DataFrame中的区间上限列名，默认'区间上限'
        category_col: 规则DataFrame中的批量分类列名，默认'批量分类'
        range_col: 规则DataFrame中的区间范围列名，默认'区间范围'
    
    返回:
        新增了'批量分类'和'区间范围'列的目标DataFrame
    """
    # 复制数据避免修改原数据
    df_target_copy = df_target.copy()
    df_rules_copy = df_rules.copy()
    
    # 处理无穷大值，将字符串'Infinity'转换为np.inf
    df_rules_copy[upper_col] = df_rules_copy[upper_col].replace('Infinity', np.inf)
    
    # 定义匹配函数
    def get_category_and_range(value):
        # 遍历规则找到匹配的区间
        for _, row in df_rules_copy.iterrows():
            lower = row[lower_col]
            upper = row[upper_col]
            
            # 处理闭开区间 [下限, 上限)
            if pd.notna(value) and lower <= value < upper:
                return row[category_col], row[range_col]
        
        # 未匹配到返回空值
        return np.nan, np.nan
    
    # 应用匹配函数，新增两列
    result = df_target_copy[value_col].apply(get_category_and_range)
    df_target_copy['批量分类'] = [x[0] for x in result]
    df_target_copy['区间范围'] = [x[1] for x in result]
    
    return df_target_copy

def group_normality_test(
    df: pd.DataFrame,
    group_cols: list = ['生产线', '批量分类'],
    value_col: str = '损耗率',
    min_sample_size: int = 8,  # 最小样本量（低于则跳过检验）
    figsize: tuple = (10, 8)  # 单个分组图表尺寸
) -> pd.DataFrame:
    """
    按指定分组列做正态性检验
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
