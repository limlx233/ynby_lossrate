import os
import io
from datetime import datetime
# 第三方库
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 自定义库
from DP import dp4

# 忽略无关警告
import warnings
warnings.filterwarnings('ignore')

# ===================== 工具函数 =====================
def setup_custom_font():
    """配置matplotlib自定义字体"""
    font_filename = "MSYH.TTC"
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current_script_dir, "font")
    font_path = os.path.join(font_dir, font_filename)
    
    if not os.path.exists(font_path):
        st.warning(f"字体文件不存在：{font_path}")
        return False

    try:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        st.warning(f"加载字体失败: {str(e)}")
        return False

# ===================== 核心修复：会话状态初始化函数 =====================
def init_session_state():
    """安全初始化会话状态，避免提前访问"""
    # 仅在首次加载/未初始化时执行
    if "session_initialized" not in st.session_state:
        session_vars = [
            "historical_processed_fhg_p5", "batch_nodes_fhg_p5", "IMR_params_fhg_p5",
            "historical_processed_zh_p5", "batch_nodes_zh_p5", "IMR_params_zh_p5",
            "historical_processed_zx_p5", "batch_nodes_zx_p5", "IMR_params_zx_p5",
            "current_fhg_p5", "current_zh_p5", "current_zx_p5",
            "fhg_outlier_p5", "zh_outlier_p5", "zx_outlier_p5",
            "fhg_p5", "zh_p5", "zx_p5",
            "fhg_low_loss_rate", "zh_low_loss_rate", "zx_low_loss_rate"
        ]
        for var in session_vars:
            if var not in st.session_state:
                st.session_state[var] = None
        # 标记初始化完成
        st.session_state.session_initialized = True

# ======== 主页面逻辑（核心：先初始化，再渲染） ========

# 1. 先初始化会话状态（修复核心）
init_session_state()

# 2. 配置字体（移到初始化后）
setup_custom_font()

# 全局配置
plt.rcParams['axes.unicode_minus'] = False

# 3. 页面内容渲染
st.header("包材损耗率分析", divider="rainbow")
with st.expander(label='说明'):
    st.markdown('''
                ℹ️ 操作流程：
                1. 包材损耗率所需上传的Excel文件应包含: 复合管、纸盒、纸箱 三个Sheet
                2. 先上传「历史耗用数据」
                3. 再上传「月度耗用数据
                4. 点击下载按钮，下载结果文件
                ''') 

with st.container(border=True):
    # 划分两列，宽度比例为 1:2
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col1:
        st.markdown('##### 1. 选择组织:')
    with col2:
        Org = st.selectbox(label=" ",options=["口腔-JKC","口腔-JKY"])
        org = 'JKC' if Org == "口腔-JKC" else 'JKY'

    col4, col5, clo6 = st.columns([1, 2.5, 1])
    with col4:
        st.markdown('##### 2. 上传文件:')
    with col5:
        # 创建文件上传组件，限制文件类型为 XLSX
        uploaded_file1 = st.file_uploader(
            "历史耗用数据",
            type=["xlsx"],
            accept_multiple_files=False,
            key="file1"  # 增加唯一key，避免渲染冲突
        )
        uploaded_file2 = st.file_uploader(
            "月度耗用数据",
            type=["xlsx"],
            accept_multiple_files=False,
            key="file2"  # 增加唯一key
        )
    
    # 4. 文件处理逻辑（增加安全校验）
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.success("文件已上传。", icon="✅")
        try:
            # 读取数据（增加异常捕获）
            with st.spinner("正在读取数据..."):
                # 读取历史数据
                df_fhg = pd.read_excel(uploaded_file1, sheet_name='复合管', header=0)
                df_zh = pd.read_excel(uploaded_file1, sheet_name="纸盒", header=0)
                df_zx = pd.read_excel(uploaded_file1, sheet_name="纸箱", header=0)
                # 读取当月数据
                df_cur_fhg = pd.read_excel(uploaded_file2, sheet_name="复合管", header=0)
                df_cur_zh = pd.read_excel(uploaded_file2, sheet_name="纸盒", header=0)
                df_cur_zx = pd.read_excel(uploaded_file2, sheet_name="纸箱", header=0)
            
            # 清洗历史数据
            df_fhg = dp4.filter1(df_fhg)
            df_zh = dp4.filter1(df_zh)
            df_zx = dp4.filter1(df_zx)

            # 统一清洗批号列
            for df in [df_fhg, df_zh, df_zx]:
                if "批号" in df.columns:
                    df["批号"] = df["批号"].fillna("").astype(str)
            
            with st.expander("历史数据正态性检验", expanded=False):
                pl = "区间范围"

                # 业务处理（核心：分步处理，避免一次性写入大量session_state）
                df1, batch_nodes, analysis_res = dp4.batch_kmeans_clustering(df_fhg)
                st.markdown('---')
                st.markdown('#### 复合管数据正态性检验')
                df1 = dp4.group_normality_test(df1)
                imr_params = dp4.calculate_imr_control_chart_params(df1)
                
                imr_params['批量分类'] = imr_params['分组名称'].str.extract(r'-(.+)')
                imr_params = pd.merge(imr_params, batch_nodes[['批量分类','区间范围']], on='批量分类')
                col_data1 = imr_params.pop(pl)
                imr_params.insert(loc=1, column=pl, value=col_data1)

                df2, batch_nodes2, analysis_res2 = dp4.batch_kmeans_clustering(df_zh)
                st.markdown('---')
                st.markdown('#### 纸盒数据正态性检验')
                df2 = dp4.group_normality_test(df2)
                imr_params2 = dp4.calculate_imr_control_chart_params(df2)

                imr_params2['批量分类'] = imr_params2['分组名称'].str.extract(r'-(.+)')
                imr_params2 = pd.merge(imr_params2, batch_nodes2[['批量分类','区间范围']], on='批量分类')
                col_data2 = imr_params2.pop(pl)
                imr_params2.insert(loc=1, column=pl, value=col_data2)

                df3, batch_nodes3, analysis_res3 = dp4.batch_kmeans_clustering(df_zx)
                st.markdown('---')
                st.markdown('#### 纸箱数据正态性检验')
                df3 = dp4.group_normality_test(df3)
                imr_params3 = dp4.calculate_imr_control_chart_params(df3)

                imr_params3['批量分类'] = imr_params3['分组名称'].str.extract(r'-(.+)')
                imr_params3 = pd.merge(imr_params3, batch_nodes3[['批量分类','区间范围']], on='批量分类')
                col_data3 = imr_params3.pop(pl)
                imr_params3.insert(loc=1, column=pl, value=col_data3)

            # 分步写入session_state（减少一次性IO压力）
            st.session_state.historical_processed_fhg_p5 = df1
            st.session_state.batch_nodes_fhg_p5 = batch_nodes
            st.session_state.IMR_params_fhg_p5 = imr_params

            st.session_state.historical_processed_zh_p5 = df2
            st.session_state.batch_nodes_zh_p5 = batch_nodes2
            st.session_state.IMR_params_zh_p5 = imr_params2

            st.session_state.historical_processed_zx_p5 = df3
            st.session_state.batch_nodes_zx_p5 = batch_nodes3
            st.session_state.IMR_params_zx_p5 = imr_params3
            
            st.success(f"✅ 历史数据处理完成：{uploaded_file1.name}")

            # 清洗批号
            for df in [df_cur_fhg, df_cur_zh, df_cur_zx]:
                if "批号" in df.columns:
                    df["批号"] = df["批号"].fillna("").astype(str)
            
            # 处理复合管数据
            df_cur_fhg = dp4.generate_batch_sequence(df_cur_fhg, batch_col='批号', unit_col='生产线')
            df_cur_fhg = dp4.match_batch_category_by_quantity(
                monthly_data=df_cur_fhg,
                batch_category_rules=st.session_state.batch_nodes_fhg_p5,
                quantity_col="入库数量",  
                category_col="批量分类"
            )
            
            # 处理纸盒数据
            df_cur_zh = dp4.generate_batch_sequence(df_cur_zh, batch_col='批号', unit_col='生产线')
            df_cur_zh = dp4.match_batch_category_by_quantity(
                monthly_data=df_cur_zh,
                batch_category_rules=st.session_state.batch_nodes_zh_p5,
                quantity_col="入库数量",  
                category_col="批量分类"
            )
            
            # 处理纸箱数据
            df_cur_zx = dp4.generate_batch_sequence(df_cur_zx, batch_col='批号', unit_col='生产线')
            df_cur_zx = dp4.match_batch_category_by_quantity(
                monthly_data=df_cur_zx,
                batch_category_rules=st.session_state.batch_nodes_zx_p5,
                quantity_col="入库数量",  
                category_col="批量分类"
            )

            # 缓存月度数据处理结果
            st.session_state.current_fhg_p5 = df_cur_fhg
            st.session_state.current_zh_p5 = df_cur_zh
            st.session_state.current_zx_p5 = df_cur_zx

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

            with st.expander("控制图结果输出", expanded= False):
                # 生成控制图并获取异常数据
                df_outlier, df_control_summary, fhg_low_loss_rate = dp4.plot_imr_control_charts(
                    df_analysis=df_cur_fhg,
                    df_control_params=st.session_state.IMR_params_fhg_p5,
                    category='复合管'
                )
                st.session_state.fhg_outlier_p5 = df_outlier[df_outlier['异常值'] == True]
                st.session_state.fhg_low_loss_rate = fhg_low_loss_rate
                

                df_outlier2, df_control_summary2,zh_low_loss_rate = dp4.plot_imr_control_charts(
                    df_analysis=df_cur_zh,
                    df_control_params=st.session_state.IMR_params_zh_p5,
                    category='纸盒'
                )
                st.session_state.zh_outlier_p5 = df_outlier2[df_outlier2['异常值'] == True]
                st.session_state.zh_low_loss_rate = zh_low_loss_rate

                df_outlier3, df_control_summary3,zx_low_loss_rate = dp4.plot_imr_control_charts(
                    df_analysis=df_cur_zx,
                    df_control_params=st.session_state.IMR_params_zx_p5,
                    category='纸箱'
                )
                st.session_state.zx_outlier_p5 = df_outlier3[df_outlier3['异常值'] == True]
                st.session_state.zx_low_loss_rate = zx_low_loss_rate
            
            # 将当月数据汇总到历史数据中
            cols_to_keep1 = ['年月', '任务单', '批号', '生产线', '物料编码', '名称', '入库数量', '耗用数', '损耗率']
            # 复合管数据合并
            df_fhg_res1 = st.session_state.historical_processed_fhg_p5[cols_to_keep1]
            df_fhg_res2 = st.session_state.current_fhg_p5[cols_to_keep1]
            df_fhg = pd.concat([df_fhg_res1, df_fhg_res2], axis=0, ignore_index=True)
            df_fhg['年月'] = df_fhg['年月'].apply(dp4.format_month)
            st.session_state.fhg_p5 = df_fhg
            # 纸盒数据合并
            df_zh_res1 = st.session_state.historical_processed_zh_p5[cols_to_keep1]
            df_zh_res2 = st.session_state.current_zh_p5[cols_to_keep1]
            df_zh = pd.concat([df_zh_res1, df_zh_res2], axis=0, ignore_index=True)
            df_zh['年月'] = df_zh['年月'].apply(dp4.format_month)
            st.session_state.zh_p5 = df_zh
            # 纸箱数据合并
            df_zx_res1 = st.session_state.historical_processed_zx_p5[cols_to_keep1]
            df_zx_res2 = st.session_state.current_zx_p5[cols_to_keep1]
            df_zx = pd.concat([df_zx_res1, df_zx_res2], axis=0, ignore_index=True)
            df_zx['年月'] = df_zx['年月'].apply(dp4.format_month)
            st.session_state.zx_p5 = df_zx
            st.success(f"✅ 月度数据处理完成：{uploaded_file2.name}")

        except Exception as e:
            # 异常时重置session_state，避免脏数据
            st.session_state.session_initialized = False
            st.error(f"数据处理失败：{str(e)}")
        
        # 下载按钮（独立封装，增加空值校验）
        st.markdown('##### 3. 结果下载:')
        def create_excel():
            # 创建一个 BytesIO 对象来存储 Excel 数据
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            
            # 安全写入：增加空值判断
            def safe_to_excel(df, sheet_name):
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 写入异常数据
            safe_to_excel(st.session_state.fhg_outlier_p5, '复合管异常数据')
            safe_to_excel(st.session_state.zh_outlier_p5, '纸盒异常数据')
            safe_to_excel(st.session_state.zx_outlier_p5, '纸箱异常数据')
            
            # 写入负损耗数据
            safe_to_excel(st.session_state.fhg_low_loss_rate, '复合管负损耗数据')
            safe_to_excel(st.session_state.zh_low_loss_rate, '纸盒负损耗数据')
            safe_to_excel(st.session_state.zx_low_loss_rate, '纸箱负损耗数据')

            # 写入合并后的数据
            safe_to_excel(st.session_state.fhg_p5, '复合管')
            safe_to_excel(st.session_state.zh_p5, '纸盒')
            safe_to_excel(st.session_state.zx_p5, '纸箱')
            
            # 写入IMR控制图参数
            safe_to_excel(st.session_state.IMR_params_fhg_p5, 'IMR参数_复合管')
            safe_to_excel(st.session_state.IMR_params_zh_p5, 'IMR参数_纸盒')
            safe_to_excel(st.session_state.IMR_params_zx_p5, 'IMR参数_纸箱')
            
            # 写入批量分类规则
            def process_batch_nodes(batch_nodes, sheet_name):
                if batch_nodes is None:
                    return
                try:
                    if isinstance(batch_nodes, pd.DataFrame):
                        df = batch_nodes
                    elif isinstance(batch_nodes, (list, np.ndarray)):
                        df = pd.DataFrame(batch_nodes, columns=['批量分类规则'])
                    elif isinstance(batch_nodes, dict):
                        df = pd.DataFrame(list(batch_nodes.items()), columns=['键', '值'])
                    else:
                        df = pd.DataFrame({'批量分类规则': [str(batch_nodes)]})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    error_df = pd.DataFrame({'错误信息': [f"处理批量分类规则失败: {str(e)}"]})
                    error_df.to_excel(writer, sheet_name=sheet_name + '_错误', index=False)
            
            process_batch_nodes(st.session_state.batch_nodes_fhg_p5, '批量分类规则_复合管')
            process_batch_nodes(st.session_state.batch_nodes_zh_p5, '批量分类规则_纸盒')
            process_batch_nodes(st.session_state.batch_nodes_zx_p5, '批量分类规则_纸箱')

            # 保存并重置指针
            writer.close()
            output.seek(0)
            return output.getvalue()
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 添加下载按钮（增加disabled状态，避免空数据下载）
        download_disabled = any([
            st.session_state.fhg_p5 is None,
            st.session_state.zh_p5 is None,
            st.session_state.zx_p5 is None
        ])
        
        st.download_button(
            label="下载结果",
            type="primary",
            data=create_excel(),
            file_name=f"{Org}-包材物耗分析_{current_time}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            disabled=download_disabled
        )
        
    elif uploaded_file1 is None and uploaded_file2 is not None:
        st.warning("请上传历史耗用数据！", icon="⚠️")
    elif uploaded_file2 is None and uploaded_file1 is not None:
        st.warning("请上传月度耗用数据！", icon="⚠️")
    else:
        st.info('请上传历史和月度耗用数据！', icon="ℹ️")