import pandas as pd
import numpy as np


def detect_common_outliers(df, var_columns=None, method='iqr', threshold=2, return_details=True):
    """
    多变量联合异常检测：识别同一时间点至少N个变量同时异常的行

    参数
    ----------
    df : pd.DataFrame
        包含各监测变量的数据框（每列一个变量，每行一个时间点）
    var_columns : list, optional
        需要检测的变量列名列表，默认使用所有数值列
    method : str, 'iqr' 或 '3sigma'
        单变量异常检测方法
    threshold : int
        共同异常阈值，默认2（≥2个变量异常即标记为共同异常点）
    return_details : bool
        是否返回详细标记列，默认True

    返回
    -------
    result_df : pd.DataFrame
        包含以下列：
        - 原始数据的所有列（包括时间列）
        - 各变量的异常标记列（_outlier后缀）
        - outlier_count: 每时间点异常变量个数
        - is_common_outlier: 是否为共同异常点（≥threshold个变量异常）
    """

    # 自动选择需要检测的列（只选数值列，排除时间列和已存在的_outlier列）
    if var_columns is None:
        var_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # 移除可能已经存在的标记列
        var_columns = [col for col in var_columns if not col.endswith('_outlier')]
    else:
        # 确保传入的列存在
        var_columns = [col for col in var_columns if col in df.columns]

    if len(var_columns) == 0:
        raise ValueError("没有找到可用于检测的数值列")

    # 复制数据框，避免修改原数据（保留所有原始列，包括时间列）
    result_df = df.copy()

    # 对每个变量进行异常检测
    for col in var_columns:
        data = df[col].dropna()  # 计算阈值时忽略缺失值

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 标记异常（NaN保留不参与判断）
            outlier_series = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == '3sigma':
            mean = data.mean()
            std = data.std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            outlier_series = (df[col] < lower_bound) | (df[col] > upper_bound)

        else:
            raise ValueError("method 必须是 'iqr' 或 '3sigma'")

        # 存储异常标记列
        result_df[f'{col}_outlier'] = outlier_series

    # 统计每行有多少个变量异常
    outlier_cols = [f'{col}_outlier' for col in var_columns]
    result_df['outlier_count'] = result_df[outlier_cols].sum(axis=1)

    # 标记共同异常点
    result_df['is_common_outlier'] = result_df['outlier_count'] >= threshold

    if not return_details:
        # 只返回原始数据 + 共同异常标记
        result_df = result_df[df.columns.tolist() + ['is_common_outlier']]

    return result_df


# ============= 你的实际使用代码 =============
# 读取你的Excel数据（假设有5个监测变量）
df = pd.read_excel('监测数据cleaned.xlsx')  # 替换为你的实际文件路径

# 执行检测（自动识别数值列）
result = detect_common_outliers(
    df,
    var_columns=['降雨量_mm', '孔隙水压力_kPa', '深部位移_mm', '微震事件数', '表面位移_mm'],  # 你的5个变量名
    method='iqr',  # 推荐用IQR，如果数据正态分布可改用'3sigma'
    threshold=2
)

# 查看共同异常点
common_points = result[result['is_common_outlier'] == True]

print("=" * 60)
print(f"共同异常点（同一时间点≥{2}个变量异常）：共找到 {len(common_points)} 个")
print("=" * 60)

if len(common_points) > 0:
    # 显示关键列（保留你的时间列名）
    # 注意：时间列名可能叫'时间'、'日期'、'datetime'等，请根据实际情况修改
    time_column = '时间'  # 修改为你的实际时间列名

    # 如果'时间'列不存在，尝试找可能的列名
    if time_column not in common_points.columns:
        # 常见的列名尝试
        possible_time_cols = ['时间', '日期', 'date', 'time', 'datetime', '监测时间']
        for col in possible_time_cols:
            if col in common_points.columns:
                time_column = col
                break
        else:
            # 如果都没有，使用索引
            print("警告：未找到时间列，将使用行索引")
            display_cols = ['降雨量_mm', '孔隙水压力_kPa', '深部位移_mm', '微震事件数', '表面位移_mm',
                            'outlier_count', 'is_common_outlier']
            print(common_points[display_cols])
            print("\n")
            # 导出到Excel
            common_points.to_excel('共同异常点结果.xlsx', index=False)
            print("结果已保存到: 共同异常点结果.xlsx")
            exit()

    # 显示结果（包括时间、原始变量值、异常计数）
    display_cols = [time_column, '降雨量_mm', '孔隙水压力_kPa', '深部位移_mm', '微震事件数', '表面位移_mm',
                    'outlier_count'] + [f'{col}_outlier' for col in
                                        ['降雨量_mm', '孔隙水压力_kPa', '深部位移_mm', '微震事件数', '表面位移_mm']]

    print(common_points[display_cols].to_string())
    print("\n")

    # 导出到Excel
    common_points.to_excel('共同异常点结果.xlsx', index=False)
    print("结果已保存到: 共同异常点结果.xlsx")

    # 可选：显示统计信息
    print("\n" + "=" * 60)
    print("异常变量分布统计：")
    outlier_counts = common_points[['降雨量_mm_outlier', '孔隙水压力_kPa_outlier',
                                    '深部位移_mm_outlier', '微震事件数_outlier', '表面位移_mm_outlier']].sum()
    for var, count in outlier_counts.items():
        var_name = var.replace('_outlier', '')
        print(f"  {var_name}: {int(count)} 次异常")
else:
    print("未发现共同异常点")