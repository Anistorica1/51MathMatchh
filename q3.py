import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


def comprehensive_denoise_interpolate(df, columns=None,
                                      denoise_method='median',
                                      interp_method='linear',
                                      window=5,
                                      n_std=3):
    """
    综合去噪和插值函数

    参数:
    - df: DataFrame
    - columns: 要处理的列名列表
    - denoise_method: 去噪方法 ('median', 'moving_average', 'exponential', 'statistical', 'iqr')
    - interp_method: 插值方法 ('linear', 'polynomial', 'spline', 'time')
    - window: 窗口大小
    - n_std: 标准差倍数
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    result_df = df.copy()

    for col in columns:
        # 1. 去噪
        if denoise_method == 'median':
            result_df[col] = result_df[col].rolling(window=window, center=True).median()
        elif denoise_method == 'moving_average':
            result_df[col] = result_df[col].rolling(window=window, center=True).mean()
        elif denoise_method == 'exponential':
            result_df[col] = result_df[col].ewm(span=window, adjust=False).mean()
        elif denoise_method == 'statistical':
            mean = result_df[col].mean()
            std = result_df[col].std()
            outliers = np.abs(result_df[col] - mean) > n_std * std
            result_df.loc[outliers, col] = np.nan
        elif denoise_method == 'iqr':
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
            result_df.loc[outliers, col] = np.nan

        # 2. 插值
        if interp_method == 'linear':
            result_df[col] = result_df[col].interpolate(method='linear', limit_direction='both')
        elif interp_method == 'polynomial':
            result_df[col] = result_df[col].interpolate(method='polynomial', order=2)
        elif interp_method == 'spline':
            result_df[col] = result_df[col].interpolate(method='spline', order=3)
        elif interp_method == 'time':
            result_df[col] = result_df[col].interpolate(method='time')

    return result_df


# 使用示例


if __name__ == '__main__':
    df = pd.read_excel('附件3：监测数据（训练集与实验集）-问题3.xlsx',sheet_name="训练集")
    df_cleaned = comprehensive_denoise_interpolate(df,
                                                   columns=['a:降雨量_mm', 'b:孔隙水压力_kPa','c:微震事件数',
                                                            'd:深部位移_mm','e:表面位移_mm'],
                                                   denoise_method='median',
                                                   interp_method='linear')

    df_cleaned.to_excel("cleaned.xlsx")