from BlindMethodComparator import *
from BlindEvaluator import *
from BlindInterpolationEvaluator import *
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


# 创建示例数据（模拟实际情况：只有含噪带缺失值的数据）
df1 = pd.read_excel('附件3：监测数据（训练集与实验集）-问题3.xlsx', sheet_name="训练集")
name = (df1.iloc[:,1:2].columns[0])
observed_data = df1.iloc[:, 1]  # 使用单索引，返回一维 Series

# 1. 定义不同的处理方法和插值方法
def method_median_filter(data):
    """中值滤波"""
    from scipy.ndimage import median_filter
    # 先插值填充NaN
    data_filled = pd.Series(data).interpolate().ffill().bfill().values
    return median_filter(data_filled, size=5)


def method_moving_average(data):
    """移动平均"""
    data_filled = pd.Series(data).interpolate().ffill().bfill()
    return data_filled.rolling(5, center=True).mean().fillna(method='bfill').fillna(method='ffill').values


def method_exponential_smoothing(data):
    """指数平滑"""
    data_filled = pd.Series(data).interpolate().ffill().bfill()
    return data_filled.ewm(alpha=0.5, adjust=False).mean().values


def method_savitzky_golay(data):
    """Savitzky-Golay滤波器"""
    from scipy.signal import savgol_filter
    data_filled = pd.Series(data).interpolate().ffill().bfill().values
    return savgol_filter(data_filled, window_length=11, polyorder=3)


# 2. 比较去噪方法
print("\n" + "=" * 60)
print("1. 不同去噪方法比较")
print("=" * 60)

comparator = BlindMethodComparator(observed_data)

methods = {
    '中值滤波': method_median_filter,
    '移动平均': method_moving_average,
    '指数平滑': method_exponential_smoothing,
    'SG滤波': method_savitzky_golay
}

comparison_results = comparator.compare_methods(methods)
print(comparison_results.to_string(index=False))

# 3. 自动选择最佳方法
best_method, best_score, all_results = comparator.auto_select_best_method(methods)
print(f"\n最佳去噪方法: {best_method}")
print(f"综合得分: {best_score:.4f}")

# 4. 比较插值方法
print("\n" + "=" * 60)
print("2. 不同插值方法比较")
print("=" * 60)

interp_evaluator = BlindInterpolationEvaluator(observed_data)


def linear_interpolation(data):
    return pd.Series(data).interpolate(method='linear').values


def quadratic_interpolation(data):
    return pd.Series(data).interpolate(method='quadratic').values


def cubic_interpolation(data):
    return pd.Series(data).interpolate(method='cubic').values


def spline_interpolation(data):
    return pd.Series(data).interpolate(method='spline', order=3).values


interp_methods = {
    '线性插值': linear_interpolation,
    '二次插值': quadratic_interpolation,
    '三次插值': cubic_interpolation,
    '样条插值': spline_interpolation
}

interp_comparison = interp_evaluator.compare_interpolation_methods(interp_methods)
print(interp_comparison.to_string(index=False))

# 5. 综合处理流程
print("\n" + "=" * 60)
print("3. 综合处理流程")
print("=" * 60)

# 先插值填充缺失值
filled_data = linear_interpolation(observed_data)

# 再去噪
best_denoise_method = methods[best_method]
cleaned_data = best_denoise_method(filled_data)
df = pd.DataFrame(cleaned_data, columns=['cleaned降水量'])

# 保存到Excel
df.to_excel('降水量.xlsx', index=False)

print(f"原始数据有效值: {observed_data[~np.isnan(observed_data)].shape[0]}")
print(f"插值后数据量: {len(filled_data)}")
print(f"去噪后数据统计:")
print(f"  - 均值: {np.mean(cleaned_data):.4f}")
print(f"  - 标准差: {np.std(cleaned_data):.4f}")
print(f"  - 最小值: {np.min(cleaned_data):.4f}")
print(f"  - 最大值: {np.max(cleaned_data):.4f}")

# 6. 生成评估报告
print("\n" + "=" * 60)
print("4. 盲评估报告")
print("=" * 60)

evaluator = BlindEvaluator()
total_score, sub_scores = evaluator.comprehensive_blind_score(filled_data, cleaned_data)

print("\n去噪效果评估:")
for metric, score in sub_scores.items():
    print(f"  {metric}: {score:.4f}")
print(f"\n综合质量得分: {total_score:.4f}")

# 评分等级
if total_score >= 0.8:
    grade = "优秀"
elif total_score >= 0.6:
    grade = "良好"
elif total_score >= 0.4:
    grade = "一般"
else:
    grade = "较差"

print(f"质量等级: {grade}")

# 7. 可视化评估（需要VisualizationEvaluator类）
try:
    from VisualizationEvaluator import VisualizationEvaluator

    viz = VisualizationEvaluator()

    # 使用修复后的可视化
    fig = viz.plot_with_nan_handling(observed_data, cleaned_data,
                                     f"{name}数据处理效果对比 (评分: {total_score:.3f})")
    import matplotlib.pyplot as plt

    plt.show()
except:
    print("\n可视化模块未找到，跳过绘图")