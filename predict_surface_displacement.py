"""
使用已训练的模型预测边坡表面位移
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体（解决中文显示问题）
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ========================== 1. 加载模型 ==========================
print("正在加载模型...")
try:
    rf_model = joblib.load('slope_displacement_model.pkl')
    rainfall_transformer = joblib.load('rainfall_transformer.pkl')
    print("✓ 模型加载成功！")
except FileNotFoundError:
    print("❌ 模型文件不存在，请先训练模型")
    exit(1)

# ========================== 2. 单次预测 ==========================
def predict_single(rainfall, pore_pressure, deep_displacement, microseismic):
    """预测单个工况下的表面位移"""
    input_data = pd.DataFrame({
        '降雨量_mm': [rainfall],
        '孔隙水压力_kPa': [pore_pressure],
        '深部位移_mm': [deep_displacement],
        '微震事件数': [microseismic]
    })

    input_data['降雨量_transformed'] = rainfall_transformer.transform(input_data[['降雨量_mm']])
    features = ['降雨量_transformed', '孔隙水压力_kPa', '深部位移_mm', '微震事件数']
    X = input_data[features]

    # 转换为数组预测，避免警告
    X_array = X.to_numpy()
    prediction = rf_model.predict(X_array)[0]
    return prediction

# ========================== 3. 批量预测 ==========================
def predict_from_file(input_file_path, output_file_path=None, plot_charts=True):
    """
    从Excel或CSV文件读取数据并批量预测

    参数:
        input_file_path: 输入文件路径
        output_file_path: 输出结果保存路径（可选）
        plot_charts: 是否绘制图表（默认True）
    """
    # 读取数据
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"文件不存在: {input_file_path}")

    if input_file_path.endswith('.csv'):
        df = pd.read_csv(input_file_path)
    else:
        df = pd.read_excel(input_file_path)

    print(f"✓ 读取到 {len(df)} 条待预测数据")

    # 检查必需列
    required_cols = ['降雨量_mm', '孔隙水压力_kPa', '深部位移_mm', '微震事件数']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    # 变换和预测
    df['降雨量_transformed'] = rainfall_transformer.transform(df[['降雨量_mm']])
    features = ['降雨量_transformed', '孔隙水压力_kPa', '深部位移_mm', '微震事件数']
    X = df[features]

    # 转换为 numpy 数组预测，避免警告
    X_array = X.to_numpy()
    predictions = rf_model.predict(X_array)

    # 添加预测结果
    df['预测表面位移_mm'] = predictions

    # 添加置信区间
    tree_preds = np.array([tree.predict(X_array) for tree in rf_model.estimators_])
    df['预测标准差_mm'] = np.std(tree_preds, axis=0)
    df['预测下限_mm'] = df['预测表面位移_mm'] - 1.96 * df['预测标准差_mm']
    df['预测上限_mm'] = df['预测表面位移_mm'] + 1.96 * df['预测标准差_mm']

    # 保存结果
    if output_file_path:
        if output_file_path.endswith('.csv'):
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        else:
            df.to_excel(output_file_path, index=False)
        print(f"✓ 预测结果已保存到: {output_file_path}")

    # 绘制图表
    if plot_charts:
        plot_prediction_charts(df)

    return df

# ========================== 4. 绘制预测结果图表 ==========================
def plot_prediction_charts(df):
    """
    绘制预测结果的两个图表：
    1. 预测结果散点分布图
    2. 带置信区间的预测图
    """
    # 创建图形，两个子图上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('边坡表面位移预测结果', fontsize=16, fontweight='bold')

    # ===== 图1：预测结果散点分布图 =====
    sample_indices = np.arange(len(df))
    ax1.scatter(sample_indices, df['预测表面位移_mm'],
                alpha=0.6, s=15, c='steelblue', edgecolors='white', linewidth=0.5)
    ax1.set_xlabel('样本序号', fontsize=12)
    ax1.set_ylabel('预测表面位移 (mm)', fontsize=12)
    ax1.set_title('预测结果散点分布图', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 添加平均值线
    mean_pred = df['预测表面位移_mm'].mean()
    ax1.axhline(y=mean_pred, color='red', linestyle='--', linewidth=1.5,
                label=f'平均值: {mean_pred:.2f} mm')
    ax1.legend(loc='upper right', fontsize=10)

    # 添加统计信息文本框
    stats_text = f'样本数: {len(df)}\n平均值: {mean_pred:.2f} mm\n标准差: {df["预测表面位移_mm"].std():.2f} mm\n最大值: {df["预测表面位移_mm"].max():.2f} mm\n最小值: {df["预测表面位移_mm"].min():.2f} mm'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ===== 图2：带置信区间的预测图 =====
    x = np.arange(len(df))

    # 绘制散点
    ax2.scatter(x, df['预测表面位移_mm'], alpha=0.5, s=15, c='steelblue', label='预测值')

    # 绘制置信区间阴影（95%置信区间）
    ax2.fill_between(x, df['预测下限_mm'], df['预测上限_mm'],
                     alpha=0.2, color='gray', label='95% 置信区间')

    # 绘制置信区间边界线（可选）
    ax2.plot(x, df['预测下限_mm'], color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    ax2.plot(x, df['预测上限_mm'], color='gray', linewidth=0.5, alpha=0.5, linestyle='--')

    # 添加趋势线（移动平均）
    window = min(50, max(5, len(df) // 20))  # 自适应窗口大小
    if window > 1:
        moving_avg = df['预测表面位移_mm'].rolling(window=window, center=True).mean()
        ax2.plot(x, moving_avg, color='red', linewidth=2, label=f'{window}点移动平均')

    ax2.set_xlabel('样本序号', fontsize=12)
    ax2.set_ylabel('预测表面位移 (mm)', fontsize=12)
    ax2.set_title('带置信区间的预测结果', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('预测结果图.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ 图表已保存为: 预测结果图.png")

# ========================== 5. 主程序 ==========================
if __name__ == "__main__":
    # 单次预测示例
    print("\n" + "="*50)
    print("单次预测示例")
    print("="*50)

    result = predict_single(rainfall=25.0, pore_pressure=45.0,
                           deep_displacement=32.5, microseismic=8)
    print(f"预测表面位移: {result:.2f} mm")

    # 批量预测示例
    print("\n" + "="*50)
    print("批量预测示例")
    print("="*50)

    try:
        # 进行批量预测并自动绘制图表
        results = predict_from_file('监测数据cleaned实验.xlsx', '预测结果.xlsx', plot_charts=True)

        # 显示前几条预测结果
        print("\n预测结果预览:")
        print(results[['降雨量_mm', '深部位移_mm', '预测表面位移_mm',
                      '预测下限_mm', '预测上限_mm']].head())

        # 打印统计信息
        print("\n" + "="*50)
        print("预测结果统计")
        print("="*50)
        print(f"样本数量: {len(results)}")
        print(f"预测位移平均值: {results['预测表面位移_mm'].mean():.2f} mm")
        print(f"预测位移标准差: {results['预测表面位移_mm'].std():.2f} mm")
        print(f"预测位移最大值: {results['预测表面位移_mm'].max():.2f} mm")
        print(f"预测位移最小值: {results['预测表面位移_mm'].min():.2f} mm")
        print(f"预测位移中位数: {results['预测表面位移_mm'].median():.2f} mm")

    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("请确保 '监测数据cleaned实验.xlsx' 文件存在")
    except Exception as e:
        print(f"❌ 预测失败: {e}")

    print("\n" + "="*50)
    print("✓ 模型使用和绘图完成")
    print("="*50)