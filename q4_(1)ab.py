import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simple_predict(experiment_file_path, start_displacement=None):
    """
    修复版预测函数 - 正确处理特征缺失值
    """
    # 加载模型和参数
    model = load_model('phase3.h5', compile=False)
    scaler_X = joblib.load('scaler3.pkl')
    params = joblib.load('preprocess_params3.pkl')

    # 读取数据
    df = pd.read_excel(experiment_file_path)
    df = df.sort_values('时间').reset_index(drop=True)

    print("=" * 60)
    print("数据检查")
    print("=" * 60)
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())

    # 获取参数
    feature_cols_raw = params['feature_cols_raw']
    lag_steps = params['lag_steps']
    window_size_feature = params['window_size_feature']

    print("\n" + "=" * 60)
    print("数据预处理参数")
    print("=" * 60)
    print(f"原始特征: {feature_cols_raw}")
    print(f"滞后步数: {lag_steps}")
    print(f"特征平滑窗口: {window_size_feature}")

    # 1. 平滑所有特征
    print("\n1. 应用移动平均平滑...")
    for col in feature_cols_raw:
        if col in df.columns:
            df[f'{col}_smooth'] = df[col].rolling(window=window_size_feature, min_periods=1).mean()
            print(f"   ✓ {col} → {col}_smooth")
        else:
            print(f"   ❌ 列 '{col}' 不存在于数据中！")
            return None, None, None

    # 2. 处理爆破点距离异常值
    if '爆破点距离_m_smooth' in df.columns:
        df['爆破点距离_m_smooth'] = df['爆破点距离_m_smooth'].replace(1e9, np.nan)
        df['爆破点距离_m_smooth'] = df['爆破点距离_m_smooth'].fillna(method='ffill').fillna(
            df['爆破点距离_m_smooth'].median())

    # 3. 创建滞后特征
    print("\n2. 创建滞后特征...")
    smoothed_cols = [f'{col}_smooth' for col in feature_cols_raw]

    for base_col in smoothed_cols:
        for lag in range(1, lag_steps + 1):
            df[f'{base_col}_lag{lag}'] = df[base_col].shift(lag)
        print(f"   ✓ {base_col} → 创建了 {lag_steps} 个滞后特征")

    # 4. 构建完整的特征列名
    feature_cols = []
    for base_col in smoothed_cols:
        feature_cols.append(base_col)
        for lag in range(1, lag_steps + 1):
            feature_cols.append(f'{base_col}_lag{lag}')

    print(f"\n3. 总特征数量: {len(feature_cols)}")

    # 5. 关键修改：只删除特征列中的NaN，不要求目标列存在
    print("\n4. 处理缺失值...")
    print(f"   原始数据行数: {len(df)}")

    # 只检查特征列，允许目标列为空
    df_features = df[feature_cols].copy()

    # 统计每列的缺失值
    missing_counts = df_features.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        print(f"   发现 {len(cols_with_missing)} 个特征列有缺失值:")
        for col, count in cols_with_missing.head(10).items():
            print(f"      {col}: {count} 个缺失值")

    # 删除包含NaN的行（只在特征中）
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"\n   删除缺失值后剩余行数: {len(df_clean)}")

    if len(df_clean) == 0:
        print("   ❌ 错误：所有特征行都包含缺失值！")
        print("\n   可能的原因：数据行数不足以创建滞后特征")
        print(f"   建议：需要至少 {lag_steps + window_size_feature + 1} 行数据才能创建有效的特征")
        return None, None, None

    print(f"   ✓ 保留 {len(df_clean)} 行有效数据")

    # 6. 提取特征
    X = df_clean[feature_cols].values
    print(f"\n5. 特征数组形状: {X.shape}")

    # 7. 归一化
    print("\n6. 应用标准化...")
    expected_features = scaler_X.mean_.shape[0]
    print(f"   期望特征数: {expected_features}")
    print(f"   实际特征数: {X.shape[1]}")

    if X.shape[1] != expected_features:
        print(f"   ❌ 特征数量不匹配！")
        return None, None, None

    X_scaled = scaler_X.transform(X)
    print("   ✓ 标准化完成")

    # 8. 创建序列
    seq_length = params['seq_length']
    print(f"\n7. 创建序列 (seq_length={seq_length})...")

    if len(X_scaled) < seq_length:
        print(f"   ❌ 数据不足！需要至少 {seq_length} 个点，当前 {len(X_scaled)} 个点")
        return None, None, None

    X_seq = []
    for i in range(len(X_scaled) - seq_length + 1):
        X_seq.append(X_scaled[i:i + seq_length])
    X_seq = np.array(X_seq)

    print(f"   ✓ 创建了 {len(X_seq)} 个序列样本")

    # 9. 预测
    print("\n8. 执行预测...")
    y_diff_pred = model.predict(X_seq, verbose=0).flatten()
    print(f"   ✓ 预测完成，预测点数: {len(y_diff_pred)}")

    # 10. 反中心化
    if 'y_mean' in params:
        y_diff_pred = y_diff_pred + params['y_mean']
        print(f"   ✓ 反中心化 (均值={params['y_mean']:.6f})")

    # 11. 获取初始位移
    print("\n9. 计算最终位移...")

    # 如果用户提供了初始位移，使用它
    if start_displacement is not None and not np.isnan(start_displacement):
        print(f"   ✓ 使用指定初始位移: {start_displacement:.4f} mm")
    else:
        # 尝试从数据中获取第一个非NaN的位移值
        if '表面位移_mm' in df.columns:
            # 找到第一个非NaN的位移值
            first_valid_idx = df['表面位移_mm'].first_valid_index()
            if first_valid_idx is not None:
                start_displacement = df.loc[first_valid_idx, '表面位移_mm']
                print(f"   ✓ 从数据中获取初始位移: {start_displacement:.4f} mm (第{first_valid_idx}行)")
            else:
                start_displacement = 0
                print(f"   ⚠ 数据中没有有效位移值，使用默认初始位移: {start_displacement} mm")
        else:
            start_displacement = 0
            print(f"   ⚠ 数据中没有'表面位移_mm'列，使用默认初始位移: {start_displacement} mm")

    # 12. 计算位移序列
    displacements = [start_displacement]
    for diff in y_diff_pred:
        displacements.append(displacements[-1] + diff)

    # 13. 生成结果索引
    # 由于我们删除了前面的缺失行，需要计算实际对应的时间点
    first_valid_time_idx = df_clean.index[0] if len(df_clean) > 0 else 0
    result_indices = range(first_valid_time_idx + seq_length,
                           first_valid_time_idx + seq_length + len(y_diff_pred))

    print(f"\n✅ 预测完成！共预测 {len(y_diff_pred)} 个时间点")
    print(f"   位移范围: {min(displacements[1:]):.6f} - {max(displacements[1:]):.6f} mm")
    print(f"   最终位移: {displacements[-1]:.6f} mm")

    return list(result_indices), y_diff_pred, displacements[1:]


# ==================== 执行预测 ====================
if __name__ == "__main__":
    import os

    data_file = '4phase3lab.xlsx'

    if not os.path.exists(data_file):
        print(f"❌ 错误：找不到数据文件 '{data_file}'")
        print("请确保文件存在于当前目录")
    else:
        # 读取并显示数据基本信息
        df_check = pd.read_excel(data_file)
        print("\n" + "=" * 60)
        print("数据文件信息")
        print("=" * 60)
        print(f"文件名: {data_file}")
        print(f"总行数: {len(df_check)}")
        print(f"总列数: {len(df_check.columns)}")
        print(f"列名: {list(df_check.columns)}")

        # 检查位移数据
        if '表面位移_mm' in df_check.columns:
            valid_disps = df_check['表面位移_mm'].dropna()
            if len(valid_disps) > 0:
                first_disp = valid_disps.iloc[0]
                print(f"第一个有效位移值: {first_disp} mm (第{valid_disps.index[0]}行)")
                print(f"有效位移数量: {len(valid_disps)}")
            else:
                first_disp = None
                print("⚠️ 警告：'表面位移_mm' 列全部为空值")
                print("   将使用默认初始位移 0 mm")
        else:
            first_disp = None
            print("⚠️ 警告：数据中没有 '表面位移_mm' 列")

        print("\n" + "=" * 60)

        # 使用修复后的函数进行预测
        # 如果表面位移全是NaN，从0开始预测
        if first_disp is None or np.isnan(first_disp):
            start_disp = 0.0
            print("提示：使用初始位移 0.0 mm 开始预测")
        else:
            start_disp = first_disp

        result_idx, diff_pred, disp_pred = simple_predict(
            data_file,
            start_displacement=start_disp
        )

        if result_idx is not None and len(result_idx) > 0:
            # 保存结果
            results_df = pd.DataFrame({
                '原始时间点索引': result_idx,
                '预测_位移变化量_mm': diff_pred,
                '预测_表面位移_mm': disp_pred
            })

            # 添加累计变化
            results_df['累计位移变化_mm'] = results_df['预测_位移变化量_mm'].cumsum()

            # 保存到文件
            output_file = '预测结果_phase3lab.xlsx'
            results_df.to_excel(output_file, index=False)

            print("\n" + "=" * 60)
            print("预测结果预览（前20行）:")
            print("=" * 60)
            print(results_df.head(20).to_string())

            print("\n" + "=" * 60)
            print("统计摘要:")
            print("=" * 60)
            print(f"总预测点数: {len(results_df)}")
            print(f"位移变化量 - 均值: {diff_pred.mean():.6f} mm")
            print(f"位移变化量 - 标准差: {diff_pred.std():.6f} mm")
            print(f"位移变化量 - 最小值: {diff_pred.min():.6f} mm")
            print(f"位移变化量 - 最大值: {diff_pred.max():.6f} mm")
            print(f"初始位移: {start_disp:.4f} mm")
            print(f"最终预测位移: {disp_pred[-1]:.6f} mm")
            print(f"总位移变化: {disp_pred[-1] - start_disp:.6f} mm")

            print(f"\n✅ 结果已保存到 '{output_file}'")

            # 可选：绘制预测结果
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(result_idx, disp_pred, 'b-', linewidth=2, label='预测位移')
                plt.xlabel('时间点索引')
                plt.ylabel('表面位移 (mm)')
                plt.title('LSTM预测结果 - 表面位移')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig('预测结果_可视化3.png', dpi=300)
                plt.show()
                print("✅ 可视化图已保存到 '预测结果_可视化.png'")
            except Exception as e:
                print(f"⚠️ 绘图失败: {e}")

        else:
            print("\n❌ 预测失败！")
            print("\n可能的原因：")
            print("  1. 数据点不足（需要至少 {} 个时间点）".format(
                joblib.load('preprocess_params3.pkl')['seq_length'] +
                joblib.load('preprocess_params3.pkl')['lag_steps']))
            print("  2. 特征列名不匹配")
            print("  3. 数据中存在过多的缺失值")