import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

class VisualizationEvaluator:
    """可视化评估工具 - 修复NaN错误版本"""

    @staticmethod
    def remove_nan_values(*arrays):
        """移除所有数组中的NaN值（保持对齐）"""
        # 创建有效掩码
        combined = np.column_stack(arrays)
        valid_mask = ~np.isnan(combined).any(axis=1)

        # 返回清理后的数组
        cleaned = [arr[valid_mask] for arr in arrays]
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    @staticmethod
    def plot_comparison(original, processed, title="去噪/插值效果对比"):
        """
        绘制对比图（处理NaN）
        """
        # 转换为numpy数组并移除NaN
        original = np.array(original).flatten()
        processed = np.array(processed).flatten()

        # 只保留两列都非NaN的值
        valid_mask = ~(np.isnan(original) | np.isnan(processed))
        original_clean = original[valid_mask]
        processed_clean = processed[valid_mask]

        if len(original_clean) == 0:
            print("错误：没有有效数据可供绘图")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 原始信号 vs 处理信号
        axes[0, 0].plot(original_clean, label='原始信号', alpha=0.7)
        axes[0, 0].plot(processed_clean, label='处理后信号', alpha=0.7)
        axes[0, 0].set_title('信号对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 误差分布（只使用有限值）
        error = original_clean - processed_clean
        error = error[np.isfinite(error)]  # 再过滤一次inf值

        if len(error) > 0:
            axes[0, 1].hist(error, bins=min(50, len(error) // 10), edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(x=0, color='r', linestyle='--', label='零误差线')
            axes[0, 1].set_title(f'误差分布 (均值={np.nanmean(error):.4f}, 标准差={np.nanstd(error):.4f})')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '无有效误差数据', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('误差分布')

        # 残差图
        axes[1, 0].scatter(original_clean, error, alpha=0.5, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('原始值')
        axes[1, 0].set_ylabel('残差')
        axes[1, 0].set_title('残差图')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q图（需要足够的数据点）
        if len(error) >= 10:
            stats.probplot(error, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q图 (正态性检验)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, f'数据点不足({len(error)}个)，无法绘制Q-Q图',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Q-Q图')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_with_nan_handling(original, processed, title="效果对比"):
        """
        处理NaN的更灵活版本
        """
        # 转换为DataFrame以便更好地处理NaN
        df = pd.DataFrame({
            'original': original,
            'processed': processed
        })

        # 方法1：可以绘制缺失值位置
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 获取索引
        x = np.arange(len(df))

        # 标记缺失值位置
        missing_original = df['original'].isna()
        missing_processed = df['processed'].isna()

        # 绘制有效数据
        ax1.plot(x[~missing_original], df.loc[~missing_original, 'original'],
                 'b-', label='原始信号(有效)', alpha=0.7)
        ax1.plot(x[~missing_processed], df.loc[~missing_processed, 'processed'],
                 'r-', label='处理后信号(有效)', alpha=0.7)

        # 标记缺失值位置
        if missing_original.any():
            ax1.scatter(x[missing_original],
                        [df['original'].min()] * missing_original.sum() if not df['original'].isna().all() else [0],
                        color='blue', s=30, marker='v', label='原始数据缺失', alpha=0.6)
        if missing_processed.any():
            ax1.scatter(x[missing_processed],
                        [df['processed'].min()] * missing_processed.sum() if not df['processed'].isna().all() else [0],
                        color='red', s=30, marker='^', label='处理后数据缺失', alpha=0.6)

        ax1.set_title('信号对比（含缺失值标记）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制误差（只在两列都有效的位置）
        valid = ~(missing_original | missing_processed)
        if valid.any():
            error = df.loc[valid, 'original'] - df.loc[valid, 'processed']
            ax2.plot(x[valid], error, 'g-', label='误差', alpha=0.7)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.fill_between(x[valid], 0, error, alpha=0.3)
            ax2.set_title(f'误差曲线 (RMSE={np.sqrt(np.mean(error ** 2)):.4f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无重叠的有效数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('误差曲线')

        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_method_comparison(data_dict, title="不同方法效果对比"):
        """
        比较多种方法（处理NaN）
        """
        # 清理数据
        clean_dict = {}
        for name, data in data_dict.items():
            data_clean = pd.Series(data).interpolate().ffill().bfill().values
            clean_dict[name] = data_clean

        n_methods = len(clean_dict)
        fig, axes = plt.subplots(n_methods, 2, figsize=(12, 4 * n_methods))

        if n_methods == 1:
            axes = axes.reshape(1, -1)

        for idx, (method_name, data) in enumerate(clean_dict.items()):
            # 时间序列图
            axes[idx, 0].plot(data, label=method_name)
            axes[idx, 0].set_title(f'{method_name} - 时间序列')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # 频谱图
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data))
            axes[idx, 1].plot(freq[:len(freq) // 2], np.abs(fft_data)[:len(freq) // 2])
            axes[idx, 1].set_title(f'{method_name} - 频谱分析')
            axes[idx, 1].set_xlabel('频率')
            axes[idx, 1].set_ylabel('幅度')
            axes[idx, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_interpolation_quality(original, interpolated, missing_mask=None):
        """
        绘制插值质量图（增强版）
        """
        # 转换为numpy数组
        original = np.array(original).flatten()
        interpolated = np.array(interpolated).flatten()

        # 自动检测缺失值
        if missing_mask is None:
            missing_mask = np.isnan(original)

        # 用插值填充原始数据的缺失值用于显示
        original_filled = original.copy()
        original_filled[missing_mask] = interpolated[missing_mask]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 插值效果图
        x = np.arange(len(original))

        # 绘制原始有效数据
        valid_mask = ~missing_mask
        ax1.plot(x[valid_mask], original[valid_mask], 'b-', label='原始数据(有效)', alpha=0.7, linewidth=1.5)

        # 绘制插值后的完整数据
        ax1.plot(x, interpolated, 'r--', label='插值后完整数据', alpha=0.7, linewidth=1)

        # 标记插值点
        if missing_mask.any():
            ax1.scatter(x[missing_mask], interpolated[missing_mask],
                        color='g', s=50, label='插值点', zorder=5, alpha=0.8)

        ax1.set_title(f'插值效果对比 (插值点数: {missing_mask.sum()})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 局部放大图（选择有插值的区域）
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0:
            # 找到最密集的插值区域
            center = missing_indices[len(missing_indices) // 2]
            start = max(0, center - 30)
            end = min(len(original), center + 31)

            ax2.plot(x[start:end], original_filled[start:end], 'b-', label='原始+插值', marker='o', markersize=4)
            ax2.scatter(x[missing_mask][(missing_mask) & (x >= start) & (x <= end)],
                        interpolated[missing_mask][(missing_mask) & (x >= start) & (x <= end)],
                        color='g', s=80, label='插值点', zorder=5)
            ax2.set_title(f'局部放大图 (区域: {start}-{end})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无插值点', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('局部放大图')

        plt.tight_layout()
        return fig

    @staticmethod
    def safe_plot(func):
        """装饰器：安全绘图，捕获NaN错误"""

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                if "not finite" in str(e) or "NaN" in str(e):
                    print(f"警告：数据包含NaN或inf值，已自动清理")
                    # 自动清理数据
                    args_list = list(args)
                    for i in range(len(args_list)):
                        if hasattr(args_list[i], '__array__'):
                            args_list[i] = np.nan_to_num(args_list[i], nan=0.0, posinf=0.0, neginf=0.0)
                    return func(*args_list, **kwargs)
                else:
                    raise e

        return wrapper


# 修复后的完整使用示例
def demo_visualization():
    """演示如何使用修复后的可视化工具"""

    # 创建测试数据（包含NaN）
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    true_signal = np.sin(x) + 0.2 * np.sin(3 * x)

    # 添加噪声
    noisy = true_signal + np.random.normal(0, 0.15, n)

    # 随机添加NaN
    nan_indices = np.random.choice(n, 30, replace=False)
    noisy_with_nan = noisy.copy()
    noisy_with_nan[nan_indices] = np.nan

    # 简单的插值处理（用线性插值填充NaN）
    denoised = pd.Series(noisy_with_nan).interpolate().values

    # 创建可视化工具实例
    viz = VisualizationEvaluator()

    # 方法1：使用修复后的plot_comparison
    print("绘制对比图...")
    fig1 = viz.plot_comparison(noisy_with_nan, denoised, "去噪效果对比")
    if fig1:
        plt.show()

    # 方法2：使用NaN处理专用版本
    print("\n绘制处理NaN的对比图...")
    fig2 = viz.plot_with_nan_handling(noisy_with_nan, denoised, "含缺失值的数据对比")
    if fig2:
        plt.show()

    # 方法3：绘制插值质量图
    print("\n绘制插值质量图...")
    fig3 = viz.plot_interpolation_quality(noisy_with_nan, denoised)
    if fig3:
        plt.show()

    # 方法4：比较不同方法
    print("\n绘制方法比较图...")
    methods_dict = {
        '原始含噪数据': noisy_with_nan,
        '线性插值': denoised,
        '移动平均': pd.Series(denoised).rolling(5, center=True).mean().fillna(method='bfill').fillna(
            method='ffill').values
    }
    fig4 = viz.plot_method_comparison(methods_dict, "不同处理方法对比")
    if fig4:
        plt.show()


# 运行演示
if __name__ == "__main__":
    demo_visualization()