import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# ==================== 检测器类 ====================
class CurvatureNodeDetector:
    """
    基于曲率/二阶导数的运动阶段转换节点检测器
    适用于检测：缓慢匀速段 -> 加速段 -> 快速匀速段的转换点
    """

    def __init__(self,
                 smooth_method='savgol',
                 window_length=31,
                 polyorder=3,
                 sigma=2.0,
                 threshold_factor=0.15,
                 min_distance=10):
        """
        参数:
        - smooth_method: 'savgol' 或 'gaussian' 或 'loess_simple'
        - window_length: Savgol滤波窗口（应为奇数）
        - polyorder: Savgol多项式阶数
        - sigma: 高斯滤波标准差
        - threshold_factor: 二阶导数峰值阈值因子（相对最大峰值的比例）
        - min_distance: 两个节点间的最小距离（防止假检测）
        """
        self.smooth_method = smooth_method
        self.window_length = window_length
        self.polyorder = polyorder
        self.sigma = sigma
        self.threshold_factor = threshold_factor
        self.min_distance = min_distance

    def smooth_data(self, data):
        """数据平滑处理"""
        if self.smooth_method == 'savgol':
            # 确保窗口长度不超过数据长度
            window = min(self.window_length, len(data) if len(data) % 2 == 0 else len(data) - 1)
            if window < 3:
                window = 3
            if window % 2 == 0:
                window -= 1
            return savgol_filter(data, window, self.polyorder)

        elif self.smooth_method == 'gaussian':
            return gaussian_filter1d(data, sigma=self.sigma)

        else:
            return data

    def compute_derivatives(self, data, dx=1.0):
        """计算一阶和二阶导数"""
        first_deriv = np.gradient(data, dx)
        second_deriv = np.gradient(first_deriv, dx)
        return first_deriv, second_deriv

    def _find_contiguous_regions(self, mask):
        """找到连续的True区域"""
        regions = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                regions.append((start, i - 1))
                start = None
        if start is not None:
            regions.append((start, len(mask) - 1))
        return regions

    def detect_transition_nodes(self, data, time=None):
        """
        检测转换节点
        返回: {
            'transition_1': idx1,
            'transition_2': idx2,
            'accel_region': [start, end],
            'confidence': {'node1': float, 'node2': float}
        }
        """
        if time is None:
            time = np.arange(len(data))

        # 1. 平滑数据
        data_smooth = self.smooth_data(data)

        # 2. 计算二阶导数
        _, second_deriv = self.compute_derivatives(data_smooth)

        # 3. 二次平滑二阶导数
        second_deriv_smooth = gaussian_filter1d(second_deriv, sigma=1.0)

        # 4. 识别加速区域（二阶导数为正的区域）
        accel_mask = second_deriv_smooth > 0
        accel_regions = self._find_contiguous_regions(accel_mask)

        if len(accel_regions) == 0:
            warnings.warn("未检测到明显的加速区域")
            return None

        # 选择主要的加速区域（最长的连续区域）
        main_accel = max(accel_regions, key=lambda x: x[1] - x[0])

        # 5. 在加速区域内找二阶导数的峰值
        accel_indices = np.arange(main_accel[0], main_accel[1] + 1)
        accel_values = second_deriv_smooth[accel_indices]
        peak_idx = accel_indices[np.argmax(accel_values)]
        peak_value = np.max(accel_values)

        # 6. 检测第一个节点：二阶导数首次显著上升超过阈值
        threshold = self.threshold_factor * peak_value
        transition_1 = None

        for i in range(main_accel[0], peak_idx):
            if second_deriv_smooth[i] > threshold:
                transition_1 = i
                break

        if transition_1 is None:
            transition_1 = main_accel[0] + max(0, (peak_idx - main_accel[0]) // 5)

        # 7. 检测第二个节点：从峰值下降回落到接近0的位置
        transition_2 = None
        zero_threshold = threshold * 0.3

        for i in range(peak_idx, min(main_accel[1], len(second_deriv_smooth) - 1)):
            if second_deriv_smooth[i] < zero_threshold and second_deriv_smooth[i + 1] < zero_threshold:
                transition_2 = i
                break

        if transition_2 is None:
            transition_2 = main_accel[1] - max(1, (main_accel[1] - peak_idx) // 4)

        # 8. 确保最小距离约束
        if transition_2 - transition_1 < self.min_distance:
            mid = (transition_1 + transition_2) // 2
            transition_1 = max(0, mid - self.min_distance // 2)
            transition_2 = min(len(data) - 1, mid + self.min_distance // 2)

        # 9. 计算置信度
        confidence = {
            'node1': min(1.0, second_deriv_smooth[transition_1] / peak_value if transition_1 else 0),
            'node2': min(1.0, 1.0 - second_deriv_smooth[transition_2] / peak_value if transition_2 else 0)
        }

        result = {
            'transition_1': transition_1,
            'transition_2': transition_2,
            'accel_region': main_accel,
            'accel_peak': peak_idx,
            'peak_value': peak_value,
            'second_derivative': second_deriv_smooth,
            'confidence': confidence,
            'data_smooth': data_smooth,
            'time': time
        }

        return result

    def plot_results(self, data, result, figsize=(14, 10)):
        """可视化检测结果"""
        if result is None:
            print("无检测结果")
            return

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # 原始数据与平滑数据
        axes[0].plot(result['time'], data, 'b-', alpha=0.5, label='原始数据', linewidth=1)
        axes[0].plot(result['time'], result['data_smooth'], 'r-', label='平滑数据', linewidth=2)
        axes[0].axvline(x=result['time'][result['transition_1']], color='g', linestyle='--',
                        label=f'节点1: 缓慢→加速', linewidth=2)
        axes[0].axvline(x=result['time'][result['transition_2']], color='orange', linestyle='--',
                        label=f'节点2: 加速→快速', linewidth=2)
        axes[0].set_ylabel('位置 / 位移')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 二阶导数
        axes[1].plot(result['time'], result['second_derivative'], 'purple', label='二阶导数', linewidth=1.5)
        axes[1].fill_between(result['time'], 0, result['second_derivative'],
                             where=(result['second_derivative'] > 0),
                             alpha=0.3, color='green', label='加速区域')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].axvline(x=result['time'][result['transition_1']], color='g', linestyle='--', alpha=0.7)
        axes[1].axvline(x=result['time'][result['transition_2']], color='orange', linestyle='--', alpha=0.7)
        axes[1].scatter(result['time'][result['accel_peak']], result['peak_value'],
                        color='red', s=100, zorder=5, label=f'峰值加速度点')
        axes[1].set_ylabel('二阶导数 (加速度特征)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 速度估计
        first_deriv, _ = self.compute_derivatives(result['data_smooth'])
        axes[2].plot(result['time'], first_deriv, 'brown', label='速度（一阶导数）', linewidth=1.5)
        axes[2].axvline(x=result['time'][result['transition_1']], color='g', linestyle='--', alpha=0.7)
        axes[2].axvline(x=result['time'][result['transition_2']], color='orange', linestyle='--', alpha=0.7)

        # 标注阶段
        axes[2].axvspan(result['time'][0], result['time'][result['transition_1']],
                        alpha=0.1, color='blue', label='缓慢匀速段')
        axes[2].axvspan(result['time'][result['transition_1']], result['time'][result['transition_2']],
                        alpha=0.2, color='green', label='加速段')
        axes[2].axvspan(result['time'][result['transition_2']], result['time'][-1],
                        alpha=0.1, color='red', label='快速匀速段')

        axes[2].set_xlabel('时间 / 帧索引')
        axes[2].set_ylabel('速度')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'运动阶段转换节点检测 (置信度: 节点1={result["confidence"]["node1"]:.2%}, '
                     f'节点2={result["confidence"]["node2"]:.2%})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ==================== Excel处理函数 ====================
def detect_nodes_from_excel(file_path,
                            position_col='表面位移_mm',
                            time_col=None,
                            smooth_method='savgol',
                            window_length=51,
                            threshold_factor=0.12,
                            min_distance_ratio=0.01,
                            auto_optimize=True,
                            plot=True):
    """
    从Excel文件检测运动阶段转换节点

    参数:
    - file_path: Excel文件路径
    - position_col: 位置/位移列名（默认'表面位移_mm'）
    - time_col: 时间列名（可选，None则使用行索引）
    - smooth_method: 平滑方法 ('savgol' 或 'gaussian')
    - window_length: 平滑窗口（自动调整如果太大）
    - threshold_factor: 阈值因子（0.1-0.3，越小越敏感）
    - min_distance_ratio: 最小节点间距占数据长度的比例
    - auto_optimize: 是否自动优化参数
    - plot: 是否显示图形

    返回:
    - result: 检测结果字典
    - nodes_df: 节点信息DataFrame
    """

    # 读取数据
    print(f"\n正在读取文件: {file_path}")
    df = pd.read_excel(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 获取位置数据
    if position_col not in df.columns:
        # 尝试查找包含'位移'或'位置'的列
        possible_cols = [col for col in df.columns if '位移' in col or '位置' in col or 'position' in col.lower()]
        if possible_cols:
            position_col = possible_cols[0]
            print(f"自动选择位置列: {position_col}")
        else:
            # 使用第一列数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                position_col = numeric_cols[0]
                print(f"自动选择数值列: {position_col}")
            else:
                raise ValueError(f"未找到位置列，请指定position_col参数")

    position = df[position_col].values
    print(f"位置数据范围: {position.min():.2f} - {position.max():.2f}")

    # 获取时间数据
    if time_col and time_col in df.columns:
        time = df[time_col].values
        print(f"使用时间列: {time_col}")
    else:
        time = np.arange(len(position))
        print(f"使用行索引作为时间")

    n_points = len(position)
    print(f"数据点数: {n_points}")

    # 自动优化参数
    if auto_optimize:
        # 根据数据长度调整窗口
        recommended_window = min(51, n_points // 10 if n_points // 10 % 2 == 1 else n_points // 10 - 1)
        if recommended_window < 5:
            recommended_window = 5
        window_length = recommended_window

        # 调整最小距离
        min_distance = max(10, int(n_points * min_distance_ratio))

        # 根据数据噪声水平调整阈值（通过计算二阶导数的标准差）
        vel = np.diff(position)
        acc = np.diff(vel)
        noise_level = np.std(acc[:min(100, len(acc))]) if len(acc) > 100 else np.std(acc)
        if noise_level > np.percentile(np.abs(acc), 50):
            threshold_factor = 0.2  # 噪声大，提高阈值
        else:
            threshold_factor = 0.12  # 噪声小，降低阈值

        print(f"\n自动优化参数:")
        print(f"  窗口长度: {window_length}")
        print(f"  阈值因子: {threshold_factor}")
        print(f"  最小节点间距: {min_distance}")
    else:
        min_distance = max(10, int(n_points * min_distance_ratio))

    # 创建检测器
    detector = CurvatureNodeDetector(
        smooth_method=smooth_method,
        window_length=window_length,
        polyorder=min(3, window_length - 1),
        threshold_factor=threshold_factor,
        min_distance=min_distance
    )

    # 检测节点
    print("\n正在检测节点...")
    result = detector.detect_transition_nodes(position, time=time)

    if result is None:
        print("❌ 未检测到有效节点！")
        print("建议：")
        print("  1. 检查数据是否有明显的加速段")
        print("  2. 尝试降低 threshold_factor (如 0.08)")
        print("  3. 调整 window_length (如 31 或 41)")
        return None, None

    # 计算各段统计信息
    print("\n" + "=" * 60)
    print("检测结果")
    print("=" * 60)

    t1, t2 = result['transition_1'], result['transition_2']

    # 计算各段速度
    if t1 > 0:
        v1 = np.mean(np.diff(position[:t1]))
        v1_std = np.std(np.diff(position[:t1]))
    else:
        v1, v1_std = 0, 0

    if t2 > t1:
        v2 = np.mean(np.diff(position[t1:t2]))
        v2_std = np.std(np.diff(position[t1:t2]))
    else:
        v2, v2_std = 0, 0

    if t2 < n_points - 1:
        v3 = np.mean(np.diff(position[t2:]))
        v3_std = np.std(np.diff(position[t2:]))
    else:
        v3, v3_std = 0, 0

    print(f"\n📍 节点1 (缓慢→加速):")
    print(f"   索引: {t1}")
    print(f"   时间: {time[t1]:.3f}")
    print(f"   位置: {position[t1]:.3f}")
    print(f"   置信度: {result['confidence']['node1']:.2%}")

    print(f"\n📍 节点2 (加速→快速):")
    print(f"   索引: {t2}")
    print(f"   时间: {time[t2]:.3f}")
    print(f"   位置: {position[t2]:.3f}")
    print(f"   置信度: {result['confidence']['node2']:.2%}")

    print(f"\n📊 各段速度分析:")
    print(f"   段1 (0-{t1}): 速度 = {v1:.4f} ± {v1_std:.4f}")
    print(f"   段2 ({t1}-{t2}): 速度 = {v2:.4f} ± {v2_std:.4f}")
    print(f"   段3 ({t2}-end): 速度 = {v3:.4f} ± {v3_std:.4f}")

    print(f"\n⏱️  加速段时长: {time[t2] - time[t1]:.3f}")
    print(f"📈 速度提升: {v3 / v1:.2f}倍" if v1 > 0 else "")

    # 创建结果DataFrame
    nodes_df = pd.DataFrame({
        '节点': ['节点1 (缓慢→加速)', '节点2 (加速→快速)'],
        '索引': [t1, t2],
        '时间': [time[t1], time[t2]],
        '位置': [position[t1], position[t2]],
        '置信度': [result['confidence']['node1'], result['confidence']['node2']]
    })

    # 保存结果到Excel
    # output_file = file_path.replace('.xlsx', '_节点检测结果.xlsx').replace('.xls', '_节点检测结果.xlsx')
    #
    # with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    #     nodes_df.to_excel(writer, sheet_name='节点信息', index=False)
    #
    #     # 添加详细数据
    #     detail_df = pd.DataFrame({
    #         '索引': time,
    #         '时间': time,
    #         '原始位置': position,
    #         '平滑位置': result['data_smooth'],
    #         '二阶导数': result['second_derivative']
    #     })
    #     detail_df.to_excel(writer, sheet_name='详细数据', index=False)
    #
    # print(f"\n💾 结果已保存到: {output_file}")

    # 可视化
    if plot:
        detector.plot_results(position, result)

    return result, nodes_df


# ==================== 批量处理多个文件 ====================
def batch_detect_nodes(file_list, output_summary=True, **kwargs):
    """
    批量处理多个Excel文件

    参数:
    - file_list: 文件路径列表
    - output_summary: 是否输出汇总结果
    - **kwargs: 传递给detect_nodes_from_excel的参数
    """
    all_results = []

    for file_path in file_list:
        print(f"\n{'=' * 60}")
        print(f"处理文件: {file_path}")
        print('=' * 60)

        try:
            result, nodes_df = detect_nodes_from_excel(file_path, **kwargs)
            if result:
                all_results.append({
                    '文件': file_path,
                    '节点1_时间': result['time'][result['transition_1']],
                    '节点2_时间': result['time'][result['transition_2']],
                    '节点1_位置': result['data_smooth'][result['transition_1']],
                    '节点2_位置': result['data_smooth'][result['transition_2']],
                    '置信度1': result['confidence']['node1'],
                    '置信度2': result['confidence']['node2']
                })
        except Exception as e:
            print(f"处理失败: {e}")

    if output_summary and all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_excel('批量检测汇总.xlsx', index=False)
        print(f"\n📊 汇总结果已保存到: 批量检测汇总.xlsx")
        print("\n汇总表:")
        print(summary_df.to_string())

    return all_results


# ==================== 主程序 ====================
if __name__ == "__main__":
    # ===== 方式1：处理单个文件 =====
    # 修改这里的文件路径为你的实际路径
    file_path = '附件2：位移时序数据-问题2.xlsx'  # ← 改成你的文件路径

    result, nodes_df = detect_nodes_from_excel(
        file_path=file_path,
        position_col='表面位移_mm',  # 位置列名
        time_col=None,  # 时间列名（None则使用索引）
        smooth_method='savgol',  # 平滑方法
        window_length=51,  # 平滑窗口（自动优化时会覆盖）
        threshold_factor=0.1,  # 阈值因子（0.1-0.3）
        auto_optimize=True,  # 自动优化参数
        plot=True  # 显示图形
    )

    # 如果需要手动调整参数（当自动检测不理想时）
    # result, nodes_df = detect_nodes_from_excel(
    #     file_path=file_path,
    #     position_col='表面位移_mm',
    #     auto_optimize=False,          # 关闭自动优化
    #     window_length=41,             # 手动设置窗口
    #     threshold_factor=0.10,        # 手动设置阈值
    #     plot=True
    # )

    # ===== 方式2：批量处理多个文件 =====
    # file_list = [
    #     'E:/py/51MMc/data1.xlsx',
    #     'E:/py/51MMc/data2.xlsx',
    #     'E:/py/51MMc/data3.xlsx'
    # ]
    # batch_detect_nodes(file_list, position_col='表面位移_mm')

    # ===== 方式3：直接使用检测器类 =====
    # df = pd.read_excel('your_data.xlsx')
    # data = df['表面位移_mm'].values
    #
    # detector = CurvatureNodeDetector(window_length=51, threshold_factor=0.12)
    # result = detector.detect_transition_nodes(data)
    #
    # if result:
    #     print(f"节点1: {result['transition_1']}, 节点2: {result['transition_2']}")
    #     detector.plot_results(data, result)