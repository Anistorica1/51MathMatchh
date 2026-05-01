import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_transition_nodes(disp_data, time_data=None, smooth_win=11,
                          init_ratio=0.15, acc_threshold_sigma=2.5,
                          peak_decay_ratio=0.35, min_continuous=8):
    """
    识别滑坡位移三段式形变阶段的转换节点

    参数:
    ----------
    disp_data : array-like
        位移数据序列
    time_data : array-like, optional
        时间数据序列，若为None则使用索引
    smooth_win : int
        平滑窗口大小（需为奇数）
    init_ratio : float
        初始稳定段比例（0-1之间）
    acc_threshold_sigma : float
        加速度阈值系数（倍数σ）
    peak_decay_ratio : float
        加速度峰值衰减比例（下降到峰值的多少倍以下）
    min_continuous : int
        最小连续点数（用于确认突变）

    返回:
    ----------
    node1 : int
        节点1的索引（匀速→加速）
    node2 : int
        节点2的索引（加速→快速）
    vel : array
        速度序列
    acc : array
        加速度序列
    """

    # 数据预处理
    disp = np.array(disp_data, dtype=float)
    n = len(disp)

    if time_data is None:
        time = np.arange(n)
    else:
        time = np.array(time_data, dtype=float)

    dt = np.mean(np.diff(time)) if n > 1 else 1.0

    # 1. 平滑位移（Savitzky-Golay滤波器）
    if n >= smooth_win and smooth_win % 2 == 1:
        disp_smooth = savgol_filter(disp, min(smooth_win, n if n % 2 == 1 else n - 1), 2)
    else:
        disp_smooth = disp.copy()

    # 2. 计算速度和加速度（中心差分法）
    vel = np.gradient(disp_smooth, dt)
    acc = np.gradient(vel, dt)

    # 对加速度再次平滑（降噪）
    if n >= 9:
        acc_smooth = savgol_filter(acc, min(9, n if n % 2 == 1 else n - 1), 2)
    else:
        acc_smooth = acc.copy()

    # 3. 确定初始稳定段（前 init_ratio 比例的数据）
    n_init = max(5, int(init_ratio * n))
    acc_init = acc_smooth[:n_init]
    acc_init_mean = np.mean(acc_init)
    acc_init_std = np.std(acc_init)

    # 4. 识别节点1（匀速 → 加速）
    threshold1 = acc_init_mean + acc_threshold_sigma * acc_init_std
    node1 = None

    for i in range(n_init, n - min_continuous):
        # 连续 min_continuous 个点加速度超过阈值且持续上升趋势
        window = acc_smooth[i:i + min_continuous]
        if np.all(window > threshold1) and window[-1] > window[0]:
            node1 = i
            break

    # 如果未找到节点1，使用加速度变化率最大点作为备选
    if node1 is None:
        acc_diff = np.abs(np.diff(acc_smooth))
        node1 = np.argmax(acc_diff[n_init:]) + n_init

    # 5. 识别节点2（加速 → 快速）- 找到最明显的转换点
    node2 = None

    if node1 < n - 10:
        # 从节点1开始往后找加速度峰值
        acc_post = acc_smooth[node1:]
        peak_idx_in_post = np.argmax(acc_post)
        peak_val = acc_post[peak_idx_in_post]

        # 只有当峰值足够显著时才寻找衰减点
        if peak_val > acc_init_mean + 2 * acc_init_std:
            threshold2 = peak_val * peak_decay_ratio

            # 完整扫描所有可能的衰减段
            best_score = -np.inf
            best_idx = None

            # 从峰值点后开始扫描
            for j in range(peak_idx_in_post + min_continuous, len(acc_post)):
                # 检查当前点之后的 min_continuous 个点是否都低于阈值
                if j + min_continuous <= len(acc_post):
                    window_below = acc_post[j:j + min_continuous]
                    if np.all(window_below < threshold2):
                        # 计算该转换点的"明显性得分"
                        # 得分考虑：加速度下降的幅度、下降的陡峭程度
                        fall_amplitude = peak_val - np.mean(window_below)
                        # 从峰值到当前点的下降梯度
                        down_slope = (peak_val - acc_post[j]) / max(1, j - peak_idx_in_post)

                        score = fall_amplitude * down_slope

                        if score > best_score:
                            best_score = score
                            best_idx = j

            if best_idx is not None:
                node2 = node1 + best_idx

    # 如果未找到满足阈值条件的节点，使用加速度梯度变化最大点
    if node2 is None and node1 < n - 10:
        # 计算加速度的负梯度（下降最快的区域）
        acc_derivative = -np.gradient(acc_smooth[node1:])  # 取负值，关注加速度下降
        # 平滑梯度
        if len(acc_derivative) >= 5:
            acc_derivative = savgol_filter(acc_derivative, min(5, len(acc_derivative) if len(
                acc_derivative) % 2 == 1 else len(acc_derivative) - 1), 1)

        # 找到加速度下降最快的位置（从后半分找最显著的）
        search_start = max(len(acc_derivative) // 3, peak_idx_in_post if 'peak_idx_in_post' in locals() else 0)
        candidate_idx = np.argmax(acc_derivative[search_start:]) + search_start
        node2 = node1 + candidate_idx

    # 备选方案：使用速度的二阶导数拐点
    if node2 is None or node2 <= node1 + 5:
        if node1 < n - 10:
            # 计算速度的变化率
            vel_acc = np.gradient(vel[node1:])
            # 找到速度加速度最大变化点（从加速到减速的转折）
            if len(vel_acc) > 10:
                # 平滑处理
                vel_acc_smooth = savgol_filter(vel_acc,
                                               min(7, len(vel_acc) if len(vel_acc) % 2 == 1 else len(vel_acc) - 1),
                                               2) if len(vel_acc) >= 7 else vel_acc
                # 找到从正转负的拐点
                zero_crossings = np.where(np.diff(np.sign(vel_acc_smooth)) < 0)[0]
                if len(zero_crossings) > 0:
                    node2 = node1 + zero_crossings[0]
                else:
                    # 如果没有过零点，找变化最大的负梯度点
                    neg_indices = np.where(vel_acc_smooth < 0)[0]
                    if len(neg_indices) > 0:
                        node2 = node1 + neg_indices[np.argmin(vel_acc_smooth[neg_indices])]

    # 最终保障：确保节点2在节点1之后且不超过数据范围
    if node2 is None or node2 <= node1:
        node2 = min(n - 1, node1 + int((n - node1) * 0.6))

    if node2 >= n:
        node2 = n - 1

    return node1, node2, vel, acc_smooth


def plot_results(disp, time, node1, node2, vel, acc, save_path=None):
    """
    绘制三段式变形阶段识别结果

    参数:
    ----------
    disp, time : array
        原始位移和时间数据
    node1, node2 : int
        转换节点索引
    vel, acc : array
        速度和加速度序列
    save_path : str, optional
        保存图片路径
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('滑坡位移三段式形变阶段识别', fontsize=16, fontweight='bold')

    # 1. 位移-时间图
    ax1 = axes[0]
    ax1.plot(time, disp, 'b-', linewidth=1.5, alpha=0.7, label='原始位移')
    ax1.axvline(x=time[node1], color='orange', linestyle='--', linewidth=2,
                label=f'节点1 (匀速→加速): t={time[node1]:.1f}')
    ax1.axvline(x=time[node2], color='red', linestyle='--', linewidth=2,
                label=f'节点2 (加速→快速): t={time[node2]:.1f}')

    # 标注三个区域
    ax1.axvspan(time[0], time[node1], alpha=0.1, color='green', label='①缓慢匀速段')
    ax1.axvspan(time[node1], time[node2], alpha=0.1, color='yellow', label='②加速形变段')
    ax1.axvspan(time[node2], time[-1], alpha=0.1, color='red', label='③快速形变段')

    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('位移', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('位移-时间曲线与阶段划分', fontsize=12)

    # 2. 速度-时间图
    ax2 = axes[1]
    ax2.plot(time, vel, 'g-', linewidth=1.5, alpha=0.7, label='速度')
    ax2.axvline(x=time[node1], color='orange', linestyle='--', linewidth=2)
    ax2.axvline(x=time[node2], color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=np.mean(vel[:node1]), color='green', linestyle=':',
                linewidth=1.5, label=f'匀速段平均速度 = {np.mean(vel[:node1]):.4f}')
    ax2.axhline(y=np.mean(vel[node2:]), color='red', linestyle=':',
                linewidth=1.5, label=f'快速段平均速度 = {np.mean(vel[node2:]):.4f}')

    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('速度', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('速度-时间曲线（速度突变是阶段转换的关键指标）', fontsize=12)

    # 3. 加速度-时间图
    ax3 = axes[2]
    ax3.plot(time, acc, 'r-', linewidth=1.5, alpha=0.7, label='加速度')
    ax3.axvline(x=time[node1], color='orange', linestyle='--', linewidth=2)
    ax3.axvline(x=time[node2], color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # 标注节点1前后的加速度特征
    ax3.axvspan(time[0], time[node1], alpha=0.1, color='green')
    ax3.axvspan(time[node1], time[node2], alpha=0.1, color='yellow')
    ax3.axvspan(time[node2], time[-1], alpha=0.1, color='red')

    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_ylabel('加速度', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('加速度-时间曲线（节点1: 加速度从0→正值；节点2: 加速度从峰值→0）', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()


def analyze_excel_stages(excel_path, sheet_name=0, disp_col=None, time_col=None,
                         smooth_win=11, init_ratio=0.15,
                         acc_threshold_sigma=2.5, peak_decay_ratio=0.35):
    """
    主函数：读取Excel文件，识别三段式形变阶段转换节点

    参数:
    ----------
    excel_path : str
        Excel文件路径
    sheet_name : str or int
        工作表名称或索引（默认0，即第一个工作表）
    disp_col : str or int
        位移数据所在列（列名或索引），若为None则默认第一列
    time_col : str or int
        时间数据所在列（列名或索引），若为None则使用行号作为时间
    smooth_win : int
        平滑窗口大小（需为奇数）
    init_ratio : float
        初始稳定段比例
    acc_threshold_sigma : float
        加速度阈值系数
    peak_decay_ratio : float
        加速度峰值衰减比例

    返回:
    ----------
    results : dict
        包含节点位置、三个阶段的统计信息
    """

    # 读取Excel
    print(f"正在读取文件: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 确定位移列
    if disp_col is None:
        disp = df.iloc[:, 1].values
        print(f"使用第一列作为位移数据")
    else:
        if isinstance(disp_col, str):
            disp = df[disp_col].values
        else:
            disp = df.iloc[:, disp_col].values
        print(f"使用列 '{disp_col}' 作为位移数据")

    # 确定时间列
    if time_col is None:
        time = np.arange(len(disp))
        print("使用行索引作为时间（无时间列）")
    else:
        if isinstance(time_col, str):
            time = df[time_col].values
        else:
            time = df.iloc[:, time_col].values
        print(f"使用列 '{time_col}' 作为时间数据")

    # 去除缺失值
    mask = ~(np.isnan(disp) | np.isnan(time))
    disp = disp[mask]
    time = time[mask]

    print(f"有效数据点数: {len(disp)}")

    # 识别转换节点
    print("\n正在识别转换节点...")
    node1, node2, vel, acc = find_transition_nodes(
        disp, time,
        smooth_win=smooth_win,
        init_ratio=init_ratio
    )

    # 计算各阶段的统计信息
    stage1_disp = disp[:node1]
    stage2_disp = disp[node1:node2]
    stage3_disp = disp[node2:]

    stage1_vel = vel[:node1]
    stage2_vel = vel[node1:node2]
    stage3_vel = vel[node2:]

    results = {
        'node1_index': node1,
        'node2_index': node2,
        'node1_time': time[node1],
        'node2_time': time[node2],
        'stage1': {
            'name': '缓慢匀速形变段',
            'velocity_mean': np.mean(stage1_vel),
            'velocity_std': np.std(stage1_vel),
            'duration': time[node1] - time[0] if len(time) > 1 else node1,
            'displacement_range': [stage1_disp[0], stage1_disp[-1]]
        },
        'stage2': {
            'name': '加速形变段',
            'velocity_mean': np.mean(stage2_vel),
            'velocity_std': np.std(stage2_vel),
            'velocity_increase_rate': (vel[node2 - 1] - vel[node1]) / (time[node2] - time[node1]) if time[node2] !=
                                                                                                     time[node1] else 0,
            'duration': time[node2] - time[node1] if len(time) > 1 else node2 - node1,
            'displacement_range': [stage2_disp[0], stage2_disp[-1]]
        },
        'stage3': {
            'name': '快速形变段',
            'velocity_mean': np.mean(stage3_vel),
            'velocity_std': np.std(stage3_vel),
            'duration': time[-1] - time[node2] if len(time) > 1 else len(disp) - node2,
            'displacement_range': [stage3_disp[0], stage3_disp[-1]]
        }
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("阶段转换节点识别结果")
    print("=" * 60)
    print(f"\n节点1（匀速 → 加速）: 索引 = {node1}, 时间 = {time[node1]:.2f}")
    print(f"节点2（加速 → 快速）: 索引 = {node2}, 时间 = {time[node2]:.2f}")

    print("\n" + "-" * 40)
    print("各阶段统计信息:")
    print("-" * 40)

    for stage in ['stage1', 'stage2', 'stage3']:
        s = results[stage]
        print(f"\n{s['name']}:")
        print(f"  平均速度: {s['velocity_mean']:.4f}")
        print(f"  速度标准差: {s['velocity_std']:.4f}")
        if stage == 'stage2':
            print(f"  速度增长率: {s['velocity_increase_rate']:.4f}")
        print(f"  持续时长: {s['duration']:.2f}")
        print(f"  位移范围: [{s['displacement_range'][0]:.2f}, {s['displacement_range'][1]:.2f}]")

    # 绘制结果
    plot_results(disp, time, node1, node2, vel, acc)

    return results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 方法1: 直接指定Excel文件路径
    excel_file = "附件2：位移时序数据-问题2.xlsx"  # 请替换为你的文件路径
    try:
        results = analyze_excel_stages(
            excel_path=excel_file,
            sheet_name=0,  # 第一个工作表
            disp_col=None,  # 默认第一列为位移
            time_col="时间_小时",  # 无时间列则使用行号
            smooth_win=11,  # 平滑窗口
            init_ratio=0.15,  # 前15%作为初始段
            acc_threshold_sigma=2.5,  # 2.5倍标准差阈值
            peak_decay_ratio=0.35  # 峰值下降到35%作为节点2
        )

        # 如果需要导出结果到Excel
        # results_df = pd.DataFrame({
        #     '节点': ['节点1(匀速→加速)', '节点2(加速→快速)'],
        #     '索引': [results['node1_index'], results['node2_index']],
        #     '时间': [results['node1_time'], results['node2_time']]
        # })
        # results_df.to_excel('转换节点结果.xlsx', index=False)
        # print("\n结果已导出到 '转换节点结果.xlsx'")

    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{excel_file}'")
        print("请修改代码中的 'excel_file' 变量为正确的文件路径")

        # 方法2: 如果你有数据可以直接粘贴测试
        print("\n如需测试，可以使用以下模拟数据:")
        # 生成模拟的三段式数据
        np.random.seed(42)
        t = np.arange(100)
        # 匀速段 (0-30)
        y1 = 0.5 * t[:30] + np.random.normal(0, 0.2, 30)
        # 加速段 (30-70)
        y2 = 15 + 0.5 * np.arange(40) + 0.02 * np.arange(40) ** 2 + np.random.normal(0, 0.3, 40)
        # 快速段 (70-100)
        y3 = 15 + 0.5 * 40 + 0.02 * 40 ** 2 + 3.5 * np.arange(30) + np.random.normal(0, 0.5, 30)
        disp_mock = np.concatenate([y1, y2, y3])

        results = analyze_excel_stages(
            excel_path=None,
            disp_col=None,
            time_col=None,
            smooth_win=11
        )
        # 注意：模拟数据需要手动传入，这里示意用法
##

