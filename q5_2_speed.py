import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
# 请根据你的Excel文件结构修改参数
# 假设Excel有两列：'time' 和 'displacement'
file_path = 'Speed.xlsx'  # 请修改为你的文件路径
df = pd.read_excel(file_path)

# 查看数据结构
print("数据前5行：")
print(df.head())
print("\n数据信息：")
print(df.info())

# 获取时间和位移数据（根据实际列名修改）
# 方案1：如果列名是 'time' 和 'displacement'
# time = df['time'].values
# displacement = df['displacement'].values

# 方案2：自动识别前两列
time = df.iloc[:, 0].values
displacement = df.iloc[:, 1].values

# 去除NaN值
mask = ~(np.isnan(time) | np.isnan(displacement))
time = time[mask]
displacement = displacement[mask]

print(f"\n有效数据点数量: {len(time)}")
print(f"时间范围: {time[0]:.3f} ~ {time[-1]:.3f}")

# ==================== 2. 降噪处理 ====================
# 方法1：Savitzky-Golay滤波器（推荐，适用于平滑且保留趋势）
# window_length: 窗口大小（必须是奇数），约为数据点数的5-20%
window_length = min(len(displacement) // 5 + 1, 51)
window_length = window_length if window_length % 2 == 1 else window_length + 1
polyorder = min(3, window_length - 1)  # 多项式阶数

displacement_denoised = savgol_filter(displacement, window_length, polyorder)

# 方法2：低通Butterworth滤波器（备选方案，可取消注释使用）
"""
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 计算采样频率（平均采样间隔）
dt = np.mean(np.diff(time))
fs = 1.0 / dt
cutoff_freq = fs * 0.2  # 截止频率为采样频率的20%
displacement_denoised = butter_lowpass_filter(displacement, cutoff_freq, fs)
"""

# ==================== 3. 计算速度 ====================
# 使用中心差分法计算速度
velocity = np.zeros_like(time)
for i in range(len(time)):
    if i == 0:
        # 前向差分
        dt = time[1] - time[0]
        velocity[i] = (displacement_denoised[1] - displacement_denoised[0]) / dt
    elif i == len(time) - 1:
        # 后向差分
        dt = time[-1] - time[-2]
        velocity[i] = (displacement_denoised[-1] - displacement_denoised[-2]) / dt
    else:
        # 中心差分
        dt = time[i + 1] - time[i - 1]
        velocity[i] = (displacement_denoised[i + 1] - displacement_denoised[i - 1]) / dt

# 可选：对速度曲线再进行一次轻微平滑
velocity_smoothed = savgol_filter(velocity, min(len(velocity) // 10 + 1, 15), 2)

# ==================== 4. 计算预警阈值 ====================
global_max_velocity = np.max(np.abs(velocity_smoothed))
threshold_blue = global_max_velocity * 0.75
threshold_yellow = global_max_velocity * 0.85
threshold_orange = global_max_velocity * 0.92
threshold_red = global_max_velocity * 0.98

print(f"\n全局最大速度: {global_max_velocity:.4f}")
print(f"蓝色预警线 (75%): {threshold_blue:.4f}")
print(f"黄色预警线 (85%): {threshold_yellow:.4f}")
print(f"橙色预警线 (92%): {threshold_orange:.4f}")
print(f"红色预警线 (98%): {threshold_red:.4f}")

# ==================== 5. 绘制位移曲线 ====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 位移曲线 - 叠加原始数据和降噪后数据
ax1.plot(time, displacement, 'o', markersize=2, alpha=0.5, color='gray', label='原始数据')
ax1.plot(time, displacement_denoised, 'b-', linewidth=2, label='降噪后位移', zorder=5)
ax1.set_ylabel('位移', fontsize=12)
ax1.set_title('位移-时间曲线', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([time[0], time[-1]])

# 速度曲线 - 带预警线
ax2.plot(time, velocity_smoothed, 'g-', linewidth=2, label='速度')
ax2.fill_between(time, threshold_blue, velocity_smoothed.max(),
                 color='blue', alpha=0.1, label='正常区域')
ax2.fill_between(time, threshold_yellow, threshold_blue,
                 color='yellow', alpha=0.2, label='75%~85%')
ax2.fill_between(time, threshold_orange, threshold_yellow,
                 color='orange', alpha=0.3, label='85%~92%')
ax2.fill_between(time, threshold_red, threshold_orange,
                 color='red', alpha=0.4, label='92%~98%')

# 绘制预警线
ax2.axhline(y=threshold_blue, color='blue', linestyle='--', linewidth=1.5, label=f'蓝色75% ({threshold_blue:.2f})')
ax2.axhline(y=threshold_yellow, color='yellow', linestyle='--', linewidth=1.5,
            label=f'黄色85% ({threshold_yellow:.2f})')
ax2.axhline(y=threshold_orange, color='orange', linestyle='--', linewidth=1.5,
            label=f'橙色92% ({threshold_orange:.2f})')
ax2.axhline(y=threshold_red, color='red', linestyle='--', linewidth=1.5, label=f'红色98% ({threshold_red:.2f})')

ax2.set_xlabel('时间', fontsize=12)
ax2.set_ylabel('速度', fontsize=12)
ax2.set_title(f'速度-时间曲线 (最大速度: {global_max_velocity:.4f})', fontsize=14)
ax2.legend(loc='upper right', ncol=2, fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('displacement_velocity_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 6. 额外分析：速度超过各阈值的时间段 ====================
print("\n" + "=" * 50)
print("速度超过各预警线的时间段分析：")
print("=" * 50)


def find_exceeding_periods(time, velocity, threshold, name, color):
    exceeding = velocity >= threshold
    periods = []
    in_period = False
    start_time = None

    for i, exceed in enumerate(exceeding):
        if exceed and not in_period:
            in_period = True
            start_time = time[i]
        elif not exceed and in_period:
            in_period = False
            periods.append((start_time, time[i]))

    if in_period:
        periods.append((start_time, time[-1]))

    print(f"\n{name} (阈值: {threshold:.4f}):")
    if periods:
        for start, end in periods:
            duration = end - start
            print(f"  时间段: {start:.6f} ~ {end:.6f}, 持续时间: {duration:.6f}")
    else:
        print(f"  无超过该阈值的时间段")

    return periods


find_exceeding_periods(time, velocity_smoothed, threshold_blue, "蓝色预警线 (75%)", "blue")
find_exceeding_periods(time, velocity_smoothed, threshold_yellow, "黄色预警线 (85%)", "yellow")
find_exceeding_periods(time, velocity_smoothed, threshold_orange, "橙色预警线 (92%)", "orange")
find_exceeding_periods(time, velocity_smoothed, threshold_red, "红色预警线 (98%)", "red")

# ==================== 7. 统计信息输出 ====================
print("\n" + "=" * 50)
print("统计信息：")
print("=" * 50)
print(f"总数据点数: {len(time)}")
print(f"时间跨度: {time[-1] - time[0]:.6f}")
print(f"位移范围: [{np.min(displacement_denoised):.6f}, {np.max(displacement_denoised):.6f}]")
print(f"速度范围: [{np.min(velocity_smoothed):.6f}, {np.max(velocity_smoothed):.6f}]")
print(f"平均速度: {np.mean(np.abs(velocity_smoothed)):.6f}")
print(f"速度标准差: {np.std(velocity_smoothed):.6f}")