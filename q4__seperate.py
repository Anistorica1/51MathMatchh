import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 读取数据
# =========================
file_path = "附件4：监测数据（训练集与实验集）-问题4.xlsx"  # 修改为你的路径

df = pd.read_excel(file_path)

# 假设第一列是时间
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df.set_index(df.columns[0], inplace=True)

# 第一列为表面位移
displacement = df.iloc[:, 0]

# =========================
# 2. 缺失值处理（爆破相关字段）
# =========================
df = df.fillna(0)

# =========================
# 3. 平滑处理（降低噪声）
# =========================
# 滑动平均
window_size = 5
disp_smooth = displacement.rolling(window=window_size, center=True).mean()

# 填补边缘NaN
disp_smooth = disp_smooth.fillna(method='bfill').fillna(method='ffill')

# =========================
# 4. 计算速度和加速度
# =========================
# 一阶差分（速度）
velocity = disp_smooth.diff()

# 二阶差分（加速度）
acceleration = velocity.diff()

# 去除NaN
velocity = velocity.fillna(0)
acceleration = acceleration.fillna(0)

# =========================
# 5. 自动分阶段（核心）
# =========================

# 用分位数作为阈值（更稳健）
v1 = velocity.quantile(0.33)
v2 = velocity.quantile(0.66)

# 阶段标签：
# 1 = 缓慢匀速
# 2 = 加速
# 3 = 快速
stage = []

for v in velocity:
    if v <= v1:
        stage.append(1)
    elif v <= v2:
        stage.append(2)
    else:
        stage.append(3)

df['stage'] = stage

# =========================
# 6. 可视化结果
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df.index, displacement, label='原始位移', alpha=0.5)
plt.plot(df.index, disp_smooth, label='平滑位移', linewidth=2)

# 用颜色标记阶段
colors = {1: 'green', 2: 'orange', 3: 'red'}
for s in [1, 2, 3]:
    idx = df['stage'] == s
    plt.scatter(df.index[idx], disp_smooth[idx],
                color=colors[s], s=10, label=f'阶段{s}')

plt.legend()
plt.title("位移分阶段结果")
plt.xlabel("时间")
plt.ylabel("位移")
plt.show()

# =========================
# 7. 输出结果
# =========================
df.to_excel("阶段划分结果.xlsx")

print("阶段划分完成，结果已保存！")