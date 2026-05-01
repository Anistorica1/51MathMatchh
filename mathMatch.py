import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


def fun1():
    # 读取数据
    df1 = pd.read_excel("附件1：两组位移时序数据-问题1.xlsx")
    A = df1[['数据A_光纤位移计数据_mm']].to_numpy()  # 形状 (n, 1)
    B = df1[['数据B_振弦式位移计数据_mm']].to_numpy()  # 形状 (n, 1)

    # 删除示例数据生成代码（不要 np.random.seed(42)）

    # ========== 2. 线性校正模型 ==========
    # A已经是二维数组，不需要reshape
    model = LinearRegression()
    model.fit(A, B)

    # 获取斜率和截距（注意索引）
    k = model.coef_[0][0]  # 二维数组，取第一行第一列
    b = model.intercept_[0]  # 一维数组，取第一个元素

    print(f"校正模型: A_corrected = {k:.6f} * A + {b:.6f}")

    # 对 A 进行校正
    A_corrected = model.predict(A)  # 返回 (n, 1) 形状

    # ========== 3. 计算偏差指标 ==========
    residuals = A_corrected - B  # 校正后的值与 B 的偏差
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))

    print(f"\n校正效果评估:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Max Error: {max_error:.6f}")

    # ========== 4. 可视化对比 ==========
    # 将二维数组转换为一维用于绘图
    A_flat = A.flatten()
    B_flat = B.flatten()
    A_corrected_flat = A_corrected.flatten()
    residuals_flat = residuals.flatten()

    plt.figure(figsize=(12, 5))

    # 子图1：校正前后对比
    plt.subplot(1, 2, 1)
    plt.scatter(A_flat, B_flat, alpha=0.6, label='原始 B (目标)', s=30)
    plt.scatter(A_flat, A_corrected_flat, alpha=0.6, label='校正后的 A', s=30, marker='x')

    # 绘制拟合线
    x_line = np.array([A_flat.min(), A_flat.max()])
    y_line = k * x_line + b
    plt.plot(x_line, y_line, 'r--', label=f'拟合线: B = {k:.3f}*A + {b:.3f}', linewidth=1)

    plt.xlabel('A (原始数据)')
    plt.ylabel('数值')
    plt.title('校正效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：残差分布
    plt.subplot(1, 2, 2)
    plt.scatter(A_flat, residuals_flat, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('A (原始数据)')
    plt.ylabel('残差 (校正后 - B)')
    plt.title(f'残差分布 (RMSE={rmse:.4f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 5. 保存校正结果 ==========
    result_df = pd.DataFrame({
        '原始_A': A_flat,
        '目标_B': B_flat,
        '校正后_A': A_corrected_flat,
        '偏差': residuals_flat
    })
    print("\n前10行结果预览:")
    print(result_df.head(10))

    # 保存到 Excel 文件（取消注释即可保存）
    # result_df.to_excel('校正结果.xlsx', index=False)
    # print("\n结果已保存到 '校正结果.xlsx'")

    return result_df


if __name__ == '__main__':
    result = fun1()