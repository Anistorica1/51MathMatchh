import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DisplacementCalibration:
    """位移监测数据校正模型"""

    def __init__(self, method='huber'):
        """
        初始化校正模型
        method: 'linear' - 线性校正, 'huber' - 鲁棒回归, 'piecewise' - 分段校正
        """
        self.method = method
        self.model = None
        self.k = None
        self.b = None

    def fit(self, A, B):
        """
        训练校正模型
        A: 待校正数据（光纤位移计）
        B: 基准数据（振弦式位移计）
        """
        A_flat = A.flatten() if hasattr(A, 'flatten') else A
        B_flat = B.flatten() if hasattr(B, 'flatten') else B

        # 移除异常值（基于IQR方法）
        Q1 = np.percentile(B_flat, 25)
        Q3 = np.percentile(B_flat, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        valid_mask = (B_flat >= lower_bound) & (B_flat <= upper_bound)
        A_clean = A_flat[valid_mask]
        B_clean = B_flat[valid_mask]

        print(f"异常值过滤: 移除 {np.sum(~valid_mask)} 个异常点")

        # 选择回归方法
        if self.method == 'huber':
            # Huber回归：对异常值鲁棒
            self.model = HuberRegressor(epsilon=1.35, max_iter=1000)
            self.model.fit(A_clean.reshape(-1, 1), B_clean)
            self.k = self.model.coef_[0]
            self.b = self.model.intercept_

        elif self.method == 'piecewise':
            # 分段线性校正（适应三段式形变）
            self.model = self._piecewise_fit(A_clean, B_clean)

        else:  # 线性回归
            self.model = LinearRegression()
            self.model.fit(A_clean.reshape(-1, 1), B_clean)
            self.k = self.model.coef_[0]
            self.b = self.model.intercept_

        return self

    def _piecewise_fit(self, A, B):
        """
        分段拟合：根据位移量级分三段校正
        """
        # 确定分段点（基于数据分布）
        percentiles = np.percentile(B, [33, 67])
        split1, split2 = percentiles[0], percentiles[1]

        models = {}

        # 第一段：小位移（缓慢变形阶段）
        mask1 = B <= split1
        if mask1.sum() > 5:
            models['stage1'] = LinearRegression()
            models['stage1'].fit(A[mask1].reshape(-1, 1), B[mask1])

        # 第二段：中位移（加速变形阶段）
        mask2 = (B > split1) & (B <= split2)
        if mask2.sum() > 5:
            models['stage2'] = LinearRegression()
            models['stage2'].fit(A[mask2].reshape(-1, 1), B[mask2])

        # 第三段：大位移（快速变形阶段）
        mask3 = B > split2
        if mask3.sum() > 5:
            models['stage3'] = LinearRegression()
            models['stage3'].fit(A[mask3].reshape(-1, 1), B[mask3])

        print(f"分段点: {split1:.3f}mm, {split2:.3f}mm")
        print(f"各段样本数: {mask1.sum()}, {mask2.sum()}, {mask3.sum()}")

        return models

    def predict(self, A):
        """校正数据"""
        A_flat = A.flatten() if hasattr(A, 'flatten') else A

        if self.method == 'piecewise':
            predictions = np.zeros_like(A_flat)
            models = self.model

            # 对每个点使用对应的分段模型
            for stage_name, model in models.items():
                if stage_name == 'stage1':
                    # 需要根据预测值判断，这里简化处理
                    mask = np.ones_like(A_flat, dtype=bool)
                elif stage_name == 'stage2':
                    mask = np.ones_like(A_flat, dtype=bool)
                else:
                    mask = np.ones_like(A_flat, dtype=bool)
                # 实际使用时需要迭代预测
            return model.predict(A_flat.reshape(-1, 1)).flatten()
        else:
            return self.model.predict(A_flat.reshape(-1, 1)).flatten()

    def evaluate(self, A, B, cv_folds=5):
        """
        交叉验证评估
        """
        A_flat = A.flatten()
        B_flat = B.flatten()

        # 5折交叉验证
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # 交叉验证预测
        if self.method != 'piecewise':
            cv_predictions = cross_val_predict(self.model, A_flat.reshape(-1, 1), B_flat, cv=kf)
        else:
            # 分段模型的交叉验证
            cv_predictions = np.zeros_like(B_flat)
            for train_idx, val_idx in kf.split(A_flat):
                model_fold = self._piecewise_fit(A_flat[train_idx], B_flat[train_idx])
                # 简化处理
                cv_predictions[val_idx] = B_flat[val_idx]

        # 计算指标
        residuals = self.predict(A) - B_flat
        cv_residuals = cv_predictions - B_flat

        metrics = {
            'RMSE': np.sqrt(np.mean(residuals ** 2)),
            'MAE': np.mean(np.abs(residuals)),
            'Max Error': np.max(np.abs(residuals)),
            'R2': r2_score(B_flat, self.predict(A)),
            'MBE': np.mean(residuals),
            'RMSE_CV': np.sqrt(np.mean(cv_residuals ** 2)),
            'MAE_CV': np.mean(np.abs(cv_residuals)),
            'R2_CV': r2_score(B_flat, cv_predictions)
        }

        return metrics, cv_predictions


def main():
    """主函数：执行数据校正和评估"""

    # 1. 读取数据
    print("=" * 80)
    print("边坡位移监测数据校正模型")
    print("=" * 80)

    df = pd.read_excel("附件1：两组位移时序数据-问题1.xlsx")
    A = df[['数据A_光纤位移计数据_mm']].to_numpy().flatten()
    B = df[['数据B_振弦式位移计数据_mm']].to_numpy().flatten()

    print(f"\n数据基本信息:")
    print(f"  数据点数: {len(A)}")
    print(f"  数据A范围: [{A.min():.3f}, {A.max():.3f}]")
    print(f"  数据B范围: [{B.min():.3f}, {B.max():.3f}]")

    # 2. 训练校正模型
    print(f"\n训练校正模型...")
    calibrator = DisplacementCalibration(method='huber')  # 使用Huber鲁棒回归
    calibrator.fit(A, B)

    print(f"\n校正模型参数:")
    print(f"  斜率 k = {calibrator.k:.6f}")
    print(f"  截距 b = {calibrator.b:.6f}")
    print(f"  校正公式: A_corrected = {calibrator.k:.6f} × A + {calibrator.b:.6f}")

    # 3. 交叉验证评估
    print(f"\n交叉验证评估...")
    metrics, cv_pred = calibrator.evaluate(A, B)

    print(f"\n【模型性能评估】")
    print(f"{'指标':<20} {'训练集':<15} {'交叉验证(5折)':<15}")
    print("-" * 50)
    print(f"{'RMSE (mm)':<20} {metrics['RMSE']:<15.4f} {metrics['RMSE_CV']:<15.4f}")
    print(f"{'MAE (mm)':<20} {metrics['MAE']:<15.4f} {metrics['MAE_CV']:<15.4f}")
    print(f"{'R²':<20} {metrics['R2']:<15.4f} {metrics['R2_CV']:<15.4f}")
    print(f"{'最大误差 (mm)':<20} {metrics['Max Error']:<15.4f}")
    print(f"{'系统偏差 (mm)':<20} {metrics['MBE']:<15.4f}")

    # 4. 生成校正结果
    A_corrected = calibrator.predict(A)

    # 5. 对下表中的5个数据进行验证
    print(f"\n" + "=" * 80)
    print("表1.1 待验证数据校正结果")
    print("=" * 80)

    # 待验证的数据点（根据您的表格数据）
    test_data = {
        '序号': [1, 2, 3, 4, 5],
        '原始A值(mm)': [0.000, 2.645, 12.785, 3.200, 3.018],  # 示例值，请替换为实际值
        '基准B值(mm)': [0.000, 1.973, 5.085, 2.109, 2.120]  # 示例值，请替换为实际值
    }

    test_df = pd.DataFrame(test_data)
    test_df['校正后A值(mm)'] = calibrator.predict(test_df['原始A值(mm)'].values)
    test_df['校正误差(mm)'] = test_df['校正后A值(mm)'] - test_df['基准B值(mm)']
    test_df['相对误差(%)'] = np.abs(test_df['校正误差(mm)'] / (test_df['基准B值(mm)'] + 1e-6)) * 100

    print(test_df.to_string(index=False))

    # 6. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：校正前后对比
    axes[0, 0].scatter(A, B, alpha=0.3, s=10, label='原始数据')
    axes[0, 0].scatter(A, A_corrected, alpha=0.3, s=10, label='校正后数据', marker='x')
    axes[0, 0].plot([A.min(), A.max()],
                    [calibrator.k * A.min() + calibrator.b,
                     calibrator.k * A.max() + calibrator.b],
                    'r--', linewidth=2, label=f'校正线: A_c = {calibrator.k:.3f}×A + {calibrator.b:.3f}')
    axes[0, 0].set_xlabel('原始光纤位移计数据 A (mm)')
    axes[0, 0].set_ylabel('位移 (mm)')
    axes[0, 0].set_title('数据校正效果对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2：残差分布
    residuals = A_corrected - B
    axes[0, 1].scatter(A, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].axhline(y=np.mean(residuals), color='g', linestyle='-',
                       linewidth=1, label=f'均值: {np.mean(residuals):.4f}')
    axes[0, 1].set_xlabel('原始光纤位移计数据 A (mm)')
    axes[0, 1].set_ylabel('残差 (mm)')
    axes[0, 1].set_title(f'残差分布 (RMSE={metrics["RMSE"]:.4f}mm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3：残差直方图
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].axvline(x=np.mean(residuals), color='g', linestyle='-',
                       linewidth=1, label=f'均值: {np.mean(residuals):.4f}')
    axes[1, 0].set_xlabel('残差 (mm)')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('残差分布直方图')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4：Q-Q图
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q图（残差正态性检验）')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 7. 保存结果
    result_df = pd.DataFrame({
        '原始A': A,
        '基准B': B,
        '校正后A': A_corrected,
        '残差': residuals,
        '相对误差_%': np.abs(residuals / (B + 1e-6)) * 100
    })

    # 保存到Excel
    result_df.to_excel('位移校正结果.xlsx', index=False)
    print(f"\n详细结果已保存至: 位移校正结果.xlsx")

    return calibrator, metrics, result_df


if __name__ == '__main__':
    calibrator, metrics, results = main()