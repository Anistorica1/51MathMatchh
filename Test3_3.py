"""
边坡表面位移预测模型
基于：降雨量、孔隙水压力、深部位移、微震事件数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========================== 1. 数据加载 ==========================
# 请根据你的实际文件路径修改
# 假设你的数据文件为 Excel 或 CSV 格式
file_path = "监测数据cleaned.xlsx"  # 或 .csv

# 尝试读取数据
try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    print(f"成功加载数据，样本数: {len(df)}")
    print(f"变量: {list(df.columns)}")
except FileNotFoundError:
    print("未找到数据文件，使用模拟数据演示...")
    # 如果没有数据文件，生成模拟数据（基于你的相关性结果）
    np.random.seed(42)
    n = 10000

    # 根据斯皮尔曼相关系数生成模拟数据
    # 深部位移(0.83) 孔隙水压力(0.79) 降雨量(0.40) 微震(0.12)
    deep_displacement = np.random.normal(50, 20, n)
    pore_pressure = 20 + 0.6 * deep_displacement + np.random.normal(0, 10, n)
    rainfall = np.random.exponential(10, n)  # 降雨量偏态分布
    microseismic = np.random.poisson(5, n)

    # 表面位移 = 主要受深部位移和孔隙水压力影响
    surface_displacement = (0.8 * deep_displacement +
                            0.3 * pore_pressure +
                            0.1 * rainfall +
                            0.01 * microseismic +
                            np.random.normal(0, 5, n))

    df = pd.DataFrame({
        '降雨量_mm': rainfall,
        '孔隙水压力_kPa': pore_pressure,
        '深部位移_mm': deep_displacement,
        '微震事件数': microseismic,
        '表面位移_mm': surface_displacement
    })
    print("使用模拟数据（基于你的相关性结果生成）")

# 查看数据基本信息
print(f"\n数据形状: {df.shape}")
print(df.head())

# ========================== 2. 数据预处理 ==========================
# 对偏态的降雨量进行 Box-Cox 变换
print("\n" + "=" * 70)
print("数据预处理")
print("=" * 70)

pt = PowerTransformer(method='yeo-johnson')  # 支持负值
df['降雨量_transformed'] = pt.fit_transform(df[['降雨量_mm']])

# 特征和目标变量
feature_cols = ['降雨量_transformed', '孔隙水压力_kPa', '深部位移_mm', '微震事件数']
target_col = '表面位移_mm'

X = df[feature_cols]
y = df[target_col]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# ========================== 3. 多元线性回归 (核心数学模型) ==========================
print("\n" + "=" * 70)
print("模型1: 多元线性回归")
print("=" * 70)

# 使用 statsmodels 获取详细统计信息
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

print("\n回归方程系数:")
print("-" * 50)
for var, coef, pval in zip(['常数项'] + feature_cols, model_sm.params, model_sm.pvalues):
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
    print(f"  {var:20s}: {coef:10.4f}  (p={pval:.4f} {sig})")

print(f"\n模型统计量:")
print(f"  R² = {model_sm.rsquared:.4f}")
print(f"  调整R² = {model_sm.rsquared_adj:.4f}")
print(f"  F统计量 = {model_sm.fvalue:.2f} (p={model_sm.f_pvalue:.4e})")
print(f"  AIC = {model_sm.aic:.2f}")
print(f"  BIC = {model_sm.bic:.2f}")

# 输出数学公式
print("\n" + "=" * 50)
print("📐 数学模型公式:")
print("=" * 50)
eqn = f"表面位移_mm = {model_sm.params[0]:.4f}"
for i, col in enumerate(feature_cols):
    sign = "+" if model_sm.params[i + 1] >= 0 else "-"
    eqn += f" {sign} {abs(model_sm.params[i + 1]):.4f} × {col}"
print(eqn)

# 线性回归预测
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ========================== 4. 岭回归 (处理多重共线性) ==========================
print("\n" + "=" * 70)
print("模型2: 岭回归 (处理潜在共线性)")
print("=" * 70)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f"岭回归系数: {dict(zip(feature_cols, ridge.coef_))}")

# ========================== 5. 随机森林 (捕捉非线性) ==========================
print("\n" + "=" * 70)
print("模型3: 随机森林回归 (捕捉非线性和交互作用)")
print("=" * 70)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 特征重要性
importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)
print("\n特征重要性:")
print(importance.to_string(index=False))

# ========================== 6. 模型评估与对比 ==========================
print("\n" + "=" * 70)
print("模型性能对比 (测试集)")
print("=" * 70)

results = []
for name, y_pred in [('多元线性回归', y_pred_lr),
                     ('岭回归', y_pred_ridge),
                     ('随机森林', y_pred_rf)]:
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append([name, r2, rmse, mae])
    print(f"\n{name}:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} mm")
    print(f"  MAE = {mae:.4f} mm")

# ========================== 7. 残差诊断 ==========================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 预测值 vs 实际值 (线性回归)
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际表面位移 (mm)')
axes[0, 0].set_ylabel('预测表面位移 (mm)')
axes[0, 0].set_title('线性回归: 预测 vs 实际')

# 预测值 vs 实际值 (随机森林)
axes[0, 1].scatter(y_test, y_pred_rf, alpha=0.5, s=10)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际表面位移 (mm)')
axes[0, 1].set_ylabel('预测表面位移 (mm)')
axes[0, 1].set_title('随机森林: 预测 vs 实际')

# 残差分布
residuals = y_test - y_pred_lr
axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(x=0, color='r', linestyle='--')
axes[0, 2].set_xlabel('残差 (mm)')
axes[0, 2].set_ylabel('频数')
axes[0, 2].set_title('残差分布 (线性回归)')

# Q-Q图
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q图 (正态性检验)')

# 残差 vs 预测值
axes[1, 1].scatter(y_pred_lr, residuals, alpha=0.5, s=10)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('预测值 (mm)')
axes[1, 1].set_ylabel('残差 (mm)')
axes[1, 1].set_title('残差 vs 预测值')

# 特征重要性条形图
axes[1, 2].barh(importance['特征'], importance['重要性'], color='steelblue')
axes[1, 2].set_xlabel('重要性')
axes[1, 2].set_title('随机森林特征重要性')

plt.tight_layout()
plt.savefig('边坡位移模型诊断图.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================== 8. 保存模型 ==========================
import joblib

# 保存最佳模型 (随机森林通常表现更好)
joblib.dump(rf, 'slope_displacement_model.pkl')
joblib.dump(pt, 'rainfall_transformer.pkl')
print("\n" + "=" * 70)
print("模型已保存:")
print("  - slope_displacement_model.pkl (随机森林模型)")
print("  - rainfall_transformer.pkl (降雨量变换器)")

# ========================== 9. 预测示例 ==========================
print("\n" + "=" * 70)
print("预测示例")
print("=" * 70)

sample = pd.DataFrame({
    '降雨量_mm': [25.0],
    '孔隙水压力_kPa': [45.0],
    '深部位移_mm': [32.5],
    '微震事件数': [8]
})
sample['降雨量_transformed'] = pt.transform(sample[['降雨量_mm']])
sample_X = sample[feature_cols]

prediction = rf.predict(sample_X)[0]
print(f"输入条件:")
print(f"  降雨量: {sample['降雨量_mm'].iloc[0]} mm")
print(f"  孔隙水压力: {sample['孔隙水压力_kPa'].iloc[0]} kPa")
print(f"  深部位移: {sample['深部位移_mm'].iloc[0]} mm")
print(f"  微震事件数: {sample['微震事件数'].iloc[0]}")
print(f"\n预测表面位移: {prediction:.2f} mm")