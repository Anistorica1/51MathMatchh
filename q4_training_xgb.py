import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
df = pd.read_excel('4phase3.xlsx')  # 请替换为实际文件路径

# 删除无用的索引列
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 转换时间列（如果需要）
if '时间' in df.columns:
    df['时间'] = pd.to_datetime(df['时间'])
    # 可选：设置时间为索引
    # df.set_index('时间', inplace=True)

# 查看数据前几行
print("原始数据前5行：")
print(df.head())
print("\n数据信息：")
print(df.info())

# ==================== 2. 特征工程 ====================

# 2.1 构造爆破能量特征 E = Q / (d^2)
# 非爆破时段（爆破点距离=1e9，单段最大药量=0）E值自动为0
df['Blast_Energy'] = df['单段最大药量_kg'] / (df['爆破点距离_m'] ** 2)

# 2.2 处理滞后效应：创建滞后特征（例如滞后1-3期）
lag_periods = [1, 2, 3]  # 可根据实际调整
for lag in lag_periods:
    df[f'Rain_lag{lag}'] = df['降雨量_mm'].shift(lag)
    df[f'PorePressure_lag{lag}'] = df['孔隙水压力_kPa'].shift(lag)
    df[f'MicroSeismic_lag{lag}'] = df['微震事件数'].shift(lag)
    df[f'BlastEnergy_lag{lag}'] = df['Blast_Energy'].shift(lag)

# 2.3 处理累计降雨：计算多时间尺度的累计降雨（3天、7天、15天）
# 注意：您的数据是10分钟间隔，需要根据实际时间间隔调整窗口大小
# 假设10分钟一个点，3天 = 3*24*6 = 432个点
# 这里先使用原始窗口数，您可以根据实际采样频率调整
for window in [432, 1008, 2160]:  # 3天、7天、15天（10分钟间隔）
    df[f'CumRain_{window}'] = df['降雨量_mm'].rolling(window=window, min_periods=1).sum()

# 2.4 处理累计爆破能量（可选）
for window in [432, 1008]:  # 3天、7天
    df[f'CumBlastEnergy_{window}'] = df['Blast_Energy'].rolling(window=window, min_periods=1).sum()

# 2.5 构造位移增量作为因变量
df['Disp_Increment'] = df['表面位移_mm'].diff()  # 当前时刻位移增量

# 2.6 删除因变量为NaN的行
df = df.dropna().reset_index(drop=True)

# ==================== 3. 准备特征矩阵和目标变量 ====================

# 选择特征列（排除原始非特征列）
exclude_cols = ['时间', '表面位移_mm', 'Disp_Increment',
                '单段最大药量_kg', '爆破点距离_m', '降雨量_mm',
                '孔隙水压力_kPa', '微震事件数', 'Blast_Energy']

# 确保所有特征列都是英文名称
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['Disp_Increment']

print(f"\n特征维度: {X.shape}")
print(f"特征列表: {feature_cols}")
print(f"\n特征数据示例：")
print(X.head())

# ==================== 4. 划分训练集和测试集 ====================
# 时序数据按时间顺序划分（前80%训练，后20%测试）
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"\n训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

# ==================== 5. 模型训练（浅层LightGBM，抑制过拟合）====================

# 浅层树模型参数: 较小的max_depth, 较高的min_child_samples, 较低的num_leaves
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 15,           # 较小的叶子数（防止过拟合）
    'max_depth': 4,             # 浅层树深（核心参数，抑制噪声过拟合）
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,           # L1正则化
    'reg_lambda': 0.1,          # L2正则化
    'min_child_samples': 20,    # 叶子节点最小样本数
    'random_state': 42,
    'verbose': 1
}

model = lgb.LGBMRegressor(**lgb_params)

# 训练模型
print("\n开始训练模型...")
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(10), lgb.log_evaluation(50)])

# ==================== 6. 模型评估 ====================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n========== 模型评估结果 ==========")
print(f"训练集 - RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.6f}, MAE: {mean_absolute_error(y_train, y_pred_train):.6f}, R2: {r2_score(y_train, y_pred_train):.4f}")
print(f"测试集 - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.6f}, MAE: {mean_absolute_error(y_test, y_pred_test):.6f}, R2: {r2_score(y_test, y_pred_test):.4f}")

# ==================== 7. 特征重要性分析 ====================
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n========== 特征重要性排序 ==========")
print(feature_importance.head(15))

# ==================== 8. 可视化预测结果 ====================
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际位移增量 (mm)')
plt.ylabel('预测位移增量 (mm)')
plt.title('测试集预测值与实际值对比')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test.values, label='实际值', alpha=0.7, linewidth=1)
plt.plot(range(len(y_test)), y_pred_test, label='预测值', alpha=0.7, linewidth=1)
plt.xlabel('测试集样本序号')
plt.ylabel('位移增量 (mm)')
plt.title('测试集时序预测结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150)
plt.show()

# ==================== 9. 可选：保存模型和预处理特征 ====================
import joblib
joblib.dump(model, 'slope_displacement_model.pkl')
print("\n模型已保存为 'slope_displacement_model.pkl'")

# 保存特征列名供后续预测使用
with open('feature_columns.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

print("\n程序执行完成！")