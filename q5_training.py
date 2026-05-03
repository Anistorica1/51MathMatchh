import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# ==================== 1. 读取数据 ====================
file_path = '5phase3.xlsx'
df = pd.read_excel(file_path)
df = df.sort_values('时间').reset_index(drop=True)

print("=" * 60)
print("数据统计分析")
print("=" * 60)
print(f"位移范围: {df['表面位移_mm'].min():.4f} - {df['表面位移_mm'].max():.4f}")
print(f"位移均值: {df['表面位移_mm'].mean():.4f} mm")
print(f"位移标准差: {df['表面位移_mm'].std():.4f} mm")
print(f"变异系数: {df['表面位移_mm'].std() / df['表面位移_mm'].mean():.4f}")

# ==================== 2. 数据平滑处理 ====================
print("\n" + "=" * 60)
print("数据平滑处理")
print("=" * 60)

# 移动平均平滑位移（减少噪声）
window_size = 5
df['表面位移_smooth'] = df['表面位移_mm'].rolling(window=window_size, min_periods=1).mean()

# 同样平滑特征
for col in ['降雨量_mm', '孔隙水压力_kPa', '微震事件数']:
    df[f'{col}_smooth'] = df[col].rolling(window=3, min_periods=1).mean()

# 使用平滑后的数据
target_col = '表面位移_smooth'  # 改为直接预测位移绝对值
feature_cols = ['孔隙水压力_kPa', '微震事件数', '爆破点距离_m',
                '单段最大药量_kg', '干湿入渗系数']

# 处理爆破点距离异常
df['爆破点距离_m'] = df['爆破点距离_m'].replace(1e9, np.nan)
df['爆破点距离_m'] = df['爆破点距离_m'].fillna(method='ffill').fillna(df['爆破点距离_m'].median())

# ==================== 3. 创建滞后特征 ====================
lag_steps = 5
for lag in range(1, lag_steps + 1):
    for col in feature_cols:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 更新特征列
new_feature_cols = []
for col in feature_cols:
    new_feature_cols.append(col)
    for lag in range(1, lag_steps + 1):
        new_feature_cols.append(f'{col}_lag{lag}')

feature_cols = new_feature_cols

# 删除NaN
df = df.dropna().reset_index(drop=True)

print(f"特征数量: {len(feature_cols)}")
print(f"处理后数据量: {len(df)}")

# ==================== 4. 不再进行差分处理 ====================
# 直接使用平滑后的位移作为目标变量
# 移除差分相关代码

# 提取特征和目标
X = df[feature_cols].values
y = df[target_col].values  # 直接使用位移绝对值

# ==================== 5. 归一化（重要修改）====================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 对目标变量也进行归一化，因为位移绝对值可能有较大范围
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


# ==================== 6. 创建序列 ====================
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])  # 预测未来一个时间点的位移
    return np.array(X_seq), np.array(y_seq)


seq_length = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# ==================== 7. 划分数据集 ====================
train_size = int(len(X_seq) * 0.7)
val_size = int(len(X_seq) * 0.15)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]

X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

print(f"\n训练集: {X_train.shape}")
print(f"验证集: {X_val.shape}")
print(f"测试集: {X_test.shape}")


# ==================== 8. LSTM模型 ====================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, activation='tanh', return_sequences=True,
             input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        LSTM(16, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_lstm_model((seq_length, X_train.shape[2]))
model.summary()

# ==================== 9. 训练 ====================
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==================== 保存模型和预处理对象 ====================
import joblib

# 保存模型
model.save('phase3_displacement.h5')

# 保存scaler
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')  # 保存目标变量的scaler

# 保存预处理参数
preprocess_params = {
    'feature_cols': feature_cols,
    'seq_length': seq_length,
    'lag_steps': lag_steps,
    'window_size_target': 5,
    'window_size_feature': 3,
    'feature_cols_raw': ['降雨量_mm', '微震事件数', '爆破点距离_m',
                         '单段最大药量_kg', '干湿入渗系数']
}
joblib.dump(preprocess_params, 'preprocess_params_displacement.pkl')

print("模型和预处理参数已保存！")
print("保存的文件：phase3_displacement.h5, scaler_X.pkl, scaler_y.pkl, preprocess_params_displacement.pkl")

# ==================== 10. 预测和评估（反归一化后评估）====================
# 预测（在归一化空间）
y_train_pred_scaled = model.predict(X_train, verbose=0)
y_val_pred_scaled = model.predict(X_val, verbose=0)
y_test_pred_scaled = model.predict(X_test, verbose=0)

# 反归一化到原始位移值
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

# 反归一化真实值
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


# 评估函数（直接评估位移绝对值）
def evaluate_displacement(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 计算平均绝对百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  MAE: {mae:.4f} mm")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return rmse, mae, r2, mape


print("\n" + "=" * 60)
print("模型评估（表面位移绝对值预测）")
print("=" * 60)

evaluate_displacement(y_train_actual, y_train_pred, "训练集")
evaluate_displacement(y_val_actual, y_val_pred, "验证集")
evaluate_displacement(y_test_actual, y_test_pred, "测试集")

# ==================== 11. 可视化 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 预测vs实际（测试集）
n_show = min(200, len(y_test_actual))
axes[0, 0].plot(y_test_actual[:n_show], label='实际位移', linewidth=1.5, alpha=0.7)
axes[0, 0].plot(y_test_pred[:n_show], label='预测位移', linewidth=1.5, alpha=0.7)
axes[0, 0].set_xlabel('时间点')
axes[0, 0].set_ylabel('表面位移 (mm)')
axes[0, 0].set_title(f'测试集 - 位移预测对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 散点图（预测vs实际）
axes[0, 1].scatter(y_test_actual, y_test_pred, alpha=0.5, s=10)
axes[0, 1].plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()],
                'r--', lw=2, label='理想预测')
axes[0, 1].set_xlabel('实际位移 (mm)')
axes[0, 1].set_ylabel('预测位移 (mm)')
axes[0, 1].set_title(f'预测 vs 实际 (R²={r2_score(y_test_actual, y_test_pred):.4f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 预测误差
errors = y_test_actual - y_test_pred
axes[0, 2].plot(errors[:n_show], linewidth=1, color='red', alpha=0.7)
axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 2].axhline(y=np.std(errors), color='gray', linestyle='--', alpha=0.5)
axes[0, 2].axhline(y=-np.std(errors), color='gray', linestyle='--', alpha=0.5)
axes[0, 2].set_xlabel('时间点')
axes[0, 2].set_ylabel('预测误差 (mm)')
axes[0, 2].set_title(f'预测误差 (std={np.std(errors):.4f} mm)')
axes[0, 2].grid(True, alpha=0.3)

# 损失曲线
axes[1, 0].plot(history.history['loss'], label='训练损失')
axes[1, 0].plot(history.history['val_loss'], label='验证损失')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss (MSE)')
axes[1, 0].set_title('训练曲线')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 残差分布
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('预测残差 (mm)')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title(f'残差分布 (均值={np.mean(errors):.4f} mm)')
axes[1, 1].grid(True, alpha=0.3)

# 误差指标柱状图
metrics = ['RMSE', 'MAE']
train_metrics = [np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                 mean_absolute_error(y_train_actual, y_train_pred)]
val_metrics = [np.sqrt(mean_squared_error(y_val_actual, y_val_pred)),
               mean_absolute_error(y_val_actual, y_val_pred)]
test_metrics = [np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                mean_absolute_error(y_test_actual, y_test_pred)]

x = np.arange(len(metrics))
width = 0.25
axes[1, 2].bar(x - width, train_metrics, width, label='训练集', alpha=0.8)
axes[1, 2].bar(x, val_metrics, width, label='验证集', alpha=0.8)
axes[1, 2].bar(x + width, test_metrics, width, label='测试集', alpha=0.8)
axes[1, 2].set_xlabel('评估指标')
axes[1, 2].set_ylabel('误差 (mm)')
axes[1, 2].set_title('各数据集误差对比')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('LSTM_位移预测_结果.png', dpi=300)
plt.show()

# 打印预测示例
print("\n" + "=" * 60)
print("预测示例（最后10个测试样本）")
print("=" * 60)
sample_df = pd.DataFrame({
    '实际位移(mm)': y_test_actual[-10:],
    '预测位移(mm)': y_test_pred[-10:],
    '绝对误差(mm)': np.abs(y_test_actual[-10:] - y_test_pred[-10:])
})
print(sample_df.to_string(index=False))