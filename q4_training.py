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

# ==================== 2. 数据平滑处理（关键！）====================
print("\n" + "=" * 60)
print("数据平滑处理")
print("=" * 60)

# 移动平均平滑位移（减少噪声）
window_size = 5  # 根据你的数据采样频率调整
df['表面位移_smooth'] = df['表面位移_mm'].rolling(window=window_size, min_periods=1).mean()

# 同样平滑特征
for col in ['干湿入渗系数', '孔隙水压力_kPa', '微震事件数']:
    df[f'{col}_smooth'] = df[col].rolling(window=3, min_periods=1).mean()

# 使用平滑后的数据
target_col = '表面位移_smooth'
# feature_cols = ['孔隙水压力_kPa', '微震事件数', '爆破点距离_m',
#                 '单段最大药量_kg','干湿入渗系数']

# 处理爆破点距离异常
# df['爆破点距离_m'] = df['爆破点距离_m'].replace(1e9, np.nan)
# df['爆破点距离_m'] = df['爆破点距离_m'].fillna(method='ffill').fillna(df['爆破点距离_m'].median())

df['是否爆破'] = (df['单段最大药量_kg'] > 0).astype(int)

# 将是否爆破加入特征列表
feature_cols = ['孔隙水压力_kPa', '微震事件数', '爆破点距离_m',
                '单段最大药量_kg', '干湿入渗系数', '是否爆破']

# ==================== 3. 创建滞后特征 ====================
lag_steps = 5  # 减少滞后步数
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

# ==================== 4. 差分处理（去趋势）====================
# 对位移进行一阶差分，预测变化量而非绝对值
df['位移_diff'] = df[target_col].diff()
df = df.dropna().reset_index(drop=True)

target_col = '位移_diff'  # 预测变化量

# 提取特征和目标
X = df[feature_cols].values
y = df[target_col].values

# ==================== 5. 归一化 ====================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 目标变量不需要归一化（差分后接近0均值）
y_centered = y - np.mean(y)


# ==================== 6. 创建序列 ====================
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


seq_length = 5  # 减小序列长度
X_seq, y_seq = create_sequences(X_scaled, y_centered, seq_length)

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


# ==================== 8. 极简LSTM模型 ====================
def build_simple_lstm(input_shape):
    model = Sequential([
        LSTM(16, activation='tanh', input_shape=input_shape,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_simple_lstm((seq_length, X_train.shape[2]))
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
model.save('phase3.h5')

# 保存scaler
joblib.dump(scaler_X, 'scaler3.pkl')

# 保存预处理参数
preprocess_params = {
    'feature_cols': feature_cols,           # 特征列名
    'seq_length': seq_length,               # 序列长度
    'lag_steps': lag_steps,                 # 滞后步数
    'window_size_target': 5,                # 目标变量平滑窗口
    'window_size_feature': 3,               # 特征平滑窗口
    'feature_cols_raw': ['降雨量_mm', '微震事件数', '爆破点距离_m',
                         '单段最大药量_kg', '干湿入渗系数'],
    'y_mean': np.mean(y)                    # 保存y的均值（用于反中心化）
}
joblib.dump(preprocess_params, 'preprocess_params3.pkl')

print("模型和预处理参数已保存！")
print("保存的文件：phase3.h5, scaler.pkl, preprocess_params.pkl")
# ==================== 10. 预测和评估 ====================
# 预测
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)


# 反差分（将预测的变化量转换回位移）
def inverse_diff(diff_pred, actual_shifted):
    """将差分预测转换回原始位移"""
    # 这里需要知道初始值，简化处理：直接评估差分预测的准确性
    return diff_pred


# 评估（在差分空间）
def evaluate_diff(y_true, y_pred, name):
    # 计算预测的方向准确率
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  方向准确率: {direction_acc:.2%}")
    return rmse, mae, r2


print("\n" + "=" * 60)
print("模型评估（位移变化量预测）")
print("=" * 60)

evaluate_diff(y_train, y_train_pred.flatten(), "训练集")
evaluate_diff(y_val, y_val_pred.flatten(), "验证集")
evaluate_diff(y_test, y_test_pred.flatten(), "测试集")

# ==================== 11. 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 预测vs实际
n_show = min(200, len(y_test))
axes[0, 0].plot(y_test[:n_show], label='实际变化量', linewidth=1, alpha=0.7)
axes[0, 0].plot(y_test_pred[:n_show], label='预测变化量', linewidth=1, alpha=0.7)
axes[0, 0].set_xlabel('时间点')
axes[0, 0].set_ylabel('位移变化量 (mm)')
axes[0, 0].set_title(f'测试集 - 位移变化量预测')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 方向准确率
axes[0, 1].hist(np.sign(y_test) * np.sign(y_test_pred.flatten()), bins=3)
axes[0, 1].set_xticks([-1, 0, 1])
axes[0, 1].set_xticklabels(['方向相反', '零变化', '方向正确'])
axes[0, 1].set_title(f'预测方向分布 (准确率: {np.mean(np.sign(y_test) == np.sign(y_test_pred.flatten())):.2%})')
axes[0, 1].grid(True, alpha=0.3)

# 损失曲线
axes[1, 0].plot(history.history['loss'], label='训练损失')
axes[1, 0].plot(history.history['val_loss'], label='验证损失')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('训练曲线')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 残差
residuals = y_test - y_test_pred.flatten()
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('预测残差')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title(f'残差分布 (std={np.std(residuals):.6f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('LSTM_改进版_结果.png', dpi=300)
plt.show()