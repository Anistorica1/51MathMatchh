import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ==================== 1. 读取训练数据 ====================
df = pd.read_excel('4phase1.xlsx')
df = df.sort_values('时间').reset_index(drop=True)

# ==================== 2. 数据平滑 ====================
df['表面位移_smooth'] = df['表面位移_mm'].rolling(window=5, min_periods=1).mean()

for col in ['降雨量_mm', '孔隙水压力_kPa', '微震事件数']:
    df[f'{col}_smooth'] = df[col].rolling(window=3, min_periods=1).mean()

target_col = '表面位移_smooth'

feature_cols = [
    '降雨量_mm_smooth', '孔隙水压力_kPa_smooth', '微震事件数_smooth',
    '爆破点距离_m', '单段最大药量_kg'
]

# 处理异常
df['爆破点距离_m'] = df['爆破点距离_m'].replace(1e9, np.nan)
df['爆破点距离_m'] = df['爆破点距离_m'].fillna(method='ffill').fillna(df['爆破点距离_m'].median())

# ==================== 3. lag特征 ====================
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

df = df.dropna().reset_index(drop=True)

# ==================== 4. 构造数据 ====================
X = df[feature_cols].values
y = df[target_col].values

# ==================== 5. 归一化 ====================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# ==================== 6. 构造序列 ====================
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# ==================== 7. 划分数据 ====================
train_size = int(len(X_seq) * 0.7)
val_size = int(len(X_seq) * 0.15)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]

X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

# ==================== 8. 模型 ====================
def build_model(input_shape):
    model = Sequential([
        LSTM(32, activation='tanh', input_shape=input_shape,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model((seq_length, X_train.shape[2]))
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

# ==================== 10. 评估 ====================
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

# 预测
y_test_pred = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred)
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

evaluate(y_test_true, y_test_pred, "测试集")

# ==================== 11. ===== 实验集预测 ===== ====================
df_exp = pd.read_excel('4phase1lab.xlsx')
df_exp = df_exp.sort_values('时间').reset_index(drop=True)

# 平滑
df_exp['表面位移_smooth'] = np.nan  # 空列也保留结构

for col in ['降雨量_mm', '孔隙水压力_kPa', '微震事件数']:
    df_exp[f'{col}_smooth'] = df_exp[col].rolling(window=3, min_periods=1).mean()

# 处理异常
df_exp['爆破点距离_m'] = df_exp['爆破点距离_m'].replace(1e9, np.nan)
df_exp['爆破点距离_m'] = df_exp['爆破点距离_m'].fillna(method='ffill').fillna(df_exp['爆破点距离_m'].median())

# lag
for lag in range(1, lag_steps + 1):
    for col in ['降雨量_mm_smooth', '孔隙水压力_kPa_smooth', '微震事件数_smooth',
                '爆破点距离_m', '单段最大药量_kg']:
        df_exp[f'{col}_lag{lag}'] = df_exp[col].shift(lag)

df_exp = df_exp.dropna().reset_index(drop=True)

# 构造X
X_exp = df_exp[feature_cols].values
X_exp_scaled = scaler_X.transform(X_exp)

# 序列
X_exp_seq, _ = create_sequences(X_exp_scaled, np.zeros(len(X_exp_scaled)), seq_length)

# 预测
y_pred_scaled = model.predict(X_exp_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 对齐
df_result = df_exp.iloc[seq_length:].copy()
df_result['预测位移_mm'] = y_pred

# 保存
df_result.to_excel('预测结果q4_1.xlsx', index=False)

print("\n✅ 实验集预测完成，已保存为 预测结果.xlsx")