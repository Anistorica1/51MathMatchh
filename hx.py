import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 数据加载与预处理 ====================

def load_and_preprocess_data():
    """加载训练集和实验集数据"""
    # 读取训练集（问题4的训练数据）
    train_df = pd.read_excel('附件4：监测数据（训练集与实验集）-问题4.xlsx',
                             sheet_name='训练集')

    # 读取实验集（用于最终预测）
    test_df = pd.read_excel('附件4：监测数据（训练集与实验集）-问题4.xlsx',
                            sheet_name='实验集')

    # 训练集重命名列
    train_df.columns = ['序号', '表面位移_mm', '降雨量_mm', '孔隙水压力_kPa',
                        '微震事件数', '爆破点距离_m', '单段最大药量_kg', '空列', '时间']
    train_df = train_df.drop(columns=['空列', '序号'], errors='ignore')

    # 实验集重命名列
    test_df.columns = ['时间', '阶段标签', '表面位移_mm', '降雨量_mm', '孔隙水压力_kPa',
                       '微震事件数', '爆破点距离_m', '单段最大药量_kg']

    # 转换时间列
    train_df['时间'] = pd.to_datetime(train_df['时间'])
    test_df['时间'] = pd.to_datetime(test_df['时间'])

    # 爆破数据空值处理：爆破点距离填充10表示无限远，单段最大药量填充0表示无爆破
    train_df['爆破点距离_m'] = train_df['爆破点距离_m'].fillna(10.0)
    train_df['单段最大药量_kg'] = train_df['单段最大药量_kg'].fillna(0.0)

    test_df['爆破点距离_m'] = test_df['爆破点距离_m'].fillna(10.0)
    test_df['单段最大药量_kg'] = test_df['单段最大药量_kg'].fillna(0.0)

    # 表面位移空值处理（实验集表面位移为空）
    test_df['表面位移_mm'] = test_df['表面位移_mm'].fillna(0)

    # 计算时间间隔（小时）
    train_df['时间间隔_小时'] = train_df['时间'].diff().dt.total_seconds() / 3600
    train_df['时间间隔_小时'].fillna(1 / 6, inplace=True)  # 默认10分钟间隔

    test_df['时间间隔_小时'] = test_df['时间'].diff().dt.total_seconds() / 3600
    test_df['时间间隔_小时'].fillna(1 / 6, inplace=True)

    return train_df, test_df


# ==================== 2. 信号去噪与阶段划分 ====================

def denoise_signal(signal, window_length=11, polyorder=3):
    """使用Savitzky-Golay滤波器去噪"""
    if len(signal) < window_length:
        return signal
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def calculate_velocity_and_acceleration(displacement, time_interval):
    """计算位移速度和加速度"""
    # 速度计算 (mm/h)
    velocity = np.zeros(len(displacement))
    for i in range(1, len(displacement)):
        if time_interval[i] > 0:
            velocity[i] = (displacement[i] - displacement[i - 1]) / time_interval[i]

    # 加速度计算 (mm/h^2)
    acceleration = np.zeros(len(displacement))
    for i in range(2, len(displacement)):
        if time_interval[i] > 0 and time_interval[i - 1] > 0:
            dt = (time_interval[i] + time_interval[i - 1]) / 2
            if dt > 0:
                acceleration[i] = (velocity[i] - velocity[i - 1]) / dt

    return velocity, acceleration


def detect_stage_transitions(velocity, acceleration,
                             threshold1_v=0.5, threshold1_a=0.1,
                             threshold2_v=2.0, threshold2_a=0.5,
                             min_duration=6):
    """
    检测阶段转换点
    threshold1_v: 第一阶段速度阈值 (mm/h)
    threshold1_a: 第一阶段加速度阈值 (mm/h^2)
    threshold2_v: 第二阶段速度阈值 (mm/h)
    threshold2_a: 第二阶段加速度阈值 (mm/h^2)
    min_duration: 最小持续时间（数据点数）
    """
    n = len(velocity)
    stage = np.zeros(n, dtype=int)  # 0:未确定, 1:第一阶段, 2:第二阶段, 3:第三阶段

    # 第一阶段：缓慢匀速形变
    for i in range(min_duration, n):
        if (velocity[i] > threshold1_v and acceleration[i] > threshold1_a and
                np.mean(velocity[i - min_duration:i]) > threshold1_v):
            stage[i] = 1
        else:
            if i > 0 and stage[i - 1] == 1:
                stage[i] = 1
            else:
                stage[i] = 0

    # 第二阶段：加速形变（在第一阶段之后）
    first_stage_end = np.max(np.where(stage == 1)[0]) if len(np.where(stage == 1)[0]) > 0 else 0

    for i in range(max(first_stage_end + min_duration, min_duration), n):
        if (velocity[i] > threshold2_v and acceleration[i] > threshold2_a and
                np.mean(velocity[i - min_duration:i]) > threshold2_v):
            stage[i] = 2
        elif i > first_stage_end:
            if i > 0 and stage[i - 1] == 2:
                stage[i] = 2
            elif stage[i] != 2:
                stage[i] = 3 if stage[i - 1] == 3 or (i > first_stage_end and velocity[i] > threshold2_v * 0.8) else 0

    # 第三阶段：快速形变（最后的阶段）
    for i in range(len(stage)):
        if stage[i] == 0 and i > first_stage_end:
            stage[i] = 3

    return stage


def get_stage_indices(stage):
    """获取各阶段的数据索引"""
    stages = {}
    for s in [1, 2, 3]:
        indices = np.where(stage == s)[0]
        if len(indices) > 0:
            stages[s] = indices
    return stages


# ==================== 3. 特征工程 ====================

def compute_cumulative_rainfall(rainfall, window_hours=24, time_interval=1 / 6):
    """计算累积降雨量"""
    window_size = int(window_hours / time_interval)
    cum_rainfall = np.zeros(len(rainfall))
    for i in range(len(rainfall)):
        start = max(0, i - window_size + 1)
        cum_rainfall[i] = np.sum(rainfall[start:i + 1])
    return cum_rainfall


def compute_blast_decay(blast_distance, blast_charge, decay_rate=0.1, time_interval=1 / 6):
    """
    计算爆破影响衰减
    I(t) = M * C * e^{-λ(t - t_b)}
    其中 M: 单段最大药量, C: 1/距离（简化处理）
    """
    n = len(blast_distance)
    blast_impact = np.zeros(n)

    # 识别爆破事件（距离<10表示有爆破）
    blast_events = []
    for i in range(n):
        if blast_distance[i] < 9.9:  # 有爆破
            blast_events.append(i)

    # 计算每个时刻的累积爆破影响
    for i in range(n):
        total_impact = 0
        for t_b in blast_events:
            if t_b <= i:
                dt = (i - t_b) * time_interval
                # 爆破影响因子：药量 * (1/距离) * 衰减
                impact = blast_charge[t_b] * (1.0 / max(blast_distance[t_b], 0.5)) * np.exp(-decay_rate * dt)
                total_impact += impact
        blast_impact[i] = total_impact

    return blast_impact


def compute_moving_average(data, window_size=6):
    """计算移动平均特征"""
    ma = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        ma[i] = np.mean(data[start:i + 1])
    return ma


def compute_trend_features(data, window_size=6):
    """计算趋势特征（线性拟合斜率）"""
    trend = np.zeros(len(data))
    for i in range(window_size, len(data)):
        x = np.arange(window_size)
        y = data[i - window_size + 1:i + 1]
        slope = np.polyfit(x, y, 1)[0]
        trend[i] = slope
    return trend


def create_features(df, stage_indices=None):
    """创建特征矩阵"""
    n = len(df)
    time_interval = df['时间间隔_小时'].iloc[0] if len(df) > 0 else 1 / 6

    # 基础特征
    features = pd.DataFrame(index=df.index)

    # 1. 原始特征
    features['降雨量'] = df['降雨量_mm'].values
    features['孔隙水压力'] = df['孔隙水压力_kPa'].values
    features['微震事件数'] = df['微震事件数'].values
    features['爆破点距离'] = df['爆破点距离_m'].values
    features['单段最大药量'] = df['单段最大药量_kg'].values

    # 2. 累积降雨量特征（多个时间窗口）
    for window in [6, 12, 24, 48, 72]:  # 小时
        cum_rf = compute_cumulative_rainfall(df['降雨量_mm'].values, window, time_interval)
        features[f'累积降雨_{window}h'] = cum_rf

    # 3. 爆破衰减特征
    features['爆破影响'] = compute_blast_decay(
        df['爆破点距离_m'].values,
        df['单段最大药量_kg'].values,
        decay_rate=0.1,
        time_interval=time_interval
    )

    # 4. 移动平均特征
    for window in [3, 6, 12]:
        features[f'孔隙水压力_MA{window}'] = compute_moving_average(
            df['孔隙水压力_kPa'].values, window)
        features[f'微震_MA{window}'] = compute_moving_average(
            df['微震事件数'].values, window)

    # 5. 趋势特征
    features['孔隙水压力趋势'] = compute_trend_features(df['孔隙水压力_kPa'].values, 6)
    features['降雨趋势'] = compute_trend_features(df['降雨量_mm'].values, 6)

    # 6. 交互特征
    features['降雨_水压乘积'] = features['降雨量'] * features['孔隙水压力']
    features['爆破_微震乘积'] = features['爆破影响'] * (features['微震事件数'] + 1)
    features['水压变化率'] = features['孔隙水压力'].diff().fillna(0) / (features['孔隙水压力'] + 0.1)

    # 7. 滞后特征（时序）
    for lag in [1, 3, 6]:
        features[f'降雨_lag{lag}'] = features['降雨量'].shift(lag).fillna(0)
        features[f'水压_lag{lag}'] = features['孔隙水压力'].shift(lag).fillna(df['孔隙水压力_kPa'].iloc[0])
        features[f'微震_lag{lag}'] = features['微震事件数'].shift(lag).fillna(0)
        features[f'爆破_lag{lag}'] = features['爆破影响'].shift(lag).fillna(0)

    return features


# ==================== 4. 分阶段建模 ====================

class StageBasedPredictor:
    """分阶段位移预测模型"""

    def __init__(self, model_type='gbr'):
        """
        model_type: 'linear', 'ridge', 'gbr', 'rf'
        """
        self.models = {}  # {stage: model}
        self.scalers = {}  # {stage: scaler}
        self.model_type = model_type

    def _get_model(self):
        """获取指定类型的模型"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'gbr':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            return GradientBoostingRegressor(random_state=42)

    def fit(self, X, y, stages):
        """
        分阶段训练模型
        X: 特征DataFrame
        y: 目标变量（表面位移）
        stages: 阶段标签数组
        """
        unique_stages = np.unique(stages)
        unique_stages = [s for s in unique_stages if s > 0]

        for stage in unique_stages:
            stage_mask = stages == stage
            X_stage = X[stage_mask]
            y_stage = y[stage_mask]

            if len(X_stage) < 10:
                print(f"阶段{stage}数据不足({len(X_stage)}个样本)，跳过")
                continue

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_stage)

            # 训练模型
            model = self._get_model()
            model.fit(X_scaled, y_stage)

            self.models[stage] = model
            self.scalers[stage] = scaler
            print(f"阶段{stage}模型训练完成，样本数: {len(X_stage)}")

    def predict(self, X, stages):
        """分阶段预测"""
        predictions = np.zeros(len(X))

        for stage in self.models.keys():
            stage_mask = stages == stage
            if np.any(stage_mask):
                X_stage = X[stage_mask]
                scaler = self.scalers[stage]
                model = self.models[stage]

                X_scaled = scaler.transform(X_stage)
                pred = model.predict(X_scaled)
                predictions[stage_mask] = pred

        return predictions


# ==================== 5. 增量预测模型（位移变化预测）====================

class IncrementalPredictor:
    """增量预测模型 - 预测位移变化量"""

    def __init__(self, model_type='gbr'):
        self.model = None
        self.scaler = None
        self.model_type = model_type

    def _get_model(self):
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        else:
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

    def create_incremental_features(self, X, displacement):
        """创建用于预测位移增量的特征"""
        X_inc = X.copy()

        # 添加历史位移信息
        X_inc['位移_lag1'] = displacement.shift(1).fillna(displacement.iloc[0])
        X_inc['位移_lag3'] = displacement.shift(3).fillna(displacement.iloc[0])
        X_inc['位移_lag6'] = displacement.shift(6).fillna(displacement.iloc[0])

        # 位移变化率
        X_inc['位移变化率'] = X_inc['位移_lag1'].diff().fillna(0) / (X_inc['位移_lag1'] + 0.1)

        return X_inc

    def fit(self, X, displacement, stages):
        """训练增量预测模型"""
        # 计算位移增量
        delta_displacement = displacement.diff().fillna(0).values

        # 创建特征
        X_inc = self.create_incremental_features(X, displacement)

        # 只使用有阶段标签的数据
        valid_mask = stages > 0
        X_train = X_inc[valid_mask]
        y_train = delta_displacement[valid_mask]

        # 去除异常值
        valid = np.abs(y_train) < np.percentile(np.abs(y_train), 99)
        X_train = X_train[valid]
        y_train = y_train[valid]

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 训练模型
        self.model = self._get_model()
        self.model.fit(X_scaled, y_train)

        print(f"增量模型训练完成，样本数: {len(X_train)}")

    def predict(self, X, initial_displacement, stages):
        """递推预测位移"""
        n = len(X)
        predictions = np.zeros(n)
        predictions[0] = initial_displacement.iloc[0] if hasattr(initial_displacement, 'iloc') else \
        initial_displacement[0]

        current_displacement = predictions[0]
        history_displacement = [current_displacement] * 6

        for i in range(1, n):
            # 构建当前时刻的特征
            X_curr = X.iloc[i:i + 1].copy()

            # 添加历史位移
            X_curr['位移_lag1'] = history_displacement[-1]
            X_curr['位移_lag3'] = history_displacement[-3] if len(history_displacement) >= 3 else history_displacement[
                -1]
            X_curr['位移_lag6'] = history_displacement[-6] if len(history_displacement) >= 6 else history_displacement[
                -1]
            X_curr['位移变化率'] = (history_displacement[-1] - history_displacement[-2]) / (
                        history_displacement[-1] + 0.1) if len(history_displacement) >= 2 else 0

            # 预测增量
            X_scaled = self.scaler.transform(X_curr)
            delta = self.model.predict(X_scaled)[0]

            # 更新位移
            current_displacement += delta
            predictions[i] = current_displacement
            history_displacement.append(current_displacement)

        return predictions


# ==================== 6. 主程序 ====================

def main():
    print("=" * 60)
    print("边坡位移分阶段预测模型 - 问题4.1")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    train_df, test_df = load_and_preprocess_data()
    print(f"训练集数据量: {len(train_df)}")
    print(f"实验集数据量: {len(test_df)}")
    print(f"时间范围: {train_df['时间'].min()} 至 {train_df['时间'].max()}")

    # 2. 信号去噪
    print("\n[2] 信号去噪与阶段划分...")
    displacement_denoised = denoise_signal(train_df['表面位移_mm'].values, window_length=11, polyorder=3)

    # 3. 计算速度和加速度
    velocity, acceleration = calculate_velocity_and_acceleration(
        displacement_denoised,
        train_df['时间间隔_小时'].values
    )

    # 4. 检测阶段转换点
    # 根据数据特征调整阈值
    v_mean = np.mean(velocity[velocity > 0]) if np.any(velocity > 0) else 0.5
    a_mean = np.mean(acceleration[np.abs(acceleration) > 0.01]) if np.any(np.abs(acceleration) > 0.01) else 0.1

    threshold1_v = max(0.3, v_mean * 0.5)
    threshold1_a = max(0.05, a_mean * 0.3)
    threshold2_v = max(1.5, v_mean * 1.5)
    threshold2_a = max(0.3, a_mean * 0.8)

    print(f"速度阈值: v1={threshold1_v:.3f}, v2={threshold2_v:.3f}")
    print(f"加速度阈值: a1={threshold1_a:.3f}, a2={threshold2_a:.3f}")

    stages = detect_stage_transitions(
        velocity, acceleration,
        threshold1_v=threshold1_v, threshold1_a=threshold1_a,
        threshold2_v=threshold2_v, threshold2_a=threshold2_a,
        min_duration=6
    )

    # 统计各阶段样本数
    stage_counts = {1: np.sum(stages == 1), 2: np.sum(stages == 2), 3: np.sum(stages == 3)}
    print(f"阶段分布: 阶段1={stage_counts[1]}, 阶段2={stage_counts[2]}, 阶段3={stage_counts[3]}")

    # 5. 创建特征
    print("\n[3] 特征工程...")
    features = create_features(train_df)
    print(f"特征维度: {features.shape[1]}")
    print(f"特征列表: {list(features.columns)}")

    # 6. 分阶段训练模型
    print("\n[4] 训练分阶段模型...")

    # 方法1: 直接预测位移
    predictor_direct = StageBasedPredictor(model_type='gbr')
    predictor_direct.fit(features, train_df['表面位移_mm'], stages)

    # 方法2: 增量预测
    predictor_inc = IncrementalPredictor(model_type='gbr')
    predictor_inc.fit(features, train_df['表面位移_mm'], stages)

    # 7. 模型评估（使用时间序列交叉验证）
    print("\n[5] 模型评估...")

    # 划分训练集和验证集（时间顺序）
    train_size = int(len(train_df) * 0.8)
    train_idx = np.arange(train_size)
    val_idx = np.arange(train_size, len(train_df))

    # 直接在训练集上评估
    pred_direct = predictor_direct.predict(features, stages)
    pred_inc = predictor_inc.predict(features, train_df['表面位移_mm'], stages)

    # 计算评估指标
    actual = train_df['表面位移_mm'].values

    # 只评估有阶段标签的数据
    valid_mask = stages > 0

    mse_direct = mean_squared_error(actual[valid_mask], pred_direct[valid_mask])
    mae_direct = mean_absolute_error(actual[valid_mask], pred_direct[valid_mask])
    r2_direct = r2_score(actual[valid_mask], pred_direct[valid_mask])

    mse_inc = mean_squared_error(actual[valid_mask], pred_inc[valid_mask])
    mae_inc = mean_absolute_error(actual[valid_mask], pred_inc[valid_mask])
    r2_inc = r2_score(actual[valid_mask], pred_inc[valid_mask])

    print(f"\n直接预测模型:")
    print(f"  MSE: {mse_direct:.4f}, MAE: {mae_direct:.4f}, R²: {r2_direct:.4f}")
    print(f"增量预测模型:")
    print(f"  MSE: {mse_inc:.4f}, MAE: {mae_inc:.4f}, R²: {r2_inc:.4f}")

    # 8. 对实验集进行预测
    print("\n[6] 实验集预测...")

    # 为实验集创建特征
    test_features = create_features(test_df)
    test_stages = test_df['阶段标签'].values

    # 预测
    test_pred_direct = predictor_direct.predict(test_features, test_stages)
    test_pred_inc = predictor_inc.predict(test_features, test_df['表面位移_mm'], test_stages)

    # 最终预测使用增量模型（通常更准确）
    final_predictions = test_pred_inc

    # 9. 可视化结果
    print("\n[7] 生成可视化图表...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 图1: 原始位移与去噪后位移
    ax1 = axes[0, 0]
    ax1.plot(train_df['时间'], train_df['表面位移_mm'], 'b-', alpha=0.5, label='原始位移', linewidth=0.5)
    ax1.plot(train_df['时间'], displacement_denoised, 'r-', label='去噪后位移', linewidth=1)
    ax1.set_ylabel('位移 (mm)')
    ax1.set_title('位移信号去噪效果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 位移速度和阶段划分
    ax2 = axes[0, 1]
    colors = {1: 'green', 2: 'orange', 3: 'red'}
    stage_colors = [colors.get(s, 'gray') for s in stages]
    ax2.scatter(train_df['时间'], velocity, c=stage_colors, s=5, alpha=0.7)
    ax2.axhline(y=threshold1_v, color='g', linestyle='--', label=f'v1={threshold1_v:.2f}')
    ax2.axhline(y=threshold2_v, color='r', linestyle='--', label=f'v2={threshold2_v:.2f}')
    ax2.set_ylabel('速度 (mm/h)')
    ax2.set_title('位移速度与阶段划分')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 位移加速度
    ax3 = axes[1, 0]
    ax3.plot(train_df['时间'], acceleration, 'purple', linewidth=0.8)
    ax3.axhline(y=threshold1_a, color='g', linestyle='--', label=f'a1={threshold1_a:.3f}')
    ax3.axhline(y=threshold2_a, color='r', linestyle='--', label=f'a2={threshold2_a:.3f}')
    ax3.set_ylabel('加速度 (mm/h²)')
    ax3.set_title('位移加速度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 预测效果对比
    ax4 = axes[1, 1]
    ax4.plot(train_df['时间'][valid_mask], actual[valid_mask], 'b-', label='实际位移', linewidth=1)
    ax4.plot(train_df['时间'][valid_mask], pred_direct[valid_mask], 'g--', label='直接预测', linewidth=1, alpha=0.8)
    ax4.plot(train_df['时间'][valid_mask], pred_inc[valid_mask], 'r-.', label='增量预测', linewidth=1, alpha=0.8)
    ax4.set_ylabel('位移 (mm)')
    ax4.set_title('模型预测效果对比（训练集）')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 图5: 实验集预测结果
    ax5 = axes[2, 0]
    ax5.plot(test_df['时间'], test_pred_inc, 'b-', linewidth=1.5, label='预测位移')
    # 标记阶段
    for stage in [1, 2, 3]:
        mask = test_stages == stage
        if np.any(mask):
            ax5.scatter(test_df['时间'][mask], test_pred_inc[mask],
                        c=colors.get(stage, 'gray'), s=10, alpha=0.6, label=f'阶段{stage}')
    ax5.set_xlabel('时间')
    ax5.set_ylabel('位移 (mm)')
    ax5.set_title('实验集位移预测结果')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 图6: 特征重要性分析
    ax6 = axes[2, 1]
    if hasattr(predictor_inc.model, 'feature_importances_'):
        importances = predictor_inc.model.feature_importances_
        feature_names = predictor_inc.scaler.get_feature_names_out() if hasattr(predictor_inc.scaler,
                                                                                'get_feature_names_out') else list(
            features.columns)
        # 取前15个重要特征
        idx_sorted = np.argsort(importances)[::-1][:15]
        ax6.barh(range(len(idx_sorted)), importances[idx_sorted])
        ax6.set_yticks(range(len(idx_sorted)))
        ax6.set_yticklabels([feature_names[i] for i in idx_sorted], fontsize=8)
        ax6.set_xlabel('特征重要性')
        ax6.set_title('特征重要性分析')
    else:
        ax6.text(0.5, 0.5, '当前模型不支持特征重要性分析', ha='center', va='center')
        ax6.set_title('特征重要性分析')

    plt.tight_layout()
    plt.savefig('边坡位移预测结果.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 10. 输出预测结果
    print("\n[8] 预测结果输出...")

    # 创建结果DataFrame
    results = pd.DataFrame({
        '时间': test_df['时间'],
        '阶段标签': test_stages,
        '预测表面位移_mm': final_predictions,
        '直接预测_mm': test_pred_direct
    })

    # 导出结果
    results.to_csv('实验集位移预测结果.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存至: 实验集位移预测结果.csv")

    # 显示预测结果统计
    print("\n预测结果统计:")
    print(f"  最大预测位移: {final_predictions.max():.2f} mm")
    print(f"  最小预测位移: {final_predictions.min():.2f} mm")
    print(f"  平均预测位移: {final_predictions.mean():.2f} mm")

    print("\n分阶段预测统计:")
    for stage in [1, 2, 3]:
        mask = test_stages == stage
        if np.any(mask):
            stage_pred = final_predictions[mask]
            print(f"  阶段{stage}: 初始位移={stage_pred[0]:.2f}mm, "
                  f"结束位移={stage_pred[-1]:.2f}mm, "
                  f"总变化={stage_pred[-1] - stage_pred[0]:.2f}mm")

    return results


# ==================== 7. 补充：阶段边界精确检测 ====================

def refine_stage_boundaries(displacement, velocity, acceleration, stages, min_segment_length=10):
    """精细化阶段边界检测"""
    n = len(stages)
    refined_stages = stages.copy()

    # 寻找阶段转换点
    transitions = []
    for i in range(1, n):
        if stages[i] != stages[i - 1]:
            transitions.append(i)

    # 在每个转换点附近寻找最优边界
    for trans in transitions:
        if trans < min_segment_length or trans > n - min_segment_length:
            continue

        search_range = range(max(0, trans - 20), min(n, trans + 20))

        # 在搜索范围内寻找曲率最大点作为最优边界
        best_score = -np.inf
        best_idx = trans

        for idx in search_range:
            if idx < 2 or idx >= n - 2:
                continue
            # 计算曲率
            curvature = abs(acceleration[idx])
            # 加上速度变化
            score = curvature + 0.1 * abs(velocity[idx])

            if score > best_score:
                best_score = score
                best_idx = idx

        # 更新阶段
        refined_stages[best_idx:] = stages[best_idx:]

    return refined_stages


# ==================== 8. 运行主程序 ====================

if __name__ == "__main__":
    results = main()

    print("\n" + "=" * 60)
    print("模型构建完成！")
    print("=" * 60)