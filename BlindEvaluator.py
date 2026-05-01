import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')


class BlindEvaluator:
    """无参考数据的盲评估器 - 修复NaN错误版本"""

    def __init__(self):
        pass

    @staticmethod
    def clean_data(*arrays):
        """清理数据，移除NaN和inf"""
        cleaned = []
        for arr in arrays:
            arr = np.array(arr).flatten()
            # 替换NaN和inf
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned.append(arr)

        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    @staticmethod
    def smoothness_metric(signal):
        """
        平滑度指标 - 值越小越平滑（去噪效果越好）
        """
        signal = BlindEvaluator.clean_data(signal)

        # 一阶差分（变化率）
        first_diff = np.diff(signal)
        # 二阶差分（曲率）
        second_diff = np.diff(first_diff)

        # 多种平滑度测度
        metrics = {
            'mean_absolute_diff': np.mean(np.abs(first_diff)),
            'std_of_diff': np.std(first_diff),
            'roughness': np.mean(first_diff ** 2),
            'curvature': np.mean(np.abs(second_diff)),
            'total_variation': np.sum(np.abs(first_diff))
        }

        return metrics

    @staticmethod
    def noise_energy_estimate(signal, method='diff'):
        """
        估计信号中的噪声能量
        """
        signal = BlindEvaluator.clean_data(signal)

        if len(signal) < 3:
            return 0.0

        if method == 'diff':
            # 差分法：高频分量被视为噪声
            diff_signal = np.diff(signal)
            noise_std = np.std(diff_signal) / np.sqrt(2)

        elif method == 'mad':
            # 中位数绝对偏差法
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            noise_std = mad / 0.6745

        elif method == 'wavelet':
            # 小波法（需要pywt库）
            try:
                import pywt
                coeffs = pywt.wavedec(signal, 'db4', level=min(3, len(signal) // 10))
                # 使用最高频细节系数估计噪声
                noise_coeff = coeffs[-1]
                noise_std = np.median(np.abs(noise_coeff)) / 0.6745
            except:
                noise_std = np.std(np.diff(signal)) / np.sqrt(2)

        elif method == 'fft':
            # FFT方法：高频能量占比
            if len(signal) > 10:
                fft_vals = fft(signal)
                freqs = fftfreq(len(signal))
                high_freq_mask = np.abs(freqs) > 0.5 * np.max(freqs)
                high_freq_energy = np.mean(np.abs(fft_vals[high_freq_mask]) ** 2)
                total_energy = np.mean(np.abs(fft_vals) ** 2)
                noise_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
                noise_std = np.std(signal) * noise_ratio
            else:
                noise_std = np.std(np.diff(signal)) / np.sqrt(2)

        return max(0, noise_std)

    @staticmethod
    def periodicity_preservation(original, processed):
        """
        周期性保持度 - 值越接近1表示保持得越好
        """
        original = BlindEvaluator.clean_data(original)
        processed = BlindEvaluator.clean_data(processed)

        if len(original) < 10:
            return 0.5

        # 自相关函数
        try:
            auto_orig = np.correlate(original - np.mean(original),
                                     original - np.mean(original), mode='full')
            auto_proc = np.correlate(processed - np.mean(processed),
                                     processed - np.mean(processed), mode='full')

            # 归一化
            auto_orig_norm = auto_orig / (auto_orig[len(auto_orig) // 2] + 1e-10)
            auto_proc_norm = auto_proc / (auto_proc[len(auto_proc) // 2] + 1e-10)

            # 计算相关性
            correlation = np.corrcoef(auto_orig_norm, auto_proc_norm)[0, 1]

            return max(0, min(1, correlation))
        except:
            return 0.5

    @staticmethod
    def spectral_distortion(original, processed, fs=1.0):
        """
        频谱失真度 - 值越小表示失真越小
        """
        original = BlindEvaluator.clean_data(original)
        processed = BlindEvaluator.clean_data(processed)

        if len(original) < 10:
            return 1.0

        try:
            # 计算功率谱密度
            nperseg = min(256, len(original) // 4)
            if nperseg < 3:
                return 1.0

            f_orig, Pxx_orig = signal.welch(original, fs=fs, nperseg=nperseg)
            f_proc, Pxx_proc = signal.welch(processed, fs=fs, nperseg=nperseg)

            # 确保长度一致
            min_len = min(len(Pxx_orig), len(Pxx_proc))
            Pxx_orig = Pxx_orig[:min_len]
            Pxx_proc = Pxx_proc[:min_len]

            # 频谱失真度
            distortion = np.mean(np.abs(np.log10(Pxx_orig + 1e-10) - np.log10(Pxx_proc + 1e-10)) ** 2)

            return max(0, min(10, distortion))
        except:
            return 1.0

    @staticmethod
    def signal_entropy(signal):
        """
        信号熵 - 去噪后熵应该降低
        """
        signal = BlindEvaluator.clean_data(signal)

        if len(signal) < 10:
            return 0.0

        # 归一化
        signal_min = np.min(signal)
        signal_max = np.max(signal)

        if signal_max == signal_min:
            return 0.0

        signal_norm = (signal - signal_min) / (signal_max - signal_min + 1e-10)

        # 移除可能的inf值
        signal_norm = np.clip(signal_norm, 0, 1)

        # 计算直方图
        try:
            hist, _ = np.histogram(signal_norm, bins=min(50, len(signal) // 5), density=True)
            hist = hist[hist > 0]

            # 计算熵
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            return max(0, min(10, entropy))
        except:
            return 0.0

    @staticmethod
    def local_variance_ratio(original, processed, window_size=5):
        """
        局部方差比 - 去噪后局部方差应该降低
        """
        original = BlindEvaluator.clean_data(original)
        processed = BlindEvaluator.clean_data(processed)

        if len(original) < window_size:
            return 1.0

        # 计算局部方差
        def local_variance(x, window):
            kernel = np.ones(window) / window
            mean = np.convolve(x, kernel, mode='same')
            var = np.convolve((x - mean) ** 2, kernel, mode='same')
            return var

        var_orig = local_variance(original, window_size)
        var_proc = local_variance(processed, window_size)

        # 方差比（避免除零）
        ratio = np.mean(var_proc) / (np.mean(var_orig) + 1e-10)

        return max(0, min(10, ratio))

    @staticmethod
    def edge_preservation_index(original, processed):
        """
        边缘保持指数 - 值越接近1表示边缘保持越好
        """
        original = BlindEvaluator.clean_data(original)
        processed = BlindEvaluator.clean_data(processed)

        if len(original) < 5:
            return 0.5

        # 计算梯度
        grad_orig = np.gradient(original)
        grad_proc = np.gradient(processed)

        # 计算相关系数
        try:
            correlation = np.corrcoef(grad_orig, grad_proc)[0, 1]
            return max(0, min(1, correlation))
        except:
            return 0.5

    @staticmethod
    def comprehensive_blind_score(original, processed):
        """
        综合盲评分数（0-1之间，越高越好）
        """
        # 清理数据
        original = BlindEvaluator.clean_data(original)
        processed = BlindEvaluator.clean_data(processed)

        # 确保长度一致
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        if min_len < 10:
            return 0.0, {}

        # 计算各个指标
        smooth_orig = BlindEvaluator.smoothness_metric(original)
        smooth_proc = BlindEvaluator.smoothness_metric(processed)

        # 平滑度改善（去噪后应该更平滑）
        smoothness_improvement = 1 - (smooth_proc['roughness'] / (smooth_orig['roughness'] + 1e-10))
        smoothness_improvement = np.clip(smoothness_improvement, 0, 1)

        # 噪声减少
        noise_orig = BlindEvaluator.noise_energy_estimate(original)
        noise_proc = BlindEvaluator.noise_energy_estimate(processed)
        noise_reduction = 1 - (noise_proc / (noise_orig + 1e-10))
        noise_reduction = np.clip(noise_reduction, 0, 1)

        # 周期性保持
        periodicity = BlindEvaluator.periodicity_preservation(original, processed)
        periodicity = np.clip(periodicity, 0, 1)

        # 边缘保持
        edge_preserve = BlindEvaluator.edge_preservation_index(original, processed)
        edge_preserve = np.clip(edge_preserve, 0, 1)

        # 信息熵变化（去噪后熵应降低）
        entropy_orig = BlindEvaluator.signal_entropy(original)
        entropy_proc = BlindEvaluator.signal_entropy(processed)
        entropy_reduction = 1 - (entropy_proc / (entropy_orig + 1e-10))
        entropy_reduction = np.clip(entropy_reduction, 0, 1)

        # 频谱失真度（转换为分数）
        spec_dist = BlindEvaluator.spectral_distortion(original, processed)
        spectral_score = 1 / (1 + spec_dist)
        spectral_score = np.clip(spectral_score, 0, 1)

        # 局部方差比（应该接近1）
        var_ratio = BlindEvaluator.local_variance_ratio(original, processed)
        var_score = 1 / (1 + np.abs(1 - var_ratio))
        var_score = np.clip(var_score, 0, 1)

        # 综合评分
        scores = {
            '平滑度改善': smoothness_improvement,
            '噪声减少': noise_reduction,
            '周期性保持': periodicity,
            '边缘保持': edge_preserve,
            '熵减少': entropy_reduction,
            '频谱保真': spectral_score,
            '方差一致性': var_score
        }

        # 加权平均
        weights = {
            '平滑度改善': 0.15,
            '噪声减少': 0.25,  # 噪声减少最重要
            '周期性保持': 0.10,
            '边缘保持': 0.20,  # 边缘保持次重要
            '熵减少': 0.10,
            '频谱保真': 0.10,
            '方差一致性': 0.10
        }

        total_score = sum(scores[k] * weights[k] for k in scores if k in weights)
        total_score = np.clip(total_score, 0, 1)

        return total_score, scores


class BlindMethodComparator:
    """无参考数据的去噪方法比较器"""

    def __init__(self, noisy_signal):
        """
        noisy_signal: 含噪信号（原始数据）
        """
        self.noisy = BlindEvaluator.clean_data(noisy_signal)
        self.evaluator = BlindEvaluator()

    def compare_methods(self, methods_dict):
        """
        比较不同的去噪/插值方法

        methods_dict: {
            '方法名': 处理函数或处理后的数据
        }
        """
        results = []

        for method_name, method_data in methods_dict.items():
            try:
                # 如果传入的是函数，则应用到数据上
                if callable(method_data):
                    processed = method_data(self.noisy.copy())
                else:
                    processed = method_data

                # 确保长度一致并清理数据
                processed = BlindEvaluator.clean_data(processed)
                min_len = min(len(self.noisy), len(processed))
                noisy_clean = self.noisy[:min_len]
                processed_clean = processed[:min_len]

                # 计算综合评分
                total_score, sub_scores = self.evaluator.comprehensive_blind_score(
                    noisy_clean, processed_clean
                )

                # 额外指标
                noise_orig = self.evaluator.noise_energy_estimate(noisy_clean)
                noise_proc = self.evaluator.noise_energy_estimate(processed_clean)
                smooth_orig = self.evaluator.smoothness_metric(noisy_clean)
                smooth_proc = self.evaluator.smoothness_metric(processed_clean)

                results.append({
                    '方法': method_name,
                    '综合得分': total_score,
                    '噪声减少(%)': (1 - noise_proc / (noise_orig + 1e-10)) * 100,
                    '粗糙度降低(%)': (1 - smooth_proc['roughness'] / (smooth_orig['roughness'] + 1e-10)) * 100,
                    '熵减少(%)': (1 - self.evaluator.signal_entropy(processed_clean) /
                                  (self.evaluator.signal_entropy(noisy_clean) + 1e-10)) * 100,
                    **{k: v for k, v in sub_scores.items()}
                })
            except Exception as e:
                print(f"警告：方法 {method_name} 评估失败: {e}")
                results.append({
                    '方法': method_name,
                    '综合得分': 0,
                    '噪声减少(%)': 0,
                    '粗糙度降低(%)': 0,
                    '熵减少(%)': 0
                })

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('综合得分', ascending=False)

        return results_df

    def auto_select_best_method(self, methods_dict):
        """自动选择最佳方法"""
        comparison = self.compare_methods(methods_dict)
        if len(comparison) > 0:
            best_method = comparison.iloc[0]['方法']
            best_score = comparison.iloc[0]['综合得分']
            return best_method, best_score, comparison
        else:
            return None, 0, pd.DataFrame()


class BlindInterpolationEvaluator:
    """无参考数据的插值评估器"""

    def __init__(self, data_with_missing):
        """
        data_with_missing: 包含缺失值的数据
        """
        self.data = np.array(data_with_missing).flatten()
        self.missing_mask = np.isnan(self.data)

        # 清理非缺失值
        self.data_clean = BlindEvaluator.clean_data(self.data)

    @staticmethod
    def continuity_score(data_interpolated):
        """
        连续性评分 - 插值后应该平滑连续
        """
        data_interpolated = BlindEvaluator.clean_data(data_interpolated)

        if len(data_interpolated) < 3:
            return 0.5

        # 计算一阶导数和二阶导数
        first_diff = np.diff(data_interpolated)
        second_diff = np.diff(first_diff)

        # 连续性惩罚：大的跳跃和不平滑
        discontinuity = np.mean(np.abs(first_diff))
        curvature = np.mean(np.abs(second_diff))

        # 转换为分数（值越小越好）
        score = 1 / (1 + discontinuity + curvature)

        return np.clip(score, 0, 1)

    @staticmethod
    def monotonicity_preservation(original_missing, interpolated):
        """
        单调性保持度 - 插值不应破坏原始趋势
        """
        original_missing = np.array(original_missing).flatten()
        interpolated = BlindEvaluator.clean_data(interpolated)

        # 只考虑原始有效数据点
        valid_mask = ~np.isnan(original_missing)
        if valid_mask.sum() < 2:
            return 0.5

        orig_valid = original_missing[valid_mask]
        interp_at_valid = interpolated[valid_mask]

        # 计算单调性一致性
        if len(orig_valid) < 2 or len(interp_at_valid) < 2:
            return 0.5

        orig_trend = np.sign(np.diff(orig_valid))
        interp_trend = np.sign(np.diff(interp_at_valid))

        consistency = np.mean(orig_trend == interp_trend)

        return np.clip(consistency, 0, 1)

    @staticmethod
    def distribution_preservation(original_missing, interpolated):
        """
        分布保持度 - 插值数据的分布应与原始数据相似
        """
        original_missing = np.array(original_missing).flatten()
        interpolated = BlindEvaluator.clean_data(interpolated)

        valid_mask = ~np.isnan(original_missing)
        if valid_mask.sum() < 10:
            return 0.5

        orig_valid = original_missing[valid_mask]

        # Kolmogorov-Smirnov检验
        try:
            ks_stat, _ = stats.ks_2samp(orig_valid, interpolated)
            score = 1 - ks_stat
            return np.clip(score, 0, 1)
        except:
            return 0.5

    def evaluate_interpolation(self, interpolated_data):
        """
        评估插值质量
        """
        interpolated = BlindEvaluator.clean_data(interpolated_data)

        if len(interpolated) != len(self.data):
            # 调整长度
            min_len = min(len(interpolated), len(self.data))
            interpolated = interpolated[:min_len]
            data_trimmed = self.data[:min_len]
            missing_mask_trimmed = self.missing_mask[:min_len]
        else:
            data_trimmed = self.data
            missing_mask_trimmed = self.missing_mask

        # 计算各项指标
        continuity = self.continuity_score(interpolated)
        monotonicity = self.monotonicity_preservation(data_trimmed, interpolated)
        distribution = self.distribution_preservation(data_trimmed, interpolated)

        # 插值点是否在合理范围内
        interpolated_points = interpolated[missing_mask_trimmed]
        valid_data = data_trimmed[~missing_mask_trimmed]

        if len(valid_data) > 0 and len(interpolated_points) > 0:
            q1, q3 = np.percentile(valid_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_in_interp = ((interpolated_points < lower_bound) |
                                  (interpolated_points > upper_bound)).sum()
            outlier_ratio = outliers_in_interp / (len(interpolated_points) + 1e-10)
            outlier_score = 1 - outlier_ratio
        else:
            outlier_score = 0.5

        # 综合评分
        total_score = (continuity * 0.3 +
                       monotonicity * 0.3 +
                       distribution * 0.2 +
                       outlier_score * 0.2)

        return {
            '连续性得分': continuity,
            '单调性保持': monotonicity,
            '分布相似度': distribution,
            '异常值得分': outlier_score,
            '综合得分': total_score
        }

    def compare_interpolation_methods(self, methods_dict):
        """
        比较不同插值方法
        """
        results = []

        for method_name, method_func in methods_dict.items():
            try:
                if callable(method_func):
                    interpolated = method_func(self.data.copy())
                else:
                    interpolated = method_func

                evaluation = self.evaluate_interpolation(interpolated)
                evaluation['方法'] = method_name
                results.append(evaluation)
            except Exception as e:
                print(f"警告：插值方法 {method_name} 评估失败: {e}")

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('综合得分', ascending=False)

        return results_df


# 主程序示例
if __name__ == "__main__":
    # 创建示例数据（模拟实际情况：只有含噪带缺失值的数据）
    np.random.seed(42)
    n = 300
    t = np.linspace(0, 10, n)

    # 真实信号（实际中未知）
    true_signal = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * np.sin(5 * t)

    # 只有我们有的数据：含噪且带缺失值
    noise = np.random.normal(0, 0.2, n)
    observed_data = true_signal + noise

    # 添加缺失值
    missing_idx = np.random.choice(n, 40, replace=False)
    observed_data[missing_idx] = np.nan

    print("=== 盲评估示例 ===")
    print(f"原始数据长度: {len(observed_data)}")
    print(f"缺失值数量: {np.isnan(observed_data).sum()}")
    print(f"有效数据量: {observed_data[~np.isnan(observed_data)].shape[0]}")


    # 1. 定义不同的处理方法和插值方法
    def method_median_filter(data):
        """中值滤波"""
        from scipy.ndimage import median_filter
        # 先插值填充NaN
        data_filled = pd.Series(data).interpolate().ffill().bfill().values
        return median_filter(data_filled, size=5)


    def method_moving_average(data):
        """移动平均"""
        data_filled = pd.Series(data).interpolate().ffill().bfill()
        return data_filled.rolling(5, center=True).mean().fillna(method='bfill').fillna(method='ffill').values


    def method_exponential_smoothing(data):
        """指数平滑"""
        data_filled = pd.Series(data).interpolate().ffill().bfill()
        return data_filled.ewm(span=5).mean().values


    # 2. 比较去噪方法
    print("\n" + "=" * 60)
    print("1. 不同去噪方法比较")
    print("=" * 60)

    comparator = BlindMethodComparator(observed_data)

    methods = {
        '中值滤波': method_median_filter,
        '移动平均': method_moving_average,
        '指数平滑': method_exponential_smoothing,
    }

    comparison_results = comparator.compare_methods(methods)
    print(comparison_results.to_string(index=False))

    # 3. 自动选择最佳方法
    best_method, best_score, all_results = comparator.auto_select_best_method(methods)
    if best_method:
        print(f"\n最佳去噪方法: {best_method}")
        print(f"综合得分: {best_score:.4f}")

    # 4. 比较插值方法
    print("\n" + "=" * 60)
    print("2. 不同插值方法比较")
    print("=" * 60)

    interp_evaluator = BlindInterpolationEvaluator(observed_data)


    def linear_interpolation(data):
        return pd.Series(data).interpolate(method='linear').values


    def quadratic_interpolation(data):
        return pd.Series(data).interpolate(method='quadratic').values


    def cubic_interpolation(data):
        return pd.Series(data).interpolate(method='cubic').values


    interp_methods = {
        '线性插值': linear_interpolation,
        '二次插值': quadratic_interpolation,
        '三次插值': cubic_interpolation,
    }

    interp_comparison = interp_evaluator.compare_interpolation_methods(interp_methods)
    print(interp_comparison.to_string(index=False))