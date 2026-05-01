import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False


class BlindInterpolationEvaluator:
    """无参考数据的插值评估器"""

    def __init__(self, data_with_missing):
        """
        data_with_missing: 包含缺失值的数据
        """
        self.data = np.array(data_with_missing).flatten()
        self.missing_mask = np.isnan(self.data)

    @staticmethod
    def continuity_score(data_interpolated):
        """
        连续性评分 - 插值后应该平滑连续
        """
        # 计算一阶导数和二阶导数
        first_diff = np.diff(data_interpolated)
        second_diff = np.diff(first_diff)

        # 连续性惩罚：大的跳跃和不平滑
        discontinuity = np.mean(np.abs(first_diff))
        curvature = np.mean(np.abs(second_diff))

        # 转换为分数（值越小越好）
        score = 1 / (1 + discontinuity + curvature)

        return score

    @staticmethod
    def monotonicity_preservation(original_missing, interpolated):
        """
        单调性保持度 - 插值不应破坏原始趋势
        """
        # 只考虑原始有效数据点
        valid_mask = ~np.isnan(original_missing)
        if valid_mask.sum() < 2:
            return 0.5

        orig_valid = original_missing[valid_mask]
        interp_at_valid = interpolated[valid_mask]

        # 计算单调性一致性
        orig_trend = np.sign(np.diff(orig_valid))
        interp_trend = np.sign(np.diff(interp_at_valid))

        consistency = np.mean(orig_trend == interp_trend)

        return consistency

    @staticmethod
    def distribution_preservation(original_missing, interpolated):
        """
        分布保持度 - 插值数据的分布应与原始数据相似
        """
        valid_mask = ~np.isnan(original_missing)
        if valid_mask.sum() < 10:
            return 0.5

        orig_valid = original_missing[valid_mask]

        # Kolmogorov-Smirnov检验
        ks_stat, _ = stats.ks_2samp(orig_valid, interpolated)

        # 转换为分数
        score = 1 - ks_stat

        return max(0, min(1, score))

    def evaluate_interpolation(self, interpolated_data):
        """
        评估插值质量
        """
        interpolated = np.array(interpolated_data).flatten()

        # 只评估插值点位置的合理性
        interpolated_points = interpolated[self.missing_mask]

        # 计算各项指标
        continuity = self.continuity_score(interpolated)
        monotonicity = self.monotonicity_preservation(self.data, interpolated)
        distribution = self.distribution_preservation(self.data, interpolated)

        # 插值点是否在合理范围内
        if (~self.missing_mask).sum() > 0:
            valid_data = self.data[~self.missing_mask]
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
            if callable(method_func):
                interpolated = method_func(self.data.copy())
            else:
                interpolated = method_func

            evaluation = self.evaluate_interpolation(interpolated)
            evaluation['方法'] = method_name
            results.append(evaluation)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('综合得分', ascending=False)

        return results_df