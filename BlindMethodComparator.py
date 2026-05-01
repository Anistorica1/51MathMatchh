from BlindEvaluator import *

class BlindMethodComparator:
    """无参考数据的去噪方法比较器"""

    def __init__(self, noisy_signal):
        """
        noisy_signal: 含噪信号（原始数据）
        """
        self.noisy = np.array(noisy_signal).flatten()
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
            # 如果传入的是函数，则应用到数据上
            if callable(method_data):
                processed = method_data(self.noisy.copy())
            else:
                processed = method_data

            # 确保长度一致
            processed = processed[:len(self.noisy)]

            # 计算综合评分
            total_score, sub_scores = self.evaluator.comprehensive_blind_score(
                self.noisy, processed
            )

            # 额外指标
            noise_orig = self.evaluator.noise_energy_estimate(self.noisy)
            noise_proc = self.evaluator.noise_energy_estimate(processed)
            smooth_orig = self.evaluator.smoothness_metric(self.noisy)
            smooth_proc = self.evaluator.smoothness_metric(processed)

            results.append({
                '方法': method_name,
                '综合得分': total_score,
                '噪声减少(%)': (1 - noise_proc / (noise_orig + 1e-10)) * 100,
                '粗糙度降低(%)': (1 - smooth_proc['roughness'] / (smooth_orig['roughness'] + 1e-10)) * 100,
                '熵减少(%)': (1 - self.evaluator.signal_entropy(processed) /
                              (self.evaluator.signal_entropy(self.noisy) + 1e-10)) * 100,
                **{k: v for k, v in sub_scores.items()}
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('综合得分', ascending=False)

        return results_df

    def auto_select_best_method(self, methods_dict):
        """自动选择最佳方法"""
        comparison = self.compare_methods(methods_dict)
        best_method = comparison.iloc[0]['方法']
        best_score = comparison.iloc[0]['综合得分']

        return best_method, best_score, comparison