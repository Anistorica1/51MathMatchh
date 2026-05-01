
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union
from pandas.plotting import scatter_matrix
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False
def spearman_analysis(
        data: pd.DataFrame,
        target: str = None,
        variables: list = None,
        p_threshold: float = 0.05,
        plot_heatmap: bool = True,
        plot_clustermap: bool = False,
        verbose: bool = True
):
    """
    斯皮尔曼相关系数分析函数（适用于非正态分布数据）

    Parameters
    ----------
    data : pd.DataFrame
        输入数据，每列为一个变量
    target : str, optional
        目标变量，如果指定则只计算其他变量与target的相关性
    variables : list, optional
        要分析的变量列表，None表示所有数值型变量
    p_threshold : float, default=0.05
        显著性阈值
    plot_heatmap : bool, default=True
        是否绘制热力图
    plot_clustermap : bool, default=False
        是否绘制聚类图（数据量大时可能较慢）
    verbose : bool, default=True
        是否打印详细信息

    Returns
    -------
    dict or pd.DataFrame
        包含相关系数、p值、显著性标记等结果
    """

    # 1. 数据准备
    if variables is None:
        # 选择数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("数据中没有数值型变量！")
        data_analysis = data[numeric_cols].copy()
    else:
        data_analysis = data[variables].copy()

    # 处理缺失值
    if data_analysis.isnull().sum().sum() > 0:
        if verbose:
            print(f"⚠️ 发现缺失值，将删除包含缺失值的行")
        data_analysis = data_analysis.dropna()

    if verbose:
        print("=" * 70)
        print("斯皮尔曼相关系数分析")
        print("=" * 70)
        print(f"样本数: {len(data_analysis)}")
        print(f"变量数: {len(data_analysis.columns)}")
        print(f"变量列表: {data_analysis.columns.tolist()}")
        print("=" * 70)

    # 2. 如果是针对目标变量的分析
    if target is not None:
        if target not in data_analysis.columns:
            raise ValueError(f"目标变量 '{target}' 不在数据中！可用变量: {data_analysis.columns.tolist()}")

        results = []
        target_data = data_analysis[target]

        for col in data_analysis.columns:
            if col == target:
                continue

            # 计算斯皮尔曼相关系数
            corr, p_value = spearmanr(target_data, data_analysis[col])

            # 相关性强度判断
            strength = ''
            if abs(corr) >= 0.8:
                strength = '极强相关'
            elif abs(corr) >= 0.6:
                strength = '强相关'
            elif abs(corr) >= 0.4:
                strength = '中等相关'
            elif abs(corr) >= 0.2:
                strength = '弱相关'
            else:
                strength = '极弱/不相关'

            # 方向判断
            direction = '正相关' if corr > 0 else '负相关'

            # 显著性标记
            sig_stars = ''
            if p_value < 0.001:
                sig_stars = '***'
            elif p_value < 0.01:
                sig_stars = '**'
            elif p_value < 0.05:
                sig_stars = '*'

            results.append({
                '变量': col,
                '相关系数 ρ': corr,
                'p值': p_value,
                '显著性': sig_stars,
                '相关性强度': strength,
                '相关方向': direction,
                '显著性(α=0.05)': p_value < p_threshold
            })

        # 排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('相关系数 ρ', key=abs, ascending=False)

        if verbose:
            print(f"\n目标变量: {target}")
            print("-" * 70)
            print(results_df.to_string(index=False))
            print("-" * 70)

        # 可视化
        if plot_heatmap:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 相关系数条形图
            ax1 = axes[0, 0]
            corr_values = results_df['相关系数 ρ'].values
            colors = ['red' if x < 0 else 'green' for x in corr_values]
            bars = ax1.barh(range(len(results_df)), corr_values, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(results_df)))
            ax1.set_yticklabels(results_df['变量'])
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel(f'斯皮尔曼相关系数 ρ')
            ax1.set_title(f'与 "{target}" 的相关性排序')
            ax1.grid(axis='x', alpha=0.3)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, corr_values)):
                ax1.text(val + (0.02 if val >= 0 else -0.08),
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center')

            # 2. 显著性散点图
            ax2 = axes[0, 1]
            colors_sig = ['red' if p < p_threshold else 'gray' for p in results_df['p值']]
            ax2.scatter(results_df['相关系数 ρ'], -np.log10(results_df['p值']),
                        c=colors_sig, s=50, alpha=0.7)
            ax2.axhline(y=-np.log10(p_threshold), color='red', linestyle='--',
                        label=f'p={p_threshold}显著性阈值')
            ax2.set_xlabel('斯皮尔曼相关系数 ρ')
            ax2.set_ylabel('-log10(p值)')
            ax2.set_title('相关性强度与显著性')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 热力图（只显示与target的相关）
            ax3 = axes[1, 0]
            # 构建相关性矩阵
            corr_matrix_target = pd.DataFrame({
                col: [results_df[results_df['变量'] == col]['相关系数 ρ'].values[0]]
                for col in results_df['变量']
            }, index=[target])
            sns.heatmap(corr_matrix_target, annot=True, fmt='.3f', cmap='RdBu_r',
                        center=0, vmin=-1, vmax=1, ax=ax3, cbar_kws={"shrink": 0.8})
            ax3.set_title(f'各变量与 {target} 的相关系数热力图')

            # 4. p值分布
            ax4 = axes[1, 1]
            p_values = results_df['p值'].values
            sig_colors = ['red' if p < p_threshold else 'gray' for p in p_values]
            ax4.barh(range(len(results_df)), p_values, color=sig_colors, alpha=0.7)
            ax4.axvline(x=p_threshold, color='red', linestyle='--', label=f'p={p_threshold}')
            ax4.set_yticks(range(len(results_df)))
            ax4.set_yticklabels(results_df['变量'])
            ax4.set_xlabel('p值')
            ax4.set_title('显著性水平分布')
            ax4.legend()
            ax4.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.show()

            # 额外绘制散点图（显示相关性最强的3个）
            top_vars = results_df.head(3)['变量'].values
            fig, axes = plt.subplots(1, min(3, len(top_vars)), figsize=(15, 4))
            if len(top_vars) == 1:
                axes = [axes]

            for idx, var in enumerate(top_vars):
                corr_val = results_df[results_df['变量'] == var]['相关系数 ρ'].values[0]
                axes[idx].scatter(data_analysis[var], data_analysis[target], alpha=0.5, s=20)

                # 添加趋势线（使用秩次）
                x_rank = rankdata(data_analysis[var])
                y_rank = rankdata(data_analysis[target])
                z = np.polyfit(x_rank, y_rank, 1)
                p = np.poly1d(z)
                axes[idx].plot(np.sort(x_rank), p(np.sort(x_rank)), "r-", linewidth=2)

                axes[idx].set_xlabel(var)
                axes[idx].set_ylabel(target)
                axes[idx].set_title(f'{var} vs {target}\nρ = {corr_val:.3f}')
                axes[idx].grid(True, alpha=0.3)

            plt.suptitle(f'最强相关性散点图（基于秩次）', fontsize=12)
            plt.tight_layout()
            plt.show()

        return results_df

    # 3. 完整矩阵分析
    else:
        n_vars = len(data_analysis.columns)
        var_names = data_analysis.columns.tolist()

        # 计算相关系数矩阵和p值矩阵
        corr_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                   index=var_names, columns=var_names)
        p_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                index=var_names, columns=var_names)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix.iloc[i, j] = 1.0
                    p_matrix.iloc[i, j] = 0.0
                elif i < j:  # 只计算一次，对称填充
                    corr, p_val = spearmanr(data_analysis.iloc[:, i],
                                            data_analysis.iloc[:, j])
                    corr_matrix.iloc[i, j] = corr
                    corr_matrix.iloc[j, i] = corr
                    p_matrix.iloc[i, j] = p_val
                    p_matrix.iloc[j, i] = p_val

        if verbose:
            print("\n斯皮尔曼相关系数矩阵:")
            print("-" * 70)
            print(corr_matrix.round(4))

            print("\n\np值矩阵:")
            print("-" * 70)
            print(p_matrix.round(6))

            # 找出最强相关性
            print("\n\n" + "=" * 70)
            print("最强相关性 Top 10")
            print("=" * 70)

            corr_pairs = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    corr_pairs.append({
                        '变量1': var_names[i],
                        '变量2': var_names[j],
                        '相关系数 ρ': corr_matrix.iloc[i, j],
                        'p值': p_matrix.iloc[i, j],
                        '显著性': '***' if p_matrix.iloc[i, j] < 0.001 else
                        '**' if p_matrix.iloc[i, j] < 0.01 else
                        '*' if p_matrix.iloc[i, j] < 0.05 else ''
                    })

            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df = corr_pairs_df.sort_values('相关系数 ρ', key=abs, ascending=False)
            print(corr_pairs_df.head(10).to_string(index=False))

        # 可视化
        if plot_heatmap:
            if plot_clustermap:
                # 情况1：两个图分别显示（推荐）
                # 普通热力图
                fig1, ax1 = plt.subplots()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                            square=True, linewidths=0.5, ax=ax1,
                            cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})
                ax1.set_title('斯皮尔曼相关系数矩阵热力图\n(适用于非正态数据)', fontsize=12)
                plt.tight_layout()
                plt.show()

                # 聚类图（单独显示）
                g = sns.clustermap(corr_matrix, annot=True, fmt='.2f',
                                   cmap='RdBu_r', center=0, figsize=(10, 8))
                g.fig.suptitle('相关系数聚类图', y=1.02)
                plt.show()

            else:
                # 情况2：只显示普通热力图
                fig, ax = plt.subplots(figsize=(14, 6))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                            square=True, linewidths=0.5, ax=ax,
                            cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})
                ax.set_title('斯皮尔曼相关系数矩阵热力图\n(适用于非正态数据)', fontsize=12)
                plt.tight_layout()
                plt.show()

        return {
            'correlation_matrix': corr_matrix,
            'p_value_matrix': p_matrix,
            'method': 'spearman'
        }
def jb_test_analysis(data, column=None, alpha=0.05):
    """
    Jarque-Bera正态性检验

    Parameters:
    -----------
    data : array-like or DataFrame
        待检验数据
    column : str, optional
        如果data是DataFrame，指定要检验的列名
    alpha : float, default=0.05
        显著性水平

    Returns:
    --------
    dict : 包含检验统计量、p值、结论等
    """

    # 提取数据
    if isinstance(data, pd.DataFrame):
        if column:
            x = data[column].dropna().values
        else:
            # 检验所有数值列
            results = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                results[col] = jb_test_analysis(data[col], alpha=alpha)
            return results
    else:
        x = np.array(data).flatten()
        x = x[~np.isnan(x)]  # 删除缺失值

    # 计算基本统计量
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    skewness = stats.skew(x)
    kurtosis = stats.kurtosis(x, fisher=True)  # Fisher定义：正态分布峰度=0

    # Jarque-Bera检验
    jb_stat, p_value = stats.jarque_bera(x)

    # 判断结果
    is_normal = p_value > alpha
    significance_stars = ''
    if p_value < 0.001:
        significance_stars = '***'
    elif p_value < 0.01:
        significance_stars = '**'
    elif p_value < 0.05:
        significance_stars = '*'

    # 输出结果
    print("=" * 60)
    print(f"Jarque-Bera 正态性检验")
    print("=" * 60)
    print(f"样本量 (n): {n}")
    print(f"均值 (Mean): {mean:.4f}")
    print(f"标准差 (Std): {std:.4f}")
    print(f"偏度 (Skewness): {skewness:.4f}")
    print(f"峰度 (Kurtosis): {kurtosis:.4f}")
    print(f"\nJB统计量: {jb_stat:.4f}")
    print(f"p值: {p_value:.6f} {significance_stars}")
    print(f"\n结论: {'✅ 接受正态分布' if is_normal else '❌ 拒绝正态分布'}")

    if abs(skewness) > 1:
        print(f"⚠️  偏度绝对值 > 1，分布明显不对称")
    if abs(kurtosis) > 3:
        print(f"⚠️  峰度绝对值 > 3，分布明显过于尖峭或平坦")

    return {
        'variable': column if column else 'data',
        'sample_size': n,
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jb_statistic': jb_stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'significance': significance_stars
    }
def plot_normality_check(data, column=None, figsize=(12, 4)):
    """可视化正态性检验"""

    if isinstance(data, pd.DataFrame) and column:
        x = data[column].dropna().values
        title = column
    else:
        x = np.array(data).flatten()
        x = x[~np.isnan(x)]
        title = "Data"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. 直方图 + 密度曲线
    axes[0].hist(x, bins='auto', density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x_range = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_range, stats.norm.pdf(x_range, np.mean(x), np.std(x)),
                 'r-', linewidth=2, label='正态分布')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Histogram of {title}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Q-Q图
    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)

    # 3. 箱线图
    axes[2].boxplot(x, vert=True)
    axes[2].set_ylabel('Value')
    axes[2].set_title(f'Boxplot of {title}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出统计量
    jb_stat, p_value = stats.jarque_bera(x)
    print(f"\n{title} - JB统计量: {jb_stat:.4f}, p值: {p_value:.6f}")
    print(f"偏度: {stats.skew(x):.4f}, 峰度: {stats.kurtosis(x, fisher=True):.4f}")
def JB_test(data: pd.DataFrame,):
    input_int = int(input("输入有多少列"))  #这里要注意数据所在列
    for i in range(input_int):
        df1 = data.iloc[:,i]
        jb_test_analysis(df1)
        plot_normality_check(df1)
def sparse(df: pd.DataFrame):
    key_vars = ['孔隙水压力_kPa', '深部位移_mm', '表面位移_mm',"降雨量_mm","微震事件数"]
    scatter_matrix(df[key_vars], figsize=(10, 10), diagonal='hist', alpha=0.5)
    plt.suptitle('变量散点图矩阵', y=1.02)
    plt.show()


def plot_correlation_network(corr_matrix, threshold=0.3):
    """相关性网络图"""
    G = nx.Graph()
    n_vars = len(corr_matrix.columns)

    # 添加节点
    for var in corr_matrix.columns:
        G.add_node(var)

    # 添加边（只保留相关系数绝对值 > threshold 的）
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                           weight=abs(corr_val), corr=corr_val)

    # 绘制
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]  # 边粗细表示相关性强弱

    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=2000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.6)

    # 添加边标签（相关系数）
    edge_labels = {(u, v): f"{G[u][v]['corr']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f'相关性网络图 (|ρ| > {threshold})', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    df = pd.read_excel("监测数据cleaned.xlsx")
    spearman_analysis(df,plot_clustermap=True,plot_heatmap=True)

