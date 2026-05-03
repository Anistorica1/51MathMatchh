
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

def fun1():
    df = pd.read_excel("5phase3.xlsx")
    df['is_blasting'] = df['爆破点距离_m'].notna().astype(int)

    # 或者如果两个字段都为空才视为非爆破
    df['is_blasting'] = ((df['爆破点距离_m'] != -1) & (df['单段最大药量_kg'] != -1)).astype(int)
    # 用 -1 填充（因为距离和药量物理上 ≥ 0，-1 明显异常）
    df['爆破点距离_m'] = df['爆破点距离_m'].fillna(1000000000)
    df['单段最大药量_kg'] = df['单段最大药量_kg'].fillna(0)
    df.to_excel("5phase3.xlsx")

if __name__ == '__main__':
    fun1()
##

