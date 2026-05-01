import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
from scipy import stats

warnings.filterwarnings('ignore')



def fun1():
    df1 = pd.read_excel("附件3：监测数据（训练集与实验集）-问题3.xlsx",sheet_name='训练集')
    df1 = df1.iloc[:,1:2]
    df2 = pd.read_excel("cleaned.xlsx")
    df2 = df2.iloc[:,2:3]

if __name__ == '__main__':
    fun1()