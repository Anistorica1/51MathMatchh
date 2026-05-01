import numpy as np
import pandas as pd

def fun1():
    df1 = pd.read_excel("共同异常点结果.xlsx")
    df = df1.iloc[:,13]
    df2 = df1.copy()
    num = 0
    for i in df:
        mark = 0
        t = 0
        str = ""
        print(i)
        for j in range(0,5):
            if i[0+t:1+t] == "F":
                t+= 5
                mark+=1
            else:
                if mark == 0:
                    str += "a"
                elif mark == 1:
                    str += "b"
                elif mark == 2:
                    str += "c"
                elif mark == 3:
                    str += "d"
                elif mark == 4:
                    str += "e"
                mark+=1
                t+= 4
        df2.loc[df2["编号"]==num,["共同异常点处的异常变量"]]=[str]
        num+=1
    df2.to_excel("Test3_2.xlsx")
if __name__ == '__main__':
    fun1()