import numpy as np
import math


"""
代码4-1 数据标准化
"""


class DataNorm:
    def __init__(self):
        self.arr = [1,2,3,4,5,6,7,8,9]
        self.x_max = max(self.arr)
        self.x_min = min(self.arr)
        self.x_mean = sum(self.arr) / len(self.arr)
        self.x_std = np.std(self.arr)  # 标准差

    def Min_Max(self):
        arr_ = list()
        for x in self.arr:
            _x = (x-self.x_min)/(self.x_max - self.x_min)
            arr_.append(round(_x,4))
        return arr_

    def Z_Score(self):
        arr_ = list()
        for x in self.arr:
            arr_.append(round((x - self.x_mean) / self.x_std, 4))
        return arr_

    def DecimalScaling(self):  # 小数定标标准化
        arr_ = list()
        j = 1
        x_max = max([abs(one) for one in self.arr])
        while x_max /10 >= 1.0:
            j += 1
            x_max = x_max / 10
        for x in self.arr:
            arr_.append(round(x / math.pow(10,j), 4))
        return arr_

    def Mean(self):  # 均值归一化
        arr_ = list()
        for x in self.arr:
            arr_.append(round((x - self.x_mean) / (self.x_max - self.x_min), 4))
        return arr_

    def Vector(self):  # 向量归一化
        arr_ = list()
        for x in self.arr:
            arr_.append(round(x / sum(self.arr), 4))
        return arr_


if __name__ == "__main__":
    nor = DataNorm()
    min_max = nor.Min_Max()
    print("{}".format(min_max))