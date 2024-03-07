# -*- coding: utf-8 -*-
import pandas as pd

def read_gridod():
    # 读取文件
    # pd.set_option('display.max_columns', None)
    data = pd.read_csv('./raw_data/NYC_TOD/NYC_TOD_100.gridod')

    # 显示数据的前几行
    print(data.head())
    num_rows = data.shape[0]
    # print(f"num of lines：{num_rows}")
    print(num_rows)

    # 获取最后一列的数据
    last_column = data.iloc[:, -1]

    # 计算最后一列数据的和
    sum_of_last_column = last_column.sum()

    print("last col sum =", sum_of_last_column)

def create_geo():
    # 创建一个数据字典
    data = {
        'geo_id': [],
        'type': [],
        'coordinates': [],
        'row_id': [],
        'column_id': []
    }

    for i in range(0, 10):
        for j in range(0, 10):
            id = i * 10 + j
            print(id)
            data['geo_id'].append(id)
            data['type'].append('Polygon')
            data['coordinates'].append([])
            data['row_id'].append(i)
            data['column_id'].append(j)

    # 将数据字典转换为DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame写入.geo文件
    df.to_csv('./raw_data/NYC_TOD/NYC_TOD2.geo', index=False)


if __name__ == '__main__':
    # create_geo()
    read_gridod()
