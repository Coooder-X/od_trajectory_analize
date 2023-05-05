import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import spatial

from utils import lonlat2meters, meters2lonlat

# from utils import lonlat2meters, meters2lonlat


config_dict = {
    'poi_dir': "/home/linzhengxuan/project/hangzhou-POI",
    'poi_file_name_lst': ['餐饮.xlsx', '政府机构及社会团体.xlsx', '生活服务.xlsx', '商务住宅.xlsx', '金融保险服务.xlsx', '风景名胜.xlsx', '医疗保险服务.xlsx', '购物服务.xlsx', '交通设施服务.xlsx', '体育休闲服务.xlsx', '科技文化服务.xlsx', '住宿服务.xlsx', '公共设施.xlsx']
}


def getPOI_Coor(data_dir):
    """
    :param
    :return total_poi_coor: numpy 数组，存储所有 POI 坐标
    :return file_name_poi_id_dict: 存储 map<文件id（POI类别）, 全量POI索引> 的映射
    :return poi_id_file_id_dict: 存储 map<全量POI索引, 文件id（POI类别）> 的映射
    """
    file_name_lst = []
    print('当前要读取的 POI 文件: ')
    file_list = config_dict['poi_file_name_lst']
    print('file_list', file_list)
    for i in range(len(file_list)):
        file_name_lst.append(file_list[i])
        # file_list[i] = data_dir + '/' + file_list[i]
        # print(file_list[i])

    total_poi_coor = []
    file_id_poi_id_dict = {}
    poi_id_file_id_dict = {}
    poi_id = 0
    for i, file_name in enumerate(file_name_lst):
        df1 = pd.read_excel(data_dir + '/' + file_name)  # 读取xlsx中的第一个sheet
        poi_coor = np.asarray(df1.iloc[:, [-2, -1]].values)
        print(poi_coor)
        total_poi_coor.extend(poi_coor)

        file_id_poi_id_dict[i] = []
        for coord in poi_coor:
            file_id_poi_id_dict[i].append(poi_id)
            poi_id_file_id_dict[poi_id] = i
            poi_id += 1

    print('最大POI Id', poi_id, 'poi 个数', len(total_poi_coor))
    return np.asarray(total_poi_coor), file_id_poi_id_dict, poi_id_file_id_dict


def buildKDTree(pois):
    """
    将所有POI坐标放入 KDTree，方便查找邻近点，从距离上判断连通性
    :param: GPS 经纬度格式的POI数组
    :return 存储POI坐标的 KDTree
    """
    kdtree = spatial.KDTree(pois)
    return kdtree


def get_poi_type_filter_by_radius(point_in_cluster, poi_id_file_id_dict, config_dict, kdtree, radius):
    """
    将 poi 根据一个 OD 点以及半径进行过滤
    输入的 kdtree 和 poi_coor 中的 poi 坐标单位是米
    :return poi_type_dict: 形如 map<'餐饮', 23> 的统计每个类别 POI 数量的 map
    """
    points = []
    for point in point_in_cluster:
        x, y = lonlat2meters(point[0], point[1])
        points.append([x, y])
    id_list = kdtree.query_ball_point(points, radius)

    id_set = set()
    for item in id_list:
        for pid in item:
            id_set.add(pid)

    poi_type_dict = {}
    for poi_id in id_set:
        print(config_dict['poi_file_name_lst'])
        print('file id', poi_id_file_id_dict[poi_id])
        print(config_dict['poi_file_name_lst'][poi_id_file_id_dict[poi_id]])
        file_type = config_dict['poi_file_name_lst'][poi_id_file_id_dict[poi_id]].split('.')[0]
        file_type = str(file_type)
        if file_type not in poi_type_dict:
            poi_type_dict[file_type] = 0
        poi_type_dict[file_type] += 1

    # label 名称修改
    tmp = []
    for key in poi_type_dict:
        t_key = key
        t_key = t_key.split('服务')[0]
        if '政府机构及社会团体' == t_key:
            t_key = '政府机构'
        tmp.append({ 'type': t_key, 'value': poi_type_dict[key] })
    poi_type_dict = tmp
    return poi_type_dict
    # coor_filtered = []
    # for row in id_list:
    #     # 加判断，如果 row 为空 list，则用 topk 的方式取一次 POI？
    #     if len(points) == 1:
    #         x, y = meters2lonlat(poi_coor[row][0], poi_coor[row][1])
    #         coor_filtered.append([x, y])
    #     else:
    #         for id in row:
    #             x, y = meters2lonlat(poi_coor[id][0], poi_coor[id][1])
    #             coor_filtered.append([x, y])
    # return coor_filtered


def lonlat2meters_poi(poi_coor):
    poi_meters_coor = []
    for poi in poi_coor:
        x, y = lonlat2meters(poi[0], poi[1])
        poi_meters_coor.append([x, y])
    return  poi_meters_coor


def lonlat2meters_coords(coords, use_time=False):
    meters_coor = []
    for coord in coords:
        (x, y), time_stamp, trj_id, flag = (lonlat2meters(coord[0], coord[1]), coord[2], coord[3], coord[4])
        if use_time:
            meters_coor.append([x, y, time_stamp, trj_id, flag])
        else:
            meters_coor.append([x, y, trj_id, flag])
    return meters_coor


def meters2lonlat_list(coord_lst):
    ans = []
    for p in coord_lst:
        x, y = meters2lonlat(p[0], p[1])
        ans.append([x, y])
    return ans


if __name__ == '__main__':
    start_time = datetime.now()
    total_poi_coor, file_id_poi_id_dict, poi_id_file_id_dict = getPOI_Coor(config_dict['poi_dir'])
    total_poi_coor = lonlat2meters_poi(total_poi_coor)
    kdtree = buildKDTree(total_poi_coor)
    with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'wb') as f:
        picklestring = pickle.dumps({
            'total_poi_coor': total_poi_coor,
            'file_id_poi_id_dict': file_id_poi_id_dict,
            'poi_id_file_id_dict': poi_id_file_id_dict,
            'kdtree': kdtree,
        })
        f.write(picklestring)
    print('写入文件结束，用时: ', (datetime.now() - start_time))

