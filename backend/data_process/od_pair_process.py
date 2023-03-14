import json
import pickle
from datetime import datetime

import numpy as np

from data_process.hierarchical_clustering import get_trip_endpoints
from poi_process.read_poi import lonlat2meters_coords
from vis.trajectoryVIS import FileInfo


def get_data():
    fileInfo = FileInfo()
    return get_trip_endpoints(fileInfo, 1, False)


def get_total_od_points():
    start_time = datetime.now()
    with open("./data/全天OD点经纬度(带轨迹id).pkl", 'rb') as file:
        od_points = pickle.loads(file.read())
    print('读取文件结束，用时: ', (datetime.now() - start_time))
    # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
    res = []
    for (idx, od) in enumerate(od_points):
        if idx % 24 == 0:
            res.append(od.tolist())
    return json.dumps(res)


if __name__ == '__main__':
    # print('开始读取OD点')
    # start_time = datetime.now()
    # od_points = np.asarray(get_data())
    # print(od_points[0])
    # print('读取OD点结束，用时: ', (datetime.now() - start_time))
    #
    # start_time = datetime.now()
    # with open("../data/全天OD点经纬度(带轨迹id).pkl", 'wb') as f:
    #     picklestring = pickle.dumps(od_points)
    #     f.write(picklestring)
    # print('写入文件结束，用时: ', (datetime.now() - start_time))

    start_time = datetime.now()
    with open("../data/全天OD点经纬度(带轨迹id).pkl", 'rb') as file:
        od_points = pickle.loads(file.read())
    print('读取文件结束，用时: ', (datetime.now() - start_time))
    print(len(od_points), od_points)   # 读取文件结束，用时:  0:00:00.004556

    # hour_od_pairs = od_points[(14*3600 < od_points[:, 2]) & (od_points[:, 2] < 15*3600)]
    # print(hour_od_pairs, len(hour_od_pairs))

