from matplotlib import pyplot as plt

from trip_process.read_trips import getTrips
from utils import lonlat2meters
from vis.trajectoryVIS import FileInfo, randomcolor
import numpy as np
import pandas as pd
from sklearn.cluster import dbscan


def get_trip_endpoints(fileInfo, filter_step, use_cell):
    points = []
    trips, lines = getTrips(fileInfo, filter_step, use_cell)
    for index, trip in enumerate(trips):
        points.append(trip[0])
        points.append(trip[-1])
    return points


def dbscan_cluster(points):
    # df = pd.DataFrame(points, columns=['lon', 'lat'])
    # print(df)
    eps, min_samples = 300, 5
    # eps为邻域半径，min_samples为最少点数目
    core_samples, cluster_ids = dbscan(points, eps=eps, min_samples=min_samples)
    # cluster_ids中-1表示对应的点为噪声点
    df = pd.DataFrame(np.c_[points, cluster_ids], columns=['lon', 'lat', 'cluster_id'])

    cluster_id = df['cluster_id'].values.tolist()
    print(cluster_id)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    color_map = {}
    for i, p in enumerate(points):
        if cluster_id[i] in color_map.keys():
            color = color_map[cluster_id[i]]
        else:
            color = randomcolor()
            color_map[cluster_id[i]] = color
        ax.scatter(p[0], p[1], c=color, marker='o', s=7)

    print(len(color_map.keys()))
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.savefig(f'../../figs/dbscan_dist{eps}_min_samples{min_samples}.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    fileInfo = FileInfo()
    points = get_trip_endpoints(fileInfo, 50, False)
    points = [lonlat2meters(p[0], p[1]) for p in points]
    dbscan_cluster(points)