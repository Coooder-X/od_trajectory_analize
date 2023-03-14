import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point, MultiPoint, GeometryCollection

from data_process.hierarchical_clustering import get_trip_endpoints
from stme_module import STME
from vis.trajectoryVIS import FileInfo, randomcolor

# k = 15 kt = 10 min_pts = 10  92 clusters
# k = 15 kt = 8 min_pts = 10  83 clusters
# k = 12 kt = 8 min_pts = 10  72 clusters   簇小 多
# k = 12 kt = 6 min_pts = 7  88 clusters
# k = 12 kt = 6 min_pts = 9  125 clusters   目前最优
k = 12  # 20
kt = 5  # 10
min_pts = 10


def get_noise_index_list(df_col, coord_list):
    """
    :param df_col: 当前 dataFrame 中 ‘cluster’ 列的标签数据(已转换成列表)，元素是 number 类型
    :param coord_list: 当前 坐标列表
    :return 返回 所有噪声点在 coords 中的索引，所有噪声点坐标
    """
    noise_index_lst = []
    noise_coord_lst = []
    lst = df_col
    for i, label in enumerate(lst):
        if label == -1:
            noise_index_lst.append(i)
            noise_coord_lst.append(coord_list[i])
    return noise_index_lst, noise_coord_lst


def stme_ones(coord_list, start_cluster_idx, k, kt, min_pts):
    if len(coord_list) <= k:
        return None, None, None
    dataframe = pd.DataFrame({'lat': coord_list[:, 0], 'lon': coord_list[:, 1], 'type': -1, 'cluster': 0})
    dataframe, clusterDensityList_nor, num_of_clusters = STME(dataframe, k=k, kt=kt, t_window=86400, min_pts=min_pts)
    print('stme_ones\n', dataframe['cluster'].values.tolist())
    labels = dataframe['cluster'].values.tolist()
    labels = [x + start_cluster_idx if x != -1 else x for x in labels]
    dataframe['cluster'] = np.array(labels)
    print(dataframe['cluster'].values.tolist())
    return dataframe, clusterDensityList_nor, num_of_clusters


def stme_iteration(coord_list, total_step, start_cluster_idx=0):
    """
    :param start_cluster_idx: 每轮聚类开始的簇的编号
    :param coord_list: 总的要聚类的点列表
    :param total_step: 第一次聚类后，需要继续聚类的次数，即总聚类次数为 total_step + 1
    :return total_df: 总的聚类得到的 dataFrame, cluster_label_set 聚类的标签集合（去重后的标签数组）list 类型, total_num_clusters 聚类的簇的个数
    """
    # 聚类时要计算 k 邻近，因此输入的点个数不能小于 k。总输入坐标数也不能少于 k，该情况主要在 stme_in_cluster() 中出现
    if len(coord_list) <= k:
        return None, None
    total_df, clusterDensityList_nor, num_of_clusters = stme_ones(coord_list, start_cluster_idx, k, kt, min_pts)
    total_cluster_labels = total_df['cluster'].values.tolist()  # 总的聚类簇 id 列表
    # 第一轮聚类得到的噪声数组、噪声点在原数组的索引列表
    noise_index, cur_noise_coords = get_noise_index_list(total_cluster_labels, coord_list)

    for i in range(total_step):
        # 聚类时要计算 k 邻近，因此输入的点个数不能小于 k
        if len(cur_noise_coords) <= k:
            break
        print('当前剩余噪声点个数: ', len(cur_noise_coords))
        start_cluster_idx = max(total_df['cluster']) + 1
        # 本轮聚类的对象是上次聚类中的噪声点，聚类的初始簇 id 是 上次聚类最大 id + 1 （start_cluster_idx）
        cur_df, clusterDensityList_nor, num_of_clusters = stme_ones(np.array(cur_noise_coords), start_cluster_idx, k, kt, min_pts)
        # 基于上次的噪声点数组聚类结果，计算出新的、索引与 cur_noise_coords 对应的簇 label 数组
        cur_cluster_labels = cur_df['cluster']
        # noise_index 是 cur_noise_coords 噪声点对应原点集的索引，本轮噪声聚类后，把本轮聚类结果更新到总聚类结果中
        for idx, label in enumerate(cur_cluster_labels):
            total_cluster_labels[noise_index[idx]] = label
        print('总聚类标签', total_cluster_labels)
        total_df['cluster'] = np.array(total_cluster_labels)
        # 本轮聚类后，检查剩下的噪声点，归到一个数组，并记录它们的索引，由下一轮使用
        noise_index, cur_noise_coords = get_noise_index_list(total_cluster_labels, coord_list)
    # 去重后的标签数组，得到标签集合
    cluster_label_set = list(set(total_df['cluster'].values.tolist()))
    # 总的簇个数
    total_num_clusters = len(cluster_label_set)
    return total_df, cluster_label_set, total_num_clusters


def stme_in_cluster(coord_list, first_step, per_step):
    """
    簇内聚类，在第一次迭代聚类后，对每个簇进行簇内聚类，使用不同的聚类参数。
    簇内聚类后，原簇id不再使用，簇内的起始id从第一次迭代聚类后的最大id开始算。
    :param coord_list: 输入的点坐标列表
    :param first_step: 第一次聚类的开始标签
    :param per_step: 暂时无用，原本作为簇内迭代聚类次数
    :return: dataframe: 总聚类后的 dataframe, old_label_lst: 总聚类后的标签列表
    """
    # 第一次迭代聚类，使得所有点基本都被聚类完成
    dataframe, label_set, num_clusters = stme_iteration(coord_list, first_step)
    # 每个簇内聚类的簇起始 id 为第一次总聚类后，簇的个数（即最大簇id + 1）
    cur_start_idx = max(label_set) + 1
    cluster_labels = dataframe['cluster']
    # 绘制第一次迭代聚类后的结果
    draw_clusters(cluster_labels, num_clusters, '迭代')
    # 由于遍历时不应修改原 dataframe，因此用 dict 记录每个簇在簇内聚类后的簇标签，遍历后再替换。dict 结构为 {id: {index: [], labels: []}}
    old_label_dict = {}
    # 此处忽略噪声点，因为 stme_iteration 后噪声点数少于 k 个
    for i in label_set:
        judge_lst = [cluster_labels == i][0]
        print('cluster_labels', cluster_labels.values.tolist())
        cur_points, cur_points_idx = [], []
        for idx, coord in enumerate(coord_list):
            if judge_lst[idx]:
                cur_points.append(coord)
                cur_points_idx.append(idx)
        print(f'当前簇id为{i}，簇大小为{len(cur_points)}')
        # cur_df, cur_num_clusters = stme_iteration(cur_points, per_step, cur_start_idx)
        cur_k, cur_kt, cur_min_ts = 10, 5, 5
        cur_df, clusterDensityList_nor, cur_num_clusters = stme_ones(np.array(cur_points), cur_start_idx, cur_k, cur_kt, cur_min_ts)
        if cur_df is None:
            continue
        # 对比当前簇内聚类前后的图片，保存
        draw_save_single_cluster_compare(i, cur_points, cur_df, k, kt, min_pts)
        cur_labels = cur_df['cluster']
        print(
            f'当前用于聚类的点数为：{len(cur_points)}，本次簇内聚类产生簇数为{cur_num_clusters}, 产生的噪声点个数：{len([cur_labels == -1][0])}')
        cur_labels = cur_labels.values.tolist()
        old_label_dict[i] = {'index': cur_points_idx, 'labels': cur_labels}
        cur_start_idx = max(cur_labels) + 1

    old_label_lst = cluster_labels.values.tolist()
    for label in old_label_dict.keys():
        idx_lst = old_label_dict[label]['index']
        new_label_lst = old_label_dict[label]['labels']
        for i, idx in enumerate(idx_lst):
            # 簇内聚类后，原有的簇点可能被变成噪声点，此时也置为 -1，若保持原值，则可能导致原簇形状不规则不紧凑
            old_label_lst[idx] = new_label_lst[i]

    dataframe['cluster'] = np.array(old_label_lst)
    return dataframe, old_label_lst


def stme():
    # 纬度在前，经度在后 [latitude, longitude]
    file_info = FileInfo()
    coord_list = get_trip_endpoints(file_info, 50, False)
    coords_set = set()
    for coord in coord_list:
        lon, lat = coord[0], coord[1]
        if not str(lon) + "_" + str(lat) in coords_set:
            coords_set.add(str(lon) + "_" + str(lat))

    coord_list = np.array(coord_list)
    dataframe = pd.DataFrame({'lat': coord_list[:, 0], 'lon': coord_list[:, 1], 'type': -1, 'cluster': 0})

    print("参数: k： " + str(k) + " kt: " + str(kt) + " min_pts: " + str(min_pts))
    dataframe, clusterDensityList_nor, num_of_clusters = STME(dataframe, k=k, kt=kt, t_window=86400,
                                                              min_pts=min_pts)  # 0.0381 751
    # earth's radius in km
    label_of_clusters = dataframe['cluster']
    ratio = len(label_of_clusters[label_of_clusters[:] == -1]) / len(label_of_clusters)  # 计算噪声点个数占总数的比例
    # num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # 获取分簇的数目
    print('ratio:' + str(ratio))
    print('Clustered ' + str(len(coord_list)) + ' points to ' + str(num_of_clusters) + ' clusters')
    dataframe.to_csv("../spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")

    # print('coords', coords)
    # 所有簇的点组成的面
    hulls = []
    figure = plt.figure(figsize=(20, 10))
    axis = figure.subplots()
    for p in coord_list[label_of_clusters == -1]:
        x, y = p[0], p[1]
        axis.scatter(x, y, c='#000', marker='o', s=10)
    for n in range(num_of_clusters):
        # print(n, coords[cluster_labels == n + 1])
        points = []
        cur_color = randomcolor()
        for p in coord_list[label_of_clusters == n + 1]:
            x, y = p[0], p[1]
            points.append(Point(x, y))
            axis.scatter(x, y, c=cur_color, marker='o', s=10)
        # points = [Point(i, j) for i, j in coords[cluster_labels == n + 1]]
        multi_points = MultiPoint(points)
        hulls.append(multi_points.convex_hull)

    axis.set_xlabel('lon')  # 画出坐标轴
    axis.set_ylabel('lat')
    plt.savefig(f'../../figs/k{k}_kt{kt}_mpts{min_pts}.png')
    plt.show()
    return hulls


def draw_save_single_cluster_compare(cluster_id, cluster_points, dataframe, k, kt, min_pts):
    """
    对一个聚类结果中的一个簇进行聚类，可视化前后的对比，分别存储到图片。
    :param cluster_id: 当前簇的 id
    :param cluster_points: 当前簇中的点
    :param dataframe: 当前聚类结果的 dataframe
    :param k:
    :param kt:
    :param min_pts:
    """
    # print('draw_save_single_cluster_compare', cluster_points)
    cluster_points = np.array(cluster_points)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c='g', marker='o', s=20)
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    folder = '../../figs/single_cluster_compare'
    if not os.path.exists(folder):
        os.mkdir(folder)
    # plt.savefig(f'{folder}/cluster{cluster_id}_k{k}_kt{kt}_mpts{min_pts}.png')
    # plt.show()
    plt.savefig(f'{folder}/aaa_cluster.png')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    df_cluster_label_col = dataframe['cluster']
    cluster_label_set = list(set(df_cluster_label_col.values.tolist()))
    for p in cluster_points[df_cluster_label_col == -1]:
        x, y = p[0], p[1]
        ax.scatter(x, y, c='#000', marker='o', s=20)
    for n in cluster_label_set:
        # color = randomcolor()
        for p in cluster_points[df_cluster_label_col == n]:
            x, y = p[0], p[1]
            # print('x, y', x, y)
            ax.scatter(x, y, c='r', marker='o', s=20)
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    # plt.savefig(f'{folder}/cluster{cluster_id}_k{k}_kt{kt}_mpts{min_pts}_after.png')
    plt.savefig(f'{folder}/aaa_cluster_after.png')
    # plt.show()

def draw_clusters(df_cluster_label_col, num_clusters, cluster_type):
    """
    :param df_cluster_label_col: 聚类结果，一个 dataframe 类型的 'cluster' 列的数据
    :param num_clusters: 聚类的簇个数
    :param cluster_type: '簇内' or '迭代'，代表输入数据是使用的哪种算法，用于图片保存命名
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    cluster_label_set = list(set(df_cluster_label_col.values.tolist()))
    # print(coords.shape, cluster_labels.shape)
    for p in coords[df_cluster_label_col == -1]:
        x, y = p[0], p[1]
        ax.scatter(x, y, c='#000', marker='o', s=10)
    for n in cluster_label_set:
        points = []
        color = randomcolor()
        for p in coords[df_cluster_label_col == n]:
            x, y = p[0], p[1]
            points.append(Point(x, y))
            ax.scatter(x, y, c=color, marker='o', s=10)
        # points = [Point(i, j) for i, j in coords[cluster_labels == n + 1]]
        multipoints = MultiPoint(points)
        # hulls.append(multipoints.convex_hull)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.savefig(f'../../figs/{cluster_type}聚类_k{k}_kt{kt}_mpts{min_pts}.png', dpi=300)
    # plt.show()


if __name__ == "__main__":
    fileInfo = FileInfo()
    coords = get_trip_endpoints(fileInfo, 50, False)
    coords = np.array(coords)

    # 迭代聚类
    # df, label_set, num_clusters = stme_iteration(coords, 4)
    # print(f'簇个数：{num_clusters}')
    # cluster_labels = df['cluster']
    # draw_clusters(cluster_labels, '迭代')

    # # 簇内聚类
    # df, cluster_label_list = stme_in_cluster(coords, 4, 0)
    # print('最终聚类结果：', cluster_label_list)
    # print(sorted(cluster_label_list))
    # cluster_labels = df['cluster']
    # num_clusters = max(cluster_label_list) + 1
    # draw_clusters(cluster_labels, num_clusters, '簇内')

    # df, _, num_clusters = stme_ones(coords, 0)
    # draw_clusters(df['cluster'], num_clusters, '')

    # 单簇 簇内聚类 test
    df, label_set, num_clusters = stme_iteration(coords, 4)
    cluster_id = 5
    cluster_points = coords[df['cluster'] == cluster_id]
    df, _, num_clusters = stme_ones(cluster_points, 0, 14, 7, 7)
    draw_save_single_cluster_compare(cluster_id, cluster_points, df, 20, 10, 10)
