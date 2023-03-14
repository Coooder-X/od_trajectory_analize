from datetime import datetime

import numpy
import numpy as np  # version 1.17.2
from scipy.spatial import Delaunay  # version 1.4.1
import matplotlib.pyplot as plt  # version 3.1.2

from data_process.hierarchical_clustering import get_trip_endpoints, get_trip_endpoints_filter_by_coords
from poi_process.read_poi import buildKDTree, lonlat2meters_coords
from utils import UnionFindSet
from vis.trajectoryVIS import FileInfo, randomcolor


def get_data():
    fileInfo = FileInfo()
    return get_trip_endpoints(fileInfo, 120, False)


def get_data_filter_by_coords(p1, p2):
    fileInfo = FileInfo()
    return get_trip_endpoints_filter_by_coords(fileInfo, 12, False, p1, p2)


def create_delauney(points):
    """
    :param points:  npArray 类型的轨迹端点数组（暂不包含时间）
    :return:        scipy.spatial.Delaunay 创建的三角剖分对象
    """
    # create a Delauney object using (x, y)
    tri = Delaunay(points)

    # paint a triangle
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), c='black')
    # plt.plot(points[:, 0], points[:, 1], 'o', c='green')
    # plt.axis('equal')
    # plt.show()
    return tri


def create_delaunay_graph(tri, points):
    """
    :param tri:     scipy.spatial.Delaunay 创建的三角剖分对象
    :param points:  npArray 类型的轨迹端点数组（暂不包含时间）
    :return:        返回三角剖分得到的图的邻接表. [[to_i, to_j, ...], [...], ...] (无向图，双向边)
    """
    # print(tri.vertex_neighbor_vertices)
    adj_list = []
    (indptr, indices) = tri.vertex_neighbor_vertices
    for k in range(len(points)):
        to_index = indices[indptr[k]:indptr[k + 1]]
        cur_point_adj = [to for to in to_index]
        adj_list.append(cur_point_adj)
    return adj_list


def cal_adj_dist(adj_list, od_kdtree, od_points, k):
    """
    :param adj_list:    距离全为 0 的邻接表
    :param od_kdtree:   od 点构成的 kdtree
    :param od_points:   od 点数组
    :param k:           计算 k 邻近的参数 k
    :return:            edge_lst: 三角剖分的边数组，每个原素是元组：(from, to, dist)
    """
    edge_lst = []
    vis = np.zeros((len(od_points), len(od_points)), dtype='bool')  # 避免重复遍历，优化大量时间
    top_k_dict = {}  # 记录 点id到其topk点set 的映射，能优化大量时间
    #   a_set, b_set 是当前边两端点各自的 k 邻近点集合
    for (a_idx, adj_point) in enumerate(adj_list):
        if a_idx in top_k_dict:
            a_set = top_k_dict[a_idx]
        else:
            _, a_top_k_id = od_kdtree.query(od_points[a_idx], k)
            a_set = set()
            for idx in a_top_k_id:
                a_set.add(idx)
            top_k_dict[a_idx] = a_set

        for b_idx in adj_point:
            if vis[a_idx][b_idx]:
                continue

            if b_idx in top_k_dict:
                b_set = top_k_dict[b_idx]
            else:
                _, b_top_k_id = od_kdtree.query(od_points[b_idx], k)
                b_set = set()
                for idx in b_top_k_id:
                    b_set.add(idx)
                top_k_dict[b_idx] = b_set

            #   dist = 1 - size(a ∩ b) / size(a ∪ b)
            # dist = 1 - len(a_set.intersection(b_set)) / len(a_set.union(b_set))   # 使用下面代码优化计算
            intersection_size = len(a_set.intersection(b_set))
            dist = 1 - intersection_size / (len(a_set) + len(b_set) - intersection_size)
            edge_lst.append((a_idx, b_idx, dist))
            edge_lst.append((b_idx, a_idx, dist))
            vis[a_idx][b_idx] = vis[b_idx][a_idx] = True

    return edge_lst


def delaunay_clustering(k: int, theta: int, od_points: list):
    """
    参考自论文：《 Discovering Spatial Patterns in Origin-Destination Mobility Data 》
    :param k:           聚类参数 k，以每个点的 k 邻近计算 dist
    :param theta:       聚类参数 θ，每个簇至少 θ 个点
    :param od_points:   所有 od 点的经纬度数组 [lon, tat]
    :return:            new_point_cluster_dict, new_cluster_point_dict, 分别是点id到簇id的映射和 簇id到点id的映射
                        后者的 value 是 set 集合，包含该簇下所有的点 id
    """
    od_kdtree = buildKDTree(od_points)
    #   论文步骤1，生成三角剖分对象
    triangulation = create_delauney(od_points)
    adj_list = create_delaunay_graph(triangulation, od_points)

    #   论文步骤2，计算三角剖分得到的图的 邻接表 和 边数组
    start = datetime.now()
    edge_lst = cal_adj_dist(adj_list, od_kdtree, od_points, k)
    end = datetime.now()
    print('步骤2 遍历图计算 dist，用时: ', (end - start))

    #   论文步骤3，初始化聚类，每个 od 点都是一个簇。使用并查集优化集合操作。
    uf_set = UnionFindSet(range(len(od_points)))

    #   论文步骤4，三角剖分的边按升序排列
    edge_lst = sorted(edge_lst, key=lambda x: x[2])

    for edge in edge_lst:
        (src, tar, dist) = edge
        src_cluster_id = uf_set.find_head(src)
        tar_cluster_id = uf_set.find_head(tar)
        #  若两端点属于不同簇，且其中一个簇大小小于 θ，则合并 tar 所在簇到 src 所在簇
        if src_cluster_id != tar_cluster_id and \
                (uf_set.get_size(src_cluster_id) < theta or uf_set.get_size(tar_cluster_id) < theta):
            uf_set.union(src_cluster_id, tar_cluster_id)

    cluster_id = 0
    old_new_id_dict = {}
    cluster_point_dict = {}
    point_cluster_dict = {}

    #   将并查集结构转成 dict 数据结构，并重新编排簇 id 从 0 开始
    for p_idx in range(len(od_points)):
        old_cluster_id = uf_set.find_head(p_idx)
        if old_cluster_id not in old_new_id_dict:  # 若旧簇id还没有映射的新簇id
            old_new_id_dict[old_cluster_id] = cluster_id  # 添加新旧簇id映射
            cluster_point_dict[cluster_id] = set([p_idx])  # 初始化 簇id-点集合 映射
            point_cluster_dict[p_idx] = cluster_id
            cluster_id += 1
        else:  # 若旧簇id对应的新簇id存在
            new_cluster_id = old_new_id_dict[old_cluster_id]
            cluster_point_dict[new_cluster_id].add(p_idx)
            point_cluster_dict[p_idx] = new_cluster_id

    return point_cluster_dict, cluster_point_dict


def draw_DT_clusters(cluster_point_dict: dict, od_points: list, k: int, theta: int, start_hour: int, end_hour: int, index_set=None):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    color_dict = {idx: randomcolor() for idx in cluster_point_dict.keys()}

    for cluster_id in cluster_point_dict.keys():
        for p_idx in cluster_point_dict[cluster_id]:
            if p_idx not in index_set:
                continue
            x, y = od_points[p_idx][0:2]
            ax.scatter(x, y, c=color_dict[cluster_id], marker='o', s=4)
        # points = [Point(i, j) for i, j in coords[cluster_labels == n + 1]]
        # multipoints = MultiPoint(points)
        # hulls.append(multipoints.convex_hull)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    if start_hour is not None:
        plt.savefig(f'../../figs/三角剖分聚类_k{k}_theta{theta}_{len(od_points)}points_time{start_hour}-{end_hour}.png', dpi=100)
    else:
        plt.savefig(f'../../figs/三角剖分聚类_k{k}_theta{theta}_{len(od_points)}points.png', dpi=100)
    plt.show()


def od_points_filter_by_hour(od_points, start_hour, end_hour):
    """
    :param od_points:   od 点 npArray，原素是 [x, y, timestamp]
    :param start_hour:  几点开始
    :param end_hour:    几点结束
    :return:            该时间闭区间内的 od 点 npArray, 以及这些点对应全量 od 点的索引
    """
    index_lst = numpy.where((start_hour * 3600 <= od_points[:, 2]) & (od_points[:, 2] <= end_hour * 3600))
    return od_points[(start_hour * 3600 <= od_points[:, 2]) & (od_points[:, 2] <= end_hour * 3600)], index_lst


if __name__ == '__main__':
    k, theta = 8, 10
    print('开始读取OD点')
    start_time = datetime.now()
    od_points = np.asarray(lonlat2meters_coords(coords=get_data(), use_time=True))

    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    print('pos nums', len(od_points), '\n开始聚类')
    start_time = datetime.now()
    point_cluster_dict, cluster_point_dict = delaunay_clustering(k=k, theta=theta, od_points=total_od_coord_points)
    end_time = datetime.now()
    print('结束聚类，用时: ', (datetime.now() - start_time))
    # draw_DT_clusters(cluster_point_dict, od_points, k, theta, start_hour, end_hour)
    # draw_time = datetime.now()
    # print('画图用时: ', draw_time - end_time)

    start_hour, end_hour = 10, 15
    (part_od_coord_points, index_lst) = od_points_filter_by_hour(total_od_coord_points, start_hour, end_hour)  # 过滤出所有在该时间段的 od 点
    index_lst = index_lst[0]
    print(index_lst)
    # part_od_coord_points = part_od_coord_points[:, 0:2]  # 并去掉时间戳留下坐标
    draw_DT_clusters(cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst.tolist()))
    draw_time = datetime.now()
    print('画图用时: ', draw_time - end_time)

