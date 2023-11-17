#coding:utf-8
import pickle

import math
import numpy as np
import torch
from sklearn.cluster import KMeans
import torch.nn as nn #专门为神经网络设计的模块化接口
import os

from data_process.SpatialRegionTools import cell2coord, inregionT, gpsandtime2cell, cell2vocab, tripandtime2seq, trip2seq
from data_utils import MyDataOrderScaner
from model.models import EncoderDecoder_without_dropout, EncoderDecoder
import time, os, shutil, logging, h5py

# from graph_process.Graph import Graph
# from graph_process.Point import Point
# from cluster.ClusteringModule import ClusterModule,calculateP,calculate_center_dist
# from cluster import ClusterTool
# import numpy as np
# from functools import cmp_to_key
# from sklearn import metrics
# from scipy.spatial.distance import pdist
# from scipy.spatial.distance import squareform

from t2vec import args


def get_feature_and_trips(args, gps_trips):
    print("**************** use model to generate label and show result with picture************")
    # print('gps_trips', gps_trips)
    with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/region.pkl", 'rb') as file:
        region = pickle.loads(file.read())
    cell_trips = []
    for gps_trip in gps_trips:
        cell_trj = tripandtime2seq(region, gps_trip)
        # cell_trj = " ".join(cell_trj)
        cell_trips.append(cell_trj)
    print('cell_trips', cell_trips)
    torch.cuda.set_device(args.device)  # 指定第几块显卡
    # 初始化需要评估的数据集
    scaner = MyDataOrderScaner(cell_trips, len(cell_trips))
    scaner.load()
    # 模型框架
    m0 = EncoderDecoder_without_dropout(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.bidirectional)
    # m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),  # Sequential:按顺序构建网络
    #                    nn.LogSoftmax(dim=1))

    # 加载模型参数
    if not os.path.isfile(args.best_model):  # 如果完成了预训练，但是没有进行联合训练
        raise Exception
    else:
        print("=> loading best_model '{}'".format(args.best_model))
        logging.info("loading best_model @ {}".format(time.ctime()))
        best_model = torch.load(args.best_model, map_location='cuda:' + str(args.device))
        m0.load_state_dict(best_model["m0"])
        # m1.load_state_dict(best_model["m1"])
        if args.cuda and torch.cuda.is_available():
            m0.cuda()
        m0.eval()

    feature = []
    cell_trips = []
    # labels = []
    i = 0
    while True:
        i = i + 1
        src, lengths, invp, label = scaner.getbatch_scaner()  # src[12,64] invp是反向索引
        if src is None:
            break
        cell_trips.extend(src.t())
        if args.cuda and torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        # 计算encoder学习到的轨迹表示
        h, _ = m0.encoder(src, lengths)
        h = m0.encoder_hn2decoder_h0(h)  # (num_layers, batch, hidden_size * num_directions)
        h = h.transpose(0, 1).contiguous()  # (batch, num_layers, hidden_size * num_directions)，例如 [64, 3, 256]
        h2 = h[invp]
        size = h2.size()
        # 使用三层特征拼接的特征
        h2 = h2.view(size[0], size[1] * size[2])
        feature.append(h2.cpu().data)
        # labels.extend(label)
    feature = torch.cat(feature)
    feature = feature[scaner.shuffle_invp]
    return feature, cell_trips

    # features, cell_trips = run_model(args,
    #          args.best_model,
    #          scaner,
    #          scaner.get_data_num())
    # print(len(features), len(cell_trips))
    # # print(features)
    # # print(cell_trips)
    # return features, cell_trips


def run_model2(args, gps_trips, best_model, trj_region):
    tmp_gps_trips = []
    for trip in gps_trips:
        tmp_trip = []
        for p in trip:
            tmp_trip.append([p[0], p[1]])
        tmp_gps_trips.append(tmp_trip)
    gps_trips = tmp_gps_trips

    vecs = []
    torch.cuda.set_device(args.device)  # 指定第几块显卡
    # 创建预训练时候用到的模型评估其性能
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    # 取出最好的model
    if os.path.isfile(args.best_model):
        # print("=> loading best_model '{}'".format(args.best_model))
        # best_model = torch.load(args.best_model)
        # # 存时，"m0": m0.state_dict()
        m0.load_state_dict(best_model["m0"])
        if args.cuda and torch.cuda.is_available():
            m0.cuda()  # 注意：如果训练的时候用了cuda,这里也必须m0.cuda
        # 如果模型含dropout、batch normalization等层，需要该步骤
        m0.eval()

        # with open("/home/zhengxuan.lin/project/deepcluster/data/region.pkl", 'rb') as file:
        #     region = pickle.loads(file.read())
        cell_trips = []
        for gps_trip in gps_trips:
            # cell_trj = tripandtime2seq(region, gps_trip)
            cell_trj = trip2seq(trj_region, gps_trip)
            # cell_trj = " ".join(cell_trj)
            cell_trips.append(cell_trj)
        # print('cell_trips', len(cell_trips), cell_trips)
        # torch.cuda.set_device(args.device)  # 指定第几块显卡
        # 初始化需要评估的数据集
        scaner = MyDataOrderScaner(cell_trips, len(cell_trips))
        scaner.load()

        i = 0
        while True:
            if i % 10 == 0:
                print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
            i = i + 1
            # 获取到的排序后的轨迹数据、每个轨迹的长度及其排序前的索引
            src, lengths, invp, label = scaner.getbatch_scaner()
            # 没有数据则处理完毕
            if src is None: break
            if args.cuda and torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            # 不需要进入m0的前向传播，因为取出了解码器，只进行编码
            h, _ = m0.encoder(src, lengths)
            ## (num_layers, batch, hidden_size * num_directions)
            # 对最后一个hidden的结果进行处理
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions)
            # 转置，并强制拷贝一份tensor
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            # h = h.view(h.size(0), -1)
            # 回到轨迹的原始顺序
            h2 = h[invp]
            size = h2.size()
            h2 = h2.view(size[0], size[1] * size[2])
            # 把3层特征拼接成一个特征
            vecs.append(h2.cpu().data)
        # todo-----------------
        ## (num_seqs, num_layers, hidden_size * num_directions)
        # 把一批批的batch合并, vecs.shape 1899, 3, 256
        # 1899个轨迹
        vecs = torch.cat(vecs)
        # size = vecs.size()
        # 采取不同层的特征
        # feature = vecs[:, 2, :]  # 只提取低3层特征
        # feature1 = vecs[:, 0, :]  # 只提取第1层特征
        # feature2 = vecs[:, 1, :]  # 只提取第2层特征
        # feature = torch.cat((feature2, feature3), 1)  # 合并前两层特征

        # 把3层特征拼接成一个特征
        # feature = vecs.view(size[0], size[1] * size[2])
    # print('vecs', vecs)
    return vecs


def run_model(args, model, data, data_num):
    # 定义模型
    m0_2 = EncoderDecoder_without_dropout(args.vocab_size,
                                          args.embedding_size,
                                          args.hidden_size,
                                          args.num_layers,
                                          args.bidirectional)
    if args.cuda and torch.cuda.is_available():
        m0_2.cuda()
    m0_2_optimizer = torch.optim.Adam(m0_2.parameters(), lr=args.learning_rate)
    # 加载模型
    best_model = torch.load(model, map_location='cuda:' + str(args.device))
    m0_2.load_state_dict(best_model["m0"])
    m0_2_optimizer.load_state_dict(best_model["m0_optimizer"])

    # 神经网络前向传播，获得降维后的特征
    data.start = 0  # 每一个epoch计算qij时，先让scaner的start指针归零
    m0_2.eval()
    feature = []
    cell_trips = []
    # labels = []
    i = 0
    while True:
        i = i + 1
        src, lengths, invp, label = data.getbatch_scaner()  # src[12,64] invp是反向索引
        if src is None:
            break
        cell_trips.extend(src.t())
        if args.cuda and torch.cuda.is_available():
            src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
        # 计算encoder学习到的轨迹表示
        h, _ = m0_2.encoder(src, lengths)
        h = m0_2.encoder_hn2decoder_h0(h)  # (num_layers, batch, hidden_size * num_directions)
        h = h.transpose(0, 1).contiguous()  # (batch, num_layers, hidden_size * num_directions)，例如 [64, 3, 256]
        h2 = h[invp]
        size = h2.size()
        # 使用三层特征拼接的特征
        h2 = h2.view(size[0], size[1] * size[2])
        feature.append(h2.cpu().data)
        # labels.extend(label)
    feature = torch.cat(feature)
    feature = feature[data.shuffle_invp]
    return feature, cell_trips
        # labels = np.array(labels)[data.shuffle_invp]
    # if not args.hasLabel:
    #     labels = None

    # k-means聚类，获取初始簇中心
    # centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(path, args.sourcedata, k=args.clusterNum,


# def get_trips_lonlat(region, trj_datas, min_length=2, max_length=1000):
#     print()
#     print("Create *.ori files")
#     for idx, trj_data in enumerate(trj_datas):
#         # 判断轨迹长度
#         if not (min_length <= len(trj_data) <= max_length):
#             continue
#         print()


def get_cluster_by_trj_feature(args, feature):
    centroids, inertia_start, inertia_end, n_iter, labels_dict, labels = get_cluster_centroid(args, feature)
    return labels


def get_cluster_centroid(args, feature):
    # feature = torch.tensor(feature)
    centroids, inertia_start, inertia_end, n_iter, \
    cluster_data_neighbors, labels_dict, labels\
        = DoKMeansWithError(k=args.clusterNum,
        center=None, feature=feature.cpu().data)

    return centroids, inertia_start, inertia_end, n_iter, labels_dict, labels


def DoKMeansWithError(k=10, center=None, feature=None):
    '''
    读取h5格式的数据，进行kmeans聚类
    :param feature_path: 降维后的特征的路径（h5格式的数据 trj.h5）
    :param k: 簇的个数
    :param n: 只读取前n条轨迹
    :return:
    '''
    data = feature

    # -----3、kmeans聚类
    # (1)聚类
    if center is None:
        inertia_start = 0
        # kmeans = KMeans(n_clusters=k, n_init=30, tol=1e-5, max_iter=10000).fit(data)
        # 根据簇数目做聚类
        k = int(len(data) * 2 / 5)
        print('簇数目', k)
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(data)  # 模拟数据用这个
    # else:
    #     inertia_start = calculate_feature_center_dist(data.cpu().data.numpy(), center.numpy())
    #     # inertia_start = 0
    #     kmeans = KMeans(n_clusters=k, init=center.numpy(), n_init=1, tol=1e-5, max_iter=10000).fit(data)
    # 得到簇中心
    centroids = kmeans.cluster_centers_
    # 计算每个轨迹分配到最近的簇中心后求出其距离和
    inertia_end = calculate_feature_center_dist(data.cpu().data.numpy(), centroids)
    # inertia_end = kmeans.inertia_
    # 迭代次数
    n_iter = kmeans.n_iter_
    #
    labels = kmeans.labels_

    # 查找热点轨迹和异常轨迹，并修改标签
    # find_hot_abnormal(data, k, centroids, labels)

    # (2) 修改簇标签
    labels_dict = {}
    for i in range(len(labels)):
        # 获取每个轨迹的聚类后label
        labels_dict[i] = labels[i]  # +1让聚类结果的簇号从1-19，和真值保持一致

    cluster_data_neighbors = {}
    for i in range(k):
        data_ids = get_keys(labels_dict, i)
        for data_id in data_ids:
            neighbor = np.array(data_ids)[np.random.randint(len(data_ids) / 2, len(data_ids), 3)]
            cluster_data_neighbors[data_id] = neighbor

    return centroids, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict, labels


def calculate_feature_center_dist(feature, center):
    num_feature = len(feature)
    num_center = len(center)
    res = 0
    for i in range(num_feature):
        tmp=[]
        for j in range(num_center):
            dist = ((feature[i] - center[j]) ** 2).sum(0)
            tmp.append(dist)
        min_dist = min(tmp)
        res += min_dist
    return res


# 查找字典中指定的value的key
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def build_graph():
    # g = Graph()
    feature, cell_trips = get_feature_and_trips(args, [])
    # point_lst = []
    # for idx, trip in enumerate(cell_trips):
    #     point_lst.append(Point(name=str(idx), nodeId=idx, infoObj={'cell_trip': trip}, feature=feature[idx]))
    # for p in point_lst:
    #     g.addVertex(p)
    #
    # for i, ti in enumerate(cell_trips):
    #     tar_lst = []
    #     for j, tj in enumerate(cell_trips):
    #         if ti[-1] == tj[0]:
    #             tar_lst.append([point_lst[j], 1])
    #     g.addDirectLine(point_lst[i], tar_lst)
    #
    # # g.drawGraph()
    # return g, feature
    return feature


if __name__ == "__main__":
    # showPic(args)
    build_graph()
