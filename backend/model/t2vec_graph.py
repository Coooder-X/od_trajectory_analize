#coding:utf-8
import torch
import torch.nn as nn #专门为神经网络设计的模块化接口
import os

from data_utils import DataOrderScaner
# from data_utils import DataOrderScaner
from models import EncoderDecoder_without_dropout
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
    print('gps_trips', gps_trips)
    torch.cuda.set_device(args.device)  # 指定第几块显卡
    # 初始化需要评估的数据集
    scaner = DataOrderScaner(os.path.join(args.data, "train.ori"), os.path.join(args.data, "train.label"), args.batch)
    scaner.load()
    # 模型框架
    m0 = EncoderDecoder_without_dropout(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),  # Sequential:按顺序构建网络
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        m0.cuda()
        m1.cuda()
    # 加载模型参数
    if not os.path.isfile(args.best_model):  # 如果完成了预训练，但是没有进行联合训练
        raise Exception
    else:
        print("=> loading best_model '{}'".format(args.best_model))
        logging.info("loading best_model @ {}".format(time.ctime()))
        best_model = torch.load(args.best_model, map_location='cuda:' + str(args.device))
        m0.load_state_dict(best_model["m0"])
        m1.load_state_dict(best_model["m1"])

    features, cell_trips = run_model(args,
             args.best_model,
             scaner,
             scaner.get_data_num())
    print(len(features), len(cell_trips))
    print(features)
    print(cell_trips)
    return features, cell_trips


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
