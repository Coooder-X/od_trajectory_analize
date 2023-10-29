import os

from matplotlib import pyplot as plt
from matplotlib import collections as mc
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import pickle
from vis.trajectoryVIS import randomcolor
import time
import numpy as np
from gcc.graph_convolutional_clustering.gcc.metrics import output_metrics, print_metrics
from gcc.graph_convolutional_clustering.gcc.optimizer import optimize
from gcc.graph_convolutional_clustering.gcc.utils import read_dataset, preprocess_dataset
import tensorflow as tf


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Parameters
flags.DEFINE_string('dataset', 'pubmed', 'Name of the graph dataset (cora, citeseer, pubmed or wiki).')
flags.DEFINE_integer('power', 5, 'Propagation order.')
flags.DEFINE_integer('runs', 80, 'Number of runs per power.')
flags.DEFINE_integer('n_clusters', 0, 'Number of clusters (0 for ground truth).')
flags.DEFINE_integer('max_iter', 30, 'Number of iterations of the algorithm.')
flags.DEFINE_float('tol', 10e-7, 'Tolerance threshold of convergence.')
data_path = '/home/zhengxuan.lin/project/od_trajectory_analize/backend/gcc/graph_convolutional_clustering/data'


def draw_cluster_in_trj_view(trj_labels, gps_trips):
    print('长度： ', len(trj_labels), len(gps_trips))
    lines = []
    for trip in gps_trips:
        line = []
        for j in range(len(trip) - 1):
            line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
        lines.append(line)

    label_color_dict = {}
    st = set(trj_labels)
    for label in st:
        label_color_dict[label] = randomcolor()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for index, line in enumerate(lines):
        color = label_color_dict[trj_labels[index]]
        lc = mc.LineCollection(line, colors=color, linewidths=2)
        ax.add_collection(lc)
    for index, trip in enumerate(gps_trips):
        trip = np.array(trip)
        color = label_color_dict[trj_labels[index]]
        ax.scatter(trip[0][0], trip[0][1], s=8, c=color, marker='o')
        ax.scatter(trip[-1][0], trip[-1][1], s=8, c=color, marker='o')

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()
    plt.close()


def run(adj, features):
    """
    @param adj: csc 类型的稀疏矩阵，表示线图的邻接矩阵，节点的顺序与 G.nodes() 一致，切和 features 的顺序对应相同的节点
    @param features: 二维 numpy，第一维长度是节点个数，第二位长度是特征维度，节点顺序同上
    @return G: 返回 numpy 类型的数组，索引是节点id（从0-n），值是 label。顺序同上
    """
    # 社区划分的数量为 节点数 / 10
    n_classes = len(features) // 10
    # 在图神经网络中节点特征维度（被压缩后）(这个值最初在原项目中被设定为 n_classes 的值)
    node_feat_dim = 500
    # 保存训练完的 W 矩阵的文件名
    model_W_file_name = 'gcc_W'

    dataset = flags.FLAGS.dataset
    power = flags.FLAGS.power
    runs = flags.FLAGS.runs
    n_clusters = flags.FLAGS.n_clusters
    max_iter = flags.FLAGS.max_iter
    tolerance = flags.FLAGS.tol

    # Read the dataset
    # adj, features, labels, n_classes = read_dataset(data_path, dataset)
    # adj, features, labels, n_classes = get_data_from_frontend(G, data_path, dataset)

    if n_clusters == 0: n_clusters = n_classes
    # Process the dataset
    tf_idf = (dataset == 'cora' or dataset == 'citeseer')  # normalize binary word datasets
    norm_adj, features = preprocess_dataset(adj, features, tf_idf=tf_idf)

    run_metrics = []
    times = []

    X = features

    def infer():
        features = X
        for _ in range(power):
            features = norm_adj @ features

        model_path = f'{data_path}/{model_W_file_name}.pkl'
        if os.path.exists(model_path):
            G, F, W, losses = optimize(features, n_clusters, node_feat_dim,
                                       max_iter=max_iter, tolerance=tolerance, model_path=model_path)

            metrics = output_metrics(features @ W, None, G)
            print(G)
            if losses is not None:
                run_metrics.append(metrics + [losses[-1]])
            #  G 是最终的 label，长度为节点个数，每个节点有一个 label 值
            with open(f'{data_path}/gcc_G.pkl', 'wb') as f:
                picklestring = pickle.dumps({
                    'G': G
                })
                f.write(picklestring)
                f.close()
        print_metrics(np.mean(run_metrics, 0), np.std(run_metrics, 0))
        return G

    def train_and_infer():
        best_W = None
        final_G = None
        for run in range(runs):
            features = X
            t0 = time.time()
            for _ in range(power):
                features = norm_adj @ features

            model_path = f'{data_path}/{model_W_file_name}.pkl'

            G, F, W, losses = optimize(features, n_clusters, n_clusters,
                                       max_iter=max_iter, tolerance=tolerance, model_path=None)
            best_W = W
            final_G = G

            time_it_took = time.time() - t0
            times.append(time_it_took)
            metrics = output_metrics(features @ W, None, G)
            print(G)
            print(G.shape)
            if losses is not None:
                run_metrics.append(metrics + [losses[-1]])

        with open(model_path, 'wb') as f:
            picklestring = pickle.dumps({
                'W': best_W
            })
            f.write(picklestring)
            f.close()

        with open(f'{data_path}/gcc_G.pkl', 'wb') as f:
            picklestring = pickle.dumps({
                'G': final_G
            })
            f.write(picklestring)
            f.close()
        print_metrics(np.mean(run_metrics, 0), np.std(run_metrics, 0), np.mean(times), np.std(times))
        return G

    # 是不是训练模式
    is_train = False

    if is_train:
        return train_and_infer()
    else:
        return infer()

    # tf.Tensor([1 1 2... 1 2 1], shape=(19717,), dtype=int32)
    # loss_mean: 78.80886924076933
    # acc_mean: 0.631941979002891
    # ari_mean: 0.2945981691646357
    # nmi_mean: 0.334546834909435
    # db_mean: 0.6004290520823548
    # sil_mean: 0.5652684946100452
    # f1_mean: 0.6264030015187884
    # loss_std: 0.0
    # acc_std: 0.0
    # ari_std: 0.0
    # nmi_std: 0.0
    # f1_std: 0.0
    # db_std: 0.0
    # sil_std: 0.0

    # time_mean: 4.156139838695526
    # loss_mean: 78.80886970990355
    # acc_mean: 0.6317898260384441
    # ari_mean: 0.29454291651639497
    # nmi_mean: 0.3344630622588723
    # db_mean: 0.6005319008862792
    # sil_mean: 0.565261714155149
    # f1_mean: 0.6262074988224887
    # time_std: 0.3947568556065051
    # loss_std: 8.987733679556355e-15
    # acc_std: 1.1102230246251565e-16
    # ari_std: 5.551115123125783e-17
    # nmi_std: 0.0
    # f1_std: 2.220446049250313e-16
    # db_std: 9.381220322469734e-16
    # sil_std: 2.1784148785229479e-16

# for run in range(runs):
#     features = X
#     t0 = time.time()
#     for _ in range(power):
#         features = norm_adj @ features
#
#     model_path = f'{data_path}/gcc_W.pkl'
#     if os.path.exists(model_path):
#         G, F, W, losses = optimize(features, n_clusters, n_clusters,
#                                    max_iter=max_iter, tolerance=tolerance, model_path=model_path)
#
#         metrics = output_metrics(features @ W, labels, G)
#         print(G)
#         if losses is not None:
#             run_metrics.append(metrics + [losses[-1]])
#         with open(f'{data_path}/gcc_G.pkl', 'wb') as f:
#             picklestring = pickle.dumps({
#                 'G': G
#             })
#             f.write(picklestring)
#             f.close()
#         break
#
#     G, F, W, losses = optimize(features, n_clusters, n_clusters,
#                                max_iter=max_iter, tolerance=tolerance, model_path=None)
#     with open(model_path, 'wb') as f:
#         picklestring = pickle.dumps({
#             'W': W
#         })
#         f.write(picklestring)
#         f.close()
#
#     time_it_took = time.time() - t0
#     times.append(time_it_took)
#     metrics = output_metrics(features @ W, labels, G)
#     print(G)
#     with open(f'{data_path}/gcc_G.pkl', 'wb') as f:
#         picklestring = pickle.dumps({
#             'G': G
#         })
#         f.write(picklestring)
#         f.close()
#     print(G.shape)
#     if losses is not None:
#         run_metrics.append(metrics + [losses[-1]])
#
# # print_metrics(np.mean(run_metrics, 0), np.std(run_metrics, 0), np.mean(times), np.std(times))
