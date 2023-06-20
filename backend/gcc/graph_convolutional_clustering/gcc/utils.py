import os.path
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


def aug_normalized_adjacency(adj, add_loops=True):
    if add_loops:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True):
    if add_loops:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def is_close(a, b, c):
    return np.abs(a - b) < c


def read_dataset(data_path, dataset):
    """
    W: 大小是(19717, 19717)的一个稀疏矩阵，表示文献之间的引用关系，如果文献 i 引用了文献 j，那么 W[i, j] = 1，否则为 0。
    gnd: 大小是(19717, 1), 一个向量，表示文献的真实类别标签，取值范围为 1 到 3，分别对应三个不同的主题。
    fea: 一个19717×500的一个稀疏矩阵，表示文献的特征向量，每一行对应一个文献，每一列对应一个词汇，元素值为词汇在文献中出现的次数
    """
    print(dataset)
    data = sio.loadmat(os.path.join(f'{data_path}/', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    adj = data['W']
    adj = adj.astype(float)
    if not sp.issparse(adj):
        adj = sp.csc_matrix(adj)
    if sp.issparse(features):
        features = features.toarray()
    labels = data['gnd']
    if labels is not None:
        labels = labels.reshape(-1) - 1
        n_classes = len(np.unique(labels))
    else:
        labels = None
        n_classes = 10  # todo: 先随便写一个分类数
    return adj, features, labels, n_classes


# def get_data_from_frontend(G, adj, features, labels=None):
#     adj = G.get_adj_matrix()
#     return adj, features, labels, 10


def preprocess_dataset(adj, features, row_norm=True, sym_norm=True, feat_norm=True, tf_idf=False):
    if sym_norm:
        adj = aug_normalized_adjacency(adj, True)
    if row_norm:
        adj = row_normalize(adj, True)

    # features = np.array(features)
    # features = features.astype(np.float)
    # features = csc_matrix(features)
    # if tf_idf:
    #     features = TfidfTransformer().fit_transform(features).toarray()
    # if feat_norm:
    #     features = normalize(features)
    return adj, features


def parse_logs(filename):
    import re
    with open(file=filename) as f:
        log = f.readlines()

    metrics_names = None
    metrics = []

    for line in log:
        if line[0:4] != 'time' and line[0:4] != 'loss': continue
        if metrics_names is None:

            metrics_names = [m.group(1) for m in re.finditer(r'(\w+):', line)]
            for _ in metrics_names:
                metrics.append([])

        metrics_values = [m.group(1) for m in re.finditer(r':([\d.e-]+)', line)]

        for i in range(len(metrics_values)):
            metrics[i].append(float(metrics_values[i]))
    metrics = np.array(metrics).T
    metrics = pd.DataFrame(metrics, columns=metrics_names, index=list(range(1, len(metrics)+1)))
    return metrics


