import os.path
import pickle

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def update_rule_F(XW, G, k):
    F = tf.math.unsorted_segment_mean(XW, G, k)
    return F


def update_rule_W(X, F, G):
    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    W = U @ tf.transpose(V)
    return W


def update_rule_G(XW, F):
    centroids_expanded = F[:, None, ...]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), 2)
    G = tf.math.argmin(distances, 0, output_type=tf.dtypes.int32)
    return G


def init_G_F(XW, k):
    km = KMeans(k).fit(XW)
    G = km.labels_
    F = km.cluster_centers_
    return G, F


def init_W(X, f):
    pca = PCA(f, svd_solver='randomized').fit(X)
    W = pca.components_.T
    return W


'''
在这个项目中，这个是一个训练循环的函数，它使用了tf.function装饰器，这意味着它会被转换成一个TensorFlow的图1。这个函数的参数是：

X: 一个节点特征矩阵，形状为(n, d)，其中n是节点数，d是特征维度。
F: 一个节点隐含表示矩阵，形状为(n, k)，其中k是聚类数。
G: 一个节点聚类分配向量，形状为(n,)，其中每个元素是一个整数，表示节点属于哪个聚类。
W: 一个权重矩阵，形状为(d, k)，用于将节点特征映射到隐含表示空间。
k: 一个整数，表示聚类数。
max_iter: 一个整数，表示最大迭代次数。
tolerance: 一个浮点数，表示收敛的容忍度。
这个函数的作用是使用图卷积网络和交替最小化算法来同时学习节点的隐含表示和聚类分配。它的主要步骤是：

使用update_rule_W函数来更新权重矩阵W。
使用update_rule_G函数来更新节点聚类分配向量G。
使用update_rule_F函数来更新节点隐含表示矩阵F。
计算损失函数，即节点特征和节点隐含表示的重构误差的范数。
判断是否达到最大迭代次数或收敛条件，如果是，则停止循环。
这个函数的返回值是：

G: 更新后的节点聚类分配向量。
F: 更新后的节点隐含表示矩阵。
W: 更新后的权重矩阵。
losses: 一个张量数组，包含每次迭代的损失值。
'''
@tf.function
def train_loop(X, F, G, W, k, max_iter, tolerance):
    losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    prev_loss = tf.float64.max

    for i in tf.range(max_iter):

        W = update_rule_W(X, F, G)
        XW = X @ W
        G = update_rule_G(XW, F)
        F = update_rule_F(XW, G, k)

        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
        if prev_loss - loss < tolerance:
            break

        losses = losses.write(i, loss)
        prev_loss = loss

    return G, F, W, losses.stack()


def optimize(X, k, f, max_iter=30, tolerance=10e-7, model_path=None):
    """
    X：一个矩阵，表示文献的特征向量，也就是fea字段的值
    k：一个整数，表示聚类的个数，也就是要把文献分成几类
    f：一个整数，表示降维后的特征维度，也就是要把文献的特征向量压缩到多少维
    max_iter：一个整数，表示最大的迭代次数，也就是要进行多少轮的训练
    tolerance：一个浮点数，表示收敛的阈值，也就是当训练损失小于这个值时停止训练
    这个函数的作用是用图卷积聚类算法对文献进行聚类，并返回聚类结果和训练过程中的损失历史。
    """
    if model_path is not None:  # 推理模式
        print('exists trained W')
        with open(model_path, 'rb') as f:
            obj = pickle.loads(f.read())
            W = obj['W']

            losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
            G, F = init_G_F(X @ W, k)
            XW = X @ W
            G = update_rule_G(XW, F)
            F = update_rule_F(XW, G, k)
            loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
            losses = losses.write(0, loss)
            f.close()
        return G, F, W, losses.stack()
    else:  # 训练模式
        # init G and F
        W = init_W(X, f)
        G, F = init_G_F(X @ W, k)
        G, F, W, loss_history = train_loop(X, F, G, W, k, max_iter, tolerance)

        return G, F, W, loss_history
