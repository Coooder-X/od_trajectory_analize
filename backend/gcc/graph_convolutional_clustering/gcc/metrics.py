from sklearn import metrics
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari


def ordered_confusion_matrix(y_true, y_pred):
    if y_true is None:
        return -1
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def clustering_accuracy(y_true, y_pred):
    if y_true is None:
        return -1
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    if y_true is None:
        return -1
    def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)


def output_metrics(X, y_true, y_pred):
    return [
        clustering_accuracy(y_true, y_pred),
        -1 if y_true is None else nmi(y_true, y_pred),
        -1 if y_true is None else ari(y_true, y_pred),
        clustering_f1_score(y_true, y_pred, average='macro'),
        davies_bouldin_score(X, y_pred),
        silhouette_score(X, y_pred)
    ]


def print_metrics(metrics_means, metrics_stds, time_mean=None, time_std=None):
    """
    loss_mean: 平均损失值，表示聚类模型的优化目标函数的值，一般来说，越小越好。
    acc_mean: 平均准确率，表示聚类结果和真实标签的一致程度，一般来说，越大越好。
    ari_mean: 平均调整兰德指数，表示聚类结果和真实标签的相似度，考虑了随机分配的影响，取值范围在 [-1, 1] 之间，一般来说，越大越好。
    nmi_mean: 平均归一化互信息，表示聚类结果和真实标签的信息共享程度，取值范围在 [0, 1] 之间，一般来说，越大越好。
    db_mean: 平均戴维森堡丁指数，表示聚类结果的紧密度和分离度的比值，一般来说，越小越好。
    sil_mean: 平均轮廓系数，表示聚类结果的内聚度和分离度的比值，取值范围在 [-1, 1] 之间，一般来说，越大越好。
    f1_mean: 平均 F1 值，表示聚类结果和真实标签的准确率和召回率的调和平均数，取值范围在 [0, 1] 之间，一般来说，越大越好。
    """
    if time_mean is not None: print(f'time_mean:{time_mean} ', end='')
    print(f'loss_mean:{metrics_means[6]} '
          f'acc_mean:{metrics_means[0]} '
          f'ari_mean:{metrics_means[2]} '
          f'nmi_mean:{metrics_means[1]} '
          f'db_mean:{metrics_means[4]} '
          f'sil_mean:{metrics_means[5]} '
          f'f1_mean:{metrics_means[3]} ', end=' ')

    if time_std is not None: print(f'time_std:{time_std} ', end='')
    print(f'loss_std:{metrics_stds[6]} '
          f'acc_std:{metrics_stds[0]} '
          f'ari_std:{metrics_stds[2]} '
          f'nmi_std:{metrics_stds[1]} '
          f'f1_std:{metrics_stds[3]} '
          f'db_std:{metrics_stds[4]} '
          f'sil_std:{metrics_stds[5]} ')
