#coding:utf-8

import argparse
import os
import datetime, time
# from train import train, train_cluster
# from train_v2 import train, train_cluster
# from evaluate import evaluator, t2vec, t2vec_cluster
# from resultVisPre import showPic as showPrePic
# from resultVis import showPic as showt2vecPic

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="data",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="checkpoint.pt",
    help="The saved checkpoint (only neural network)")

parser.add_argument("-best_model", default="best_model.pt",
    help="The saved best_model (only neural network)")

parser.add_argument("-best_cluster_model", default="best_cluster_model.pt",
    help="The saved best model combined neural network with cluster model")

parser.add_argument("-cluster_model", default="cluster_model.pt",
    help="The saved last model combined neural network with cluster model")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=True,
    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.1,
    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-m2_learning_rate", type=float, default=0.008)

parser.add_argument("-batch", type=int, default=128, # 256
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=32,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=128, # 256
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=40,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=40,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=False,
    help="True if we use GPU to train the model")

parser.add_argument("-criterion_name", default="KLDIV",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default="data/hangzhou-vocab-dist-cell75-1.h5", # None
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0,
    help="Running mode (0: train, 1:evaluate, 2:t2vec)")

parser.add_argument("-vocab_size", type=int, default=11483,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(30, 30), (50, 50)],
    help="Bucket size for training")

parser.add_argument("-clusterNum", type=int, default=19*24, # 聚类簇的数目
                    help="cluster number of KMeans algorithm")

parser.add_argument("-alpha", type=int, default=2, # 4
                    help="coefficient of reconstruction loss")

parser.add_argument("-beta", type=int, default=0.01,
                    help="coefficient of clustering loss")

parser.add_argument("-gamma", type=int, default=0.1,
                    help="coefficient of distance loss between centroids")

parser.add_argument("-delta", type=int, default=0.1,
                    help="coefficient of neighbor loss between datas")

parser.add_argument("-sourcedata", default='preprocessing/make_data/20200101_jianggan.h5',
                    help="source data and label")

parser.add_argument("-devices", default=[0, 1, 2, 3, 4, 5, 6, 7],
# parser.add_argument("-devices", default=[0, 1, 2, 3],
# parser.add_argument("-devices", default=[4, 5, 7],
    help="Bucket size for training")

parser.add_argument("-expId", default=1,
    help="Bucket size for training")

parser.add_argument("-device", default=-1,
    help="Bucket size for training")

parser.add_argument("-save", default=True,
    help="tsne png save")

parser.add_argument("-hasLabel", default=True,
    help="是否有标签数据")

parser.add_argument("-kmeans", default=0,
    help="是否使用KmeansLoss")

args = parser.parse_args()


## __main__
# args.bucketsize = [(0,20),(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)]
args.bucketsize = [(100000, 100000)]  # 把src和trg的数据长度都不超过100000的数据存在同一个列表中
args.bucketsize = [(100000, 100000)]  # 把src和trg的数据长度都不超过100000的数据存在同一个列表中
for dirpath,dirnames,filenames in os.walk(args.data):
    for file in filenames:
        file_type = file.split('.')[-1]
        if (file_type == "h5"):
            args.knearestvocabs = os.path.join(dirpath, file)  # 文件全名
            break
args.expId = 1
s = str.split(args.checkpoint, '.')
args.checkpoint = s[0] + '_' + str(args.expId) + "." + s[1]
s = str.split(args.best_model, '.')
args.best_model = s[0] + '_' + str(args.expId) + "." + s[1]
s = str.split(args.best_cluster_model, '.')
args.best_cluster_model = s[0] + '_' + str(args.expId) + "." + s[1]
s = str.split(args.cluster_model, '.')
args.cluster_model = s[0] + '_' + str(args.expId) + "." + s[1]
print("execute exp of {} file name of best_model is {} ".format(args.expId, args.best_model))

args.device = args.devices[(args.expId - 1) % 8]
# args.device = args.devices[(args.expId - 1) % 3]
# args.device = 0

args.save = True
args.mode = 5
args.criterion_name = 'NLL'  #"KLDIV_cluster"
args.vocab_size = 8032
args.clusterNum = 80
args.hasLabel = True
args.embedding_size = 512 # 嵌入层输出 128
args.hidden_size = 256 # featur维度 64.
args.batch = 128 # 32可
args.t2vec_batch = 128 # 32
args.save_freq = 40
args.dropout = 0.1 # 0.3
args.learning_rate = 0.002 
args.m2_learning_rate = 0.008
args.dist_decay_speed = 0.8 # 对远距离cell的惩罚值，也就是theta的倒数，越大惩罚越大
args.alpha = 1# 10 # 控制重构
args.beta = 1 # 控制聚类损失函数
args.gamma = 1 # 簇间
args.delta = 1# 邻居
args.kmeans = 0
# args.sourcedata = 'preprocessing/experiment_data/hzd2zjg_reorder_3_inte.h5'
args.sourcedata = 'pre_data/*.h5'
args.epochs = 100
print(args)

start_dt = datetime.datetime.now()
start_t = time.time()
print("START DATETIME")
print(start_dt)
# try:
# if args.mode == 1:
#     evaluator(args)
# elif args.mode == 2:
#     # 对预训练的结果进行聚类
#     t2vec(args)
# elif args.mode == 3:
#     t2vec_cluster(args)
# elif args.mode == 4:
#     # 联合训练
#     if os.path.exists(args.cluster_model):
#         os.remove(args.cluster_model)
#         os.remove(args.best_cluster_model)
#     train_cluster(args)
# elif args.mode == 5:
#     # 可视化结果
#     # showt2vecPic(args)
#     showPrePic(args)
# else:
#     # 预训练
#     if os.path.exists(args.best_model):
#         os.remove(args.best_model)
#     train(args)
#