"""
测试MAGI方法的简单脚本
"""

import numpy as np
import networkx as nx
from MAGI.magi import run_magi
import torch

def test_magi_simple():
    """测试MAGI方法的基本功能"""
    print("=== 测试MAGI方法 ===")
    
    # 创建一个简单的测试图
    G = nx.karate_club_graph()  # 34个节点的经典测试图
    print(f"测试图节点数: {G.number_of_nodes()}")
    print(f"测试图边数: {G.number_of_edges()}")
    
    # 获取邻接矩阵
    adj_matrix = nx.adjacency_matrix(G)
    
    # 创建简单的节点特征（度 + 随机特征）
    degrees = dict(G.degree())
    node_list = list(G.nodes())
    features = []
    
    for node in node_list:
        # 10维特征：度 + 9个随机特征
        feat = [degrees[node]] + [np.random.random() for _ in range(9)]
        features.append(feat)
    
    features = np.array(features)
    print(f"特征矩阵形状: {features.shape}")
    
    # 测试不同的簇个数
    for k in [2, 3, 4]:
        print(f"\n--- 测试 k={k} ---")
        try:
            # 运行MAGI
            labels = run_magi(adj_matrix, features, num_clusters=k, epochs=100)
            
            print(f"聚类结果: {labels}")
            print(f"实际簇个数: {len(set(labels))}")
            
            # 统计每个簇的大小
            cluster_sizes = {}
            for label in labels:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            print(f"簇大小分布: {cluster_sizes}")
            
        except Exception as e:
            print(f"运行MAGI时出错: {e}")
            import traceback
            traceback.print_exc()

def test_magi_small_graph():
    """测试100节点规模的图"""
    print("\n=== 测试100节点图 ===")
    
    # 创建一个100节点的随机图
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    print(f"测试图节点数: {G.number_of_nodes()}")
    print(f"测试图边数: {G.number_of_edges()}")
    
    # 获取邻接矩阵
    adj_matrix = nx.adjacency_matrix(G)
    
    # 创建节点特征
    degrees = dict(G.degree())
    node_list = list(G.nodes())
    features = []
    
    for node in node_list:
        feat = [degrees[node]] + [np.random.random() for _ in range(9)]
        features.append(feat)
    
    features = np.array(features)
    
    # 测试k=5的聚类
    k = 5
    print(f"运行MAGI聚类，k={k}")
    
    try:
        labels = run_magi(adj_matrix, features, num_clusters=k, epochs=50)
        
        print(f"聚类完成，实际簇个数: {len(set(labels))}")
        
        # 统计簇大小
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        print(f"簇大小分布: {cluster_sizes}")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查PyTorch是否可用
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行测试
    test_magi_simple()
    test_magi_small_graph()
    
    print("\n=== 测试完成 ===")
