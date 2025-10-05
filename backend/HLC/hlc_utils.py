try:
    from cdlib import algorithms as cdlib_algorithms
    _cdlib_available = True
except Exception:
    _cdlib_available = False

# ==== Helpers: CDlib HLC runner & graph type detection ===========================================
def run_cdlib_hlc(cdlib_input_graph):
    """
    统一封装对不同版本 CDlib 的 HLC 调用：
    - 优先尝试 hierarchical_link_community_full(G, weight=None, simthr=None)
    - 回退到 hierarchical_link_community_full(G)
    - 再回退到 hierarchical_link_community(G, weight=None, simthr=None)
    - 最后 hierarchical_link_community(G)

    返回：ec（CDlib 的 Community object）
    """
    if not _cdlib_available:
        raise ImportError('CDlib 未安装，无法运行链接社区检测')
    # 兼容不同版本CDlib：优先使用 full，不存在或签名不兼容则回退到基础版本/无参数
    if hasattr(cdlib_algorithms, 'hierarchical_link_community_full'):
        try:
            ec = cdlib_algorithms.hierarchical_link_community_full(cdlib_input_graph, weight=None, simthr=None)
            print('CDlib 使用了新版本的 full 新方法: hierarchical_link_community_full')
            return ec
        except TypeError:
            ec = cdlib_algorithms.hierarchical_link_community_full(cdlib_input_graph)
            print('CDlib 使用了新版本的 full 新方法，但签名不兼容，回退到基础版本: hierarchical_link_community_full')
            return ec
    else:
        try:
            ec = cdlib_algorithms.hierarchical_link_community(cdlib_input_graph, weight=None, simthr=None)
            print('CDlib 使用了基础版本的旧方法: hierarchical_link_community')
            return ec
        except TypeError:
            ec = cdlib_algorithms.hierarchical_link_community(cdlib_input_graph)
            print('CDlib 使用了基础版本的旧方法，但签名不兼容，回退到基础版本: hierarchical_link_community')
            return ec


def detect_line_graph_nodes(g):
    """返回 (is_line_graph_nodes: bool, current_nodes: set)"""
    current_nodes = set(g.nodes())
    is_line_graph_nodes = False
    if len(current_nodes) > 0:
        sample_node = next(iter(current_nodes))
        is_line_graph_nodes = isinstance(sample_node, tuple) and len(sample_node) in (2, 3)
    return is_line_graph_nodes, current_nodes


def map_edge_communities_to_nodes(edge_comms, g, is_line_graph_nodes, current_nodes):
    """
    将边社区列表映射为节点社区：
    - 若 g 是线图：每条边tuple即为线图节点，直接作为社区节点；需存在于 current_nodes
    - 若 g 是原图：取边端点的并集作为节点社区

    返回 (cluster_point_dict, node_name_cluster_dict)
    """
    cluster_point_dict = {}
    node_name_cluster_dict = {}
    for i, edge_comm in enumerate(edge_comms):
        nodes_in_comm = []
        for e in edge_comm:
            if is_line_graph_nodes:
                if e in current_nodes:
                    nodes_in_comm.append(e)
            else:
                if isinstance(e, tuple) and len(e) >= 2:
                    u, v = e[0], e[1]
                    nodes_in_comm.append(u)
                    nodes_in_comm.append(v)
        if not is_line_graph_nodes:
            nodes_in_comm = list(set(nodes_in_comm))
        if len(nodes_in_comm) > 0:
            cluster_point_dict[i] = nodes_in_comm
            for n in nodes_in_comm:
                node_name_cluster_dict[n] = i
    return cluster_point_dict, node_name_cluster_dict


def merge_clusters_to_target(cluster_point_dict, target_k, max_iter=300, min_overlap=0.0):
    """
    按 Jaccard 重叠度将小簇迭代合并到相似度最高的大簇，直到簇数<=target_k 或达到 max_iter。
    返回合并后的 (cluster_point_dict, node_name_cluster_dict)
    """
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        if inter == 0:
            return 0.0
        union = len(sa | sb)
        return inter / union if union > 0 else 0.0

    def _rebuild_index(cpd):
        new_cpd = {}
        for new_id, old_id in enumerate(sorted(cpd.keys())):
            new_cpd[new_id] = cpd[old_id]
        return new_cpd

    iter_cnt = 0
    while len(cluster_point_dict) > target_k and iter_cnt < max_iter:
        smallest_id = min(cluster_point_dict.keys(), key=lambda k: len(cluster_point_dict[k]))
        best_id, best_sim = None, -1.0
        for cid, nodes in cluster_point_dict.items():
            if cid == smallest_id:
                continue
            sim = _jaccard(cluster_point_dict[smallest_id], nodes)
            if sim > best_sim:
                best_id, best_sim = cid, sim
        if best_id is None or best_sim < min_overlap:
            best_id = max(
                cluster_point_dict.keys(),
                key=lambda k: len(cluster_point_dict[k]) if k != smallest_id else -1,
            )
        merged = list(set(cluster_point_dict[best_id]) | set(cluster_point_dict[smallest_id]))
        cluster_point_dict[best_id] = merged
        del cluster_point_dict[smallest_id]
        iter_cnt += 1

    cluster_point_dict = _rebuild_index(cluster_point_dict)
    node_name_cluster_dict = {}
    for cid, nodes in cluster_point_dict.items():
        for n in nodes:
            node_name_cluster_dict[n] = cid
    return cluster_point_dict, node_name_cluster_dict