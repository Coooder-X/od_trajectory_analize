# HLC（链接社区检测）在本项目中的使用说明

本说明文档介绍如何在本项目中使用链接社区检测（Hierarchical Link Clustering, HLC）方法，及其在 `backend/exp5.py` 中的集成逻辑与参数开关。

- 代码入口：`backend/exp5.py`
- HLC 工具函数：`backend/HLC/hlc_utils.py`
- 依赖（可选）：`cdlib`（本项目当前使用 0.4.0 版本 API）

---

## 一、方法概述（HLC）

HLC 是一种“以边为对象”的社群检测方法：
- 在原图上计算“边-边相似度”（常用 Jaccard/Tanimoto），对边进行层次聚类；
- 在树上截断得到“边社区”。边社区天然支持重叠（一个节点可同时出现在多个边社区中）。

本项目在获得“边社区”后，会将其映射成“节点集合”（用于与其它方法统一评估），并可选地对碎片化社区进行合并（后处理），以获得更可控的社区数量。

参考：Ahn, Bagrow, Lehmann (2010) Link Communities.

---

## 二、项目中相关文件与函数

- `backend/exp5.py`
  - 开关：
    - `use_cdlib_link_community`: 是否启用 CDlib 的 HLC 分支。
    - `need_merge_hlc_cluster`: 是否执行“后处理合并”（将碎片化小社区合并到目标簇数）。
    - `need_filter_small_edge`: 是否在映射前按边数阈值过滤掉过小的边社区（降噪）。
    - `cdlib_min_edges`: 过滤阈值（当 `need_filter_small_edge=True` 时生效）。
    - `cdlib_merge_max_iter`、`cdlib_merge_min_overlap`: 合并强度与上限。
  - 评估：
    - `avg_CON(G, cluster_point_dict, node_name_cluster_dict, use_igraph)`：用 CON 指标评估社区边界比例。
    - `get_ok_cluster_num_for_line_graph_cdlib(cluster_point_dict)`：HLC 的“有效社区个数”（边社区≥2条边）统计。

- `backend/HLC/hlc_utils.py`
  - `run_cdlib_hlc(cdlib_input_graph)`：
    - 统一封装 CDlib HLC 的调用，兼容 0.4.0 的 `hierarchical_link_community` 与（若存在）`hierarchical_link_community_full`。
  - `detect_line_graph_nodes(g)`：
    - 判断当前图 `g` 的节点是否为“边 tuple”（即是否为线图）。
  - `map_edge_communities_to_nodes(edge_comms, g, is_line_graph_nodes, current_nodes)`：
    - 将边社区映射为节点社区。
      - 若 `g` 是线图：边 tuple 即为线图节点。
      - 若 `g` 是原图：取边社区中所有边的端点并集作为节点社区。
  - `merge_clusters_to_target(cluster_point_dict, target_k, max_iter=300, min_overlap=0.0)`：
    - 基于 Jaccard 重叠度的迭代合并策略，将最小簇并入与其重叠度最高的簇，直至达到目标社区数或达到迭代上限。

---

## 三、在 `exp5.py` 中的执行流程

1) 选择输入图：
- 当 `use_line_graph=False` 时，`cdlib_input_graph` = 原图（`networkx` 图）。
- 当 `use_line_graph=True` 时，仍保留原图 `original_g_nx`，HLC 以原图为输入，`g` 用于后续评估与映射。

> 建议设置 `use_line_graph=False`，

2) 运行 HLC：
- 调用 `ec = run_cdlib_hlc(cdlib_input_graph)` 获取边社区集合 `ec.communities`。

3) 过滤边社区（可选，降噪）：
- 当 `need_filter_small_edge=True` 时，按 `cdlib_min_edges` 过滤小边社区；否则直接使用原始边社区：
  ```python
  raw_edge_comms = list(ec.communities)
  edge_comms = (
      [c for c in raw_edge_comms if len(c) >= cdlib_min_edges]
      if need_filter_small_edge else raw_edge_comms
  )
  ```

4) 映射：将“边社区”映射为“节点社区”（与项目评估格式一致）：
- `is_line_graph_nodes, current_nodes = detect_line_graph_nodes(g)`
- `cluster_point_dict, node_name_cluster_dict = map_edge_communities_to_nodes(edge_comms, g, is_line_graph_nodes, current_nodes)`
  - `cluster_point_dict`: 社区 id -> 节点列表
  - `node_name_cluster_dict`: 节点 -> 社区 id

5) 后处理（可选）：当 `need_merge_hlc_cluster=True` 时执行合并：
- 若 `len(cluster_point_dict) > cluster_num`，则：
  ```python
  cluster_point_dict, node_name_cluster_dict = merge_clusters_to_target(
      cluster_point_dict,
      target_k=cluster_num,            # 或者可改为 cdlib_target_k
      max_iter=cdlib_merge_max_iter,
      min_overlap=cdlib_merge_min_overlap,
  )
  ```
- 该合并是项目层的“工程化增强”，因为实验需要指定簇数k进行，而 hlc 不支持指定簇数k。在这种情况下它分出来的社区质量不高，比如分出了10+、20+个社区，但有效社区（社区大小大于1）只有几个。而我们的线图方法的表格里，已有的方法是支持指定k的，且将k定为5-50，因此这里新增了合并算法，让最后的聚类个数等于k，不属于 2010 年 HLC 原论文的一部分。

6) 评估与日志：
- `print('CDlib 链接社区结果: ', cluster_point_dict)`
- `final_cluster_num = len(cluster_point_dict.keys())`
- `exp5_log.append(f'CDlib链接社区，设定k={cluster_num} 实际有效社区个数: {get_ok_cluster_num_for_line_graph_cdlib(cluster_point_dict)}')`
- `print(f"====> 社区个数：{final_cluster_num}, CON = {avg_CON(g, cluster_point_dict, node_name_cluster_dict, use_igraph)}")`

---

## 四、关键开关与参数说明

- **`use_cdlib_link_community`**：是否启用 HLC 分支。
- **`need_merge_hlc_cluster`**：
  - False：不执行合并；社区数量不向上“增补”。
  - True：执行合并，将簇数向本轮 `cluster_num` 靠拢（仅向下合并，不做“裂解/细分”）。

**`need_filter_small_edge`**：
  - False：不过滤，直接使用 HLC 原始边社区。
  - True：过滤边数小于 `cdlib_min_edges` 的边社区（降噪，通常会减少社区数量）。
- **`cdlib_min_edges`**：过滤边数过小的边社区，常用 2～4，越大越“保守”。
- **`cdlib_merge_max_iter`**：合并迭代上限，防止长时间运行。
- **`cdlib_merge_min_overlap`**：合并的最小 Jaccard 重叠阈值。
  - 设为 0.0：即使无重叠也会强制合并到最大簇，保证能收敛到目标簇数。
  - 提高阈值会更保守，但可能无法达到目标簇数。

---

## 五、运行与常见问题

- 运行：直接执行 `backend/exp5.py`，HLC 分支在 for-loop 内被调用（当 `use_cdlib_link_community=True`）。
- 常见日志与排查：
  - `CDlib 链接社区检测失败: ...`：查看异常内容，可能是 CDlib 未安装或 API 差异；项目已做了 0.4.0 的兼容。
  - `有效的社区个数为0，无法计算 CON`：
    - 说明映射后社区为空或“有效社区”（默认节点数>5）数量为 0。
    - 可适当下调 `cdlib_min_edges`、或开启后处理并降低 `cdlib_merge_min_overlap`。
  - 若 `use_line_graph=True`：`g` 可能为线图，映射时会将边 tuple 当作节点；`avg_CON()` 会用当前 `g` 配合 `node_name_cluster_dict` 计算。

---

## 六、与其它方法的关系

- HLC 是“边社群检测”，与基于“节点社群检测”的 Louvain/CNM/MAGI 不同。

---

## 七、建议的参数起点

- `cdlib_min_edges = 2`
- `need_filter_small_edge = True` 建议置为 True，
- `need_merge_hlc_cluster = True`（也可以置为 False，先看原始输出）→ 若簇数过多、有效社区过少，再设为 True；
- `cdlib_merge_min_overlap = 0.0`（保证能合并到目标），随后逐步提高到 0.05～0.2 以获得更“保守”的合并。

---

## 八、快速检查清单

- [ ] `use_cdlib_link_community=True` 是否启用
- [ ] `need_merge_hlc_cluster` 是否符合预期
- [ ] `need_filter_small_edge` 是否应开启（与 `cdlib_min_edges` 配合）
- [ ] `cdlib_min_edges` 是否合理（太大可能过滤掉全部社区）
- [ ] 评估 `CON` 之前，`cluster_point_dict` 是否为空
- [ ] Node/Edge 映射是否与当前图 `g` 的类型一致（原图 vs 线图）