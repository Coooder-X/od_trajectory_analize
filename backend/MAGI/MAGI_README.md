# MAGI图聚类方法集成说明

## 📋 概述

MAGI (Modularity-Aware Graph clustering with contrastIve learning) 是2024年KDD会议发表的最新图聚类方法，结合了模块度最大化和图对比学习的优势。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_magi.txt
```

主要依赖：
- torch >= 1.7.0
- torch-geometric >= 2.0.0  
- scikit-learn >= 0.24.0
- scipy >= 1.6.0

### 2. 配置使用MAGI

在 `exp/graph_exp.py` 中设置：

```python
use_magi = True          # 启用MAGI方法
use_line_graph = False   # 使用原图（推荐）
# 或
use_line_graph = True    # 使用线图（需要轨迹特征）
```

### 3. 运行实验

```bash
cd exp
python graph_exp.py
```

## 🔧 配置选项

### 基本配置
- `use_magi = True/False`: 是否使用MAGI方法
- `cluster_num`: 预设的簇个数（支持任意K值）
- `use_line_graph`: 是否使用线图模式

### MAGI参数（在model/magi.py中可调整）
- `epochs`: 训练轮数（默认200）
- `lr`: 学习率（默认0.001）
- `hidden_dim`: 隐藏层维度（默认128）
- `tau`: 对比学习温度参数（默认0.5）

## 📊 支持的图规模

- ✅ **小规模图**: 10-1000节点
- ✅ **中等规模图**: 1K-100K节点  
- ✅ **大规模图**: 100K+节点（理论上支持到100M）

## 🎯 使用场景

### 1. 原图模式（推荐）
适用于一般的图聚类任务：
```python
use_magi = True
use_line_graph = False
```

### 2. 线图模式
适用于有轨迹特征的OD数据：
```python
use_magi = True
use_line_graph = True
```

## 🧪 测试

运行测试脚本验证安装：
```bash
python test_magi.py
```

测试包括：
- 34节点的Karate Club图
- 100节点的随机图
- 不同K值的聚类测试

## 📈 性能对比

| 方法 | 训练需求 | 可扩展性 | K值支持 | 性能 |
|------|----------|----------|---------|------|
| Louvain | ❌ | 高 | ❌ 自动 | 中等 |
| CNM | ❌ | 中等 | ❌ 自动 | 中等 |
| GCC | ✅ | 中等 | ✅ | 良好 |
| **MAGI** | ✅ | **很高** | ✅ | **优秀** |

## 🔍 输出说明

MAGI运行后会输出：
```
MAGI 社区发现结果: {0: [node1, node2], 1: [node3, node4], ...}
实际有效社区个数: X
====> 社区个数：K, CON = 0.XXXX
```

其中：
- 社区发现结果：每个簇包含的节点列表
- CON指标：聚类质量评估（越小越好）

## ⚠️ 注意事项

1. **GPU支持**: 如有GPU可加速训练，在model/magi.py中设置device='cuda'
2. **内存使用**: 大图建议分批处理或使用GPU
3. **收敛性**: 如果结果不稳定，可增加训练轮数
4. **特征选择**: 原图模式使用度特征，线图模式使用轨迹特征

## 🐛 故障排除

### 常见问题

1. **ImportError: torch_geometric**
   ```bash
   pip install torch-geometric
   ```

2. **CUDA out of memory**
   - 减少batch size或使用CPU
   - 设置device='cpu'

3. **聚类结果不稳定**
   - 增加训练轮数（epochs）
   - 调整学习率（lr）

### 获取帮助

如遇到问题，请检查：
1. 依赖是否正确安装
2. 输入数据格式是否正确
3. 参数设置是否合理

## 📚 参考文献

```bibtex
@inproceedings{liu2024magi,
  title={Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective},
  author={Liu, Yunfei and Li, Jintang and Chen, Yuehe and Wu, Ruofan and Wang, Ericbk and Zhou, Jing and Tian, Sheng and Shen, Shuheng and Fu, Xing and Meng, Changhua and Wang, Weiqiang and Chen, Liang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
