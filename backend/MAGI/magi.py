"""
MAGI: Modularity-Aware Graph Clustering with Contrastive Learning
Based on "Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective" (KDD 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix


class GraphEncoder(nn.Module):
    """Graph Convolutional Network Encoder"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MAGI(nn.Module):
    """MAGI: Modularity-Aware Graph Clustering"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_clusters=4, 
                 num_layers=2, dropout=0.1, tau=0.5):
        super(MAGI, self).__init__()
        
        self.num_clusters = num_clusters
        self.tau = tau  # temperature parameter for contrastive learning
        
        # Graph encoder
        self.encoder = GraphEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # Clustering layer
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, output_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)
        
    def forward(self, x, edge_index, adj_matrix=None):
        # Get node embeddings
        z = self.encoder(x, edge_index)
        
        # Compute cluster assignments
        q = self.get_cluster_prob(z)
        
        return z, q
    
    def get_cluster_prob(self, z):
        """Compute cluster assignment probabilities"""
        # Compute distances to cluster centers
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer.unsqueeze(0)) ** 2, dim=2)
        
        # Convert to probabilities using Student's t-distribution
        alpha = 1.0
        q = 1.0 / (1.0 + dist / alpha)
        q = q ** ((alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q):
        """Compute target distribution P"""
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def modularity_loss(self, z, adj_matrix, cluster_assignments):
        """Compute modularity-based contrastive loss"""
        if adj_matrix is None:
            return torch.tensor(0.0, device=z.device)
            
        # Convert cluster assignments to hard assignments
        hard_assignments = torch.argmax(cluster_assignments, dim=1)
        
        # Compute modularity matrix
        A = adj_matrix.to_dense() if hasattr(adj_matrix, 'to_dense') else adj_matrix
        k = A.sum(dim=1)  # degree vector
        m = A.sum() / 2   # total edges
        
        if m == 0:
            return torch.tensor(0.0, device=z.device)
            
        # Modularity matrix B = A - k*k^T/(2m)
        B = A - torch.outer(k, k) / (2 * m)
        
        # Compute modularity for current clustering
        modularity = 0.0
        for c in range(self.num_clusters):
            mask = (hard_assignments == c)
            if mask.sum() > 0:
                modularity += B[mask][:, mask].sum()
        
        modularity = modularity / (2 * m)
        
        # Convert to loss (negative modularity)
        return -modularity
    
    def contrastive_loss(self, z, adj_matrix):
        """Compute contrastive loss based on graph structure"""
        if adj_matrix is None:
            return torch.tensor(0.0, device=z.device)
            
        # Normalize embeddings
        z_norm = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.tau
        
        # Create positive and negative masks based on adjacency
        A = adj_matrix.to_dense() if hasattr(adj_matrix, 'to_dense') else adj_matrix
        pos_mask = A > 0
        neg_mask = A == 0
        
        # Remove self-loops
        eye = torch.eye(A.size(0), device=A.device).bool()
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # Compute contrastive loss
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[neg_mask]
        
        if len(neg_sim) == 0:
            return torch.tensor(0.0, device=z.device)
            
        # InfoNCE-style loss
        pos_loss = -torch.log(torch.exp(pos_sim).sum() + 1e-8)
        neg_loss = torch.log(torch.exp(neg_sim).sum() + 1e-8)
        
        return pos_loss + neg_loss


def scipy_sparse_to_torch(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def run_magi(adj_matrix, features, num_clusters, device='cpu', epochs=200, lr=0.001, 
             modularity_weight=0.5, contrastive_weight=0.3):
    """
    Run MAGI clustering algorithm
    
    Args:
        adj_matrix: scipy sparse matrix or torch tensor
        features: node features (numpy array or torch tensor)
        num_clusters: number of clusters
        device: computation device
        epochs: training epochs
        lr: learning rate
    
    Returns:
        cluster_labels: numpy array of cluster assignments
    """
    
    # Convert inputs to torch tensors
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    if sp.issparse(adj_matrix):
        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        adj_tensor = scipy_sparse_to_torch(adj_matrix)
    else:
        # Assume adj_matrix is already a torch tensor
        adj_tensor = adj_matrix
        # Create edge_index from adjacency matrix
        edge_index = torch.nonzero(adj_tensor).t().contiguous()
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    adj_tensor = adj_tensor.to(device)
    
    # Initialize model
    input_dim = features.size(1)
    model = MAGI(input_dim=input_dim, num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize cluster centers with k-means (多次尝试获得更好的初始化)
    with torch.no_grad():
        z, _ = model(features, edge_index)
        best_inertia = float('inf')
        best_centers = None
        
        # 尝试多次k-means初始化，选择最好的
        for i in range(10):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42+i, n_init=10)
            y_pred = kmeans.fit_predict(z.cpu().numpy())
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
        
        model.cluster_layer.data = torch.tensor(best_centers).to(device)
        print(f"最佳初始化惯性: {best_inertia:.4f}")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        z, q = model(features, edge_index, adj_tensor)
        
        # Compute target distribution
        p = model.target_distribution(q)
        
        # Compute losses
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        modularity_loss = model.modularity_loss(z, adj_tensor, q)
        contrastive_loss = model.contrastive_loss(z, adj_tensor)
        
        # Total loss - 使用可调整的权重
        total_loss = kl_loss + modularity_weight * modularity_loss + contrastive_weight * contrastive_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            # 检查当前聚类质量
            with torch.no_grad():
                _, q_check = model(features, edge_index, adj_tensor)
                current_labels = torch.argmax(q_check, dim=1).cpu().numpy()
                unique_clusters = len(set(current_labels))
            
            print(f'Epoch {epoch}: Loss = {total_loss.item():.4f}, '
                  f'KL = {kl_loss.item():.4f}, '
                  f'Modularity = {modularity_loss.item():.4f}, '
                  f'Contrastive = {contrastive_loss.item():.4f}, '
                  f'Clusters = {unique_clusters}/{num_clusters}')
    
    # Get final cluster assignments
    model.eval()
    with torch.no_grad():
        _, q = model(features, edge_index, adj_tensor)
        cluster_labels = torch.argmax(q, dim=1).cpu().numpy()
    
    return cluster_labels


if __name__ == "__main__":
    # Test with synthetic data
    import networkx as nx
    
    # Create a test graph
    G = nx.karate_club_graph()
    adj = nx.adjacency_matrix(G)
    features = np.random.randn(G.number_of_nodes(), 10)
    
    # Run MAGI
    labels = run_magi(adj, features, num_clusters=2)
    print("Cluster labels:", labels)
