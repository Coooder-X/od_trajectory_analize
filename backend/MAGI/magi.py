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

    def center_separation_loss(self, sigma: float = 1.0):
        """Encourage cluster centers to be separated.
        loss = mean_{i<j} exp(-||ci-cj||^2 / sigma)
        Minimizing this pushes centers apart.
        """
        C = self.cluster_layer
        k = C.size(0)
        if k <= 1:
            return torch.tensor(0.0, device=C.device)
        # Pairwise squared distances
        dist2 = torch.cdist(C, C, p=2.0) ** 2  # [k,k]
        # Use upper triangle without diagonal
        triu_mask = torch.triu(torch.ones_like(dist2, dtype=torch.bool), diagonal=1)
        vals = torch.exp(-dist2[triu_mask] / max(sigma, 1e-8))
        if vals.numel() == 0:
            return torch.tensor(0.0, device=C.device)
        return vals.mean()
    
    def target_distribution(self, q):
        """Compute target distribution P"""
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def modularity_loss(self, z, adj_matrix, cluster_assignments):
        """Differentiable modularity loss using soft assignments.
        Uses S = q (n x k) to compute Q = trace(S^T B S)/(2m).
        """
        if adj_matrix is None:
            return torch.tensor(0.0, device=z.device)

        # Compute modularity matrix
        A = adj_matrix.to_dense() if hasattr(adj_matrix, 'to_dense') else adj_matrix
        A = A.float()
        # Symmetrize for modularity (undirected assumption)
        A = 0.5 * (A + A.t())
        k = A.sum(dim=1)  # degree vector
        m = A.sum() / 2.0

        if m.item() == 0.0:
            return torch.tensor(0.0, device=z.device)

        # Modularity matrix B = A - k*k^T/(2m)
        B = A - torch.outer(k, k) / (2.0 * m)

        # Soft community assignment matrix S = q
        S = cluster_assignments  # [n, k]

        # Q = trace(S^T B S) / (2m)
        BS = torch.matmul(B, S)
        ST_B_S = torch.matmul(S.transpose(0, 1), BS)
        modularity = torch.trace(ST_B_S) / (2.0 * m)

        # Negative modularity is the loss
        return -modularity
    
    def contrastive_loss(self, z, adj_matrix):
        """Node-wise InfoNCE contrastive loss.
        For each node i: positives are neighbors (A_ij>0), negatives are others.
        L = - 1/N sum_i log( sum_{j in P(i)} exp(sim(i,j)/tau) / sum_{k!=i} exp(sim(i,k)/tau) )
        """
        if adj_matrix is None:
            return torch.tensor(0.0, device=z.device)

        A = adj_matrix.to_dense() if hasattr(adj_matrix, 'to_dense') else adj_matrix
        A = A.float()
        N = A.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=z.device)

        # Normalize embeddings and compute sim matrix
        z_norm = F.normalize(z, dim=1)
        sim = torch.mm(z_norm, z_norm.t()) / self.tau

        eye = torch.eye(N, device=A.device, dtype=torch.bool)
        pos_mask = (A > 0) & (~eye)
        all_mask = ~eye

        # Avoid nodes with no positives
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Numerator: sum exp(sim_ij) over positives
        exp_sim = torch.exp(sim)
        numer = (exp_sim * pos_mask).sum(dim=1)
        # Denominator: sum exp(sim_ik) over all k!=i
        denom = (exp_sim * all_mask).sum(dim=1) + 1e-8

        per_node_loss = -torch.log((numer + 1e-8) / denom)
        loss = per_node_loss[valid].mean()
        return loss


def scipy_sparse_to_torch(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def run_magi(adj_matrix, features, num_clusters, device='cpu', epochs=200, lr=0.001,
             modularity_weight=0.5, contrastive_weight=0.3, balance_weight=0.0,
             center_sep_weight=0.0, center_sep_sigma=1.0,
             empty_cluster_threshold=0.005, empty_cluster_reinit=True,
             final_kmeans=False,
             hidden_dim=128, output_dim=64, num_layers=2, dropout=0.1, tau=0.5):
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
    model = MAGI(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_clusters=num_clusters,
        num_layers=num_layers,
        dropout=dropout,
        tau=tau,
    ).to(device)
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
    reinit_events = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        z, q = model(features, edge_index, adj_tensor)
        
        # Compute target distribution (DEC): KL(P || Q)
        with torch.no_grad():
            p = model.target_distribution(q)
        
        # Compute losses
        # KL(P||Q): encourage sharpening towards target distribution
        kl_loss = F.kl_div(p.log(), q, reduction='batchmean')
        modularity_loss = model.modularity_loss(z, adj_tensor, q)
        contrastive_loss = model.contrastive_loss(z, adj_tensor)
        center_sep = model.center_separation_loss(center_sep_sigma) if center_sep_weight > 0.0 else torch.tensor(0.0, device=z.device)
        # Cluster balance regularizer: maximize entropy of average assignment
        if balance_weight > 0.0:
            avg_q = q.mean(dim=0)
            balance = (avg_q * (avg_q + 1e-8).log()).sum()  # equals -H(avg_q)
        else:
            balance = torch.tensor(0.0, device=q.device)
        
        # Total loss - 使用可调整的权重
        total_loss = (
            kl_loss
            + modularity_weight * modularity_loss
            + contrastive_weight * contrastive_loss
            + balance_weight * balance
            + center_sep_weight * center_sep
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Empty cluster re-initialization (soft criterion) every 50 epochs
        if empty_cluster_reinit and (epoch % 50 == 0):
            with torch.no_grad():
                avg_q_epoch = q.mean(dim=0)  # [k]
                low_mass = (avg_q_epoch < empty_cluster_threshold)
                if low_mass.any():
                    # pick farthest samples from any center to reinit these centers
                    dists = torch.cdist(z, model.cluster_layer, p=2.0)  # [n,k]
                    # for each node, distance to its nearest center
                    min_to_any = dists.min(dim=1).values
                    # sort nodes by descending distance
                    order = torch.argsort(min_to_any, descending=True)
                    idx_iter = iter(order.tolist())
                    for c in torch.where(low_mass)[0].tolist():
                        # find a candidate farthest node that isn't already used
                        try:
                            node_idx = next(idx_iter)
                        except StopIteration:
                            node_idx = int(order[0].item())
                        model.cluster_layer.data[c] = z[node_idx]
                        reinit_events += 1
        
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
                  f'Balance = {balance.item():.4f}, '
                  f'CenterSep = {center_sep.item():.4f}, '
                  f'Reinit = {reinit_events}, '
                  f'Clusters = {unique_clusters}/{num_clusters}')
    
    # Get final cluster assignments
    model.eval()
    with torch.no_grad():
        z, q = model(features, edge_index, adj_tensor)
        if final_kmeans:
            km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(z.cpu().numpy())
        else:
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
