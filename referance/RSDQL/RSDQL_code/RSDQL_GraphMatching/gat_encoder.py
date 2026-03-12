#-*- coding: utf-8 -*-

import numpy as np
import os


class GATEncoder:
    def __init__(self, hidden_dim=32, num_layers=3, num_heads=4, use_pyg=False):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_pyg = use_pyg and self._check_pyg_available()

        # 同时固定NumPy和PyTorch种子（保证复现性）
        np.random.seed(42)
        if self.use_pyg:
            import torch
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
        
        if self.use_pyg:
            self._init_pyg_model()
        else:
            self._init_numpy_model()
    
    def _check_pyg_available(self):
        try:
            import torch
            from torch_geometric.nn import GATConv
            from torch_geometric.data import Data
            return True
        except ImportError as e:
            print(f"PyG依赖缺失: {e}")
            return False
    
    def _init_pyg_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        
        class PYGATEncoder(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=32, num_layers=3, num_heads=4):
                super().__init__()
                self.gat_layers = nn.ModuleList()
                self.dropout = 0.1
                head_dim = hidden_dim // num_heads

                # 第一层：3→8*4=32
                self.gat_layers.append(GATConv(input_dim, head_dim, heads=num_heads, dropout=self.dropout, edge_dim=1, add_self_loops=False))
                # 中间层：32→8*4=32
                for _ in range(num_layers - 2):
                    self.gat_layers.append(GATConv(hidden_dim, head_dim, heads=num_heads, dropout=self.dropout, edge_dim=1, add_self_loops=False))
                # 最后一层：32→32
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=self.dropout, edge_dim=1, add_self_loops=False))

            def forward(self, x, edge_index, edge_attr=None):
                h = x
                # 安全检查：边权重长度必须等于边索引列数
                if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
                    edge_attr = None  # 维度不匹配时禁用边权重

                for i, layer in enumerate(self.gat_layers):
                    h = layer(h, edge_index, edge_attr=edge_attr)
                    if i != len(self.gat_layers) - 1:
                        h = F.relu(h)
                        h = F.dropout(h, p=self.dropout, training=self.training)
                return h

        self.pyg_model = PYGATEncoder(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )
        self.pyg_model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pyg_model = self.pyg_model.to(self.device)
    
    def _init_numpy_model(self):
        self.numpy_weights = []
        input_dim = 3
        hidden_dim = self.hidden_dim

        for layer_idx in range(self.num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim
            # Xavier初始化
            W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))
            self.numpy_weights.append(W)
    
    def _build_edge_index(self, edges, num_nodes):
        edge_index = [[], []]
        edge_weights = []
        for edge in edges:
            src = edge.get('src', 0)
            dst = edge.get('dst', 0)
            weight = edge.get('weight', 0.0)
            if src < num_nodes and dst < num_nodes:
                edge_index[0].append(src)
                edge_index[1].append(dst)
                edge_weights.append(weight)
                edge_index[0].append(dst)
                edge_index[1].append(src)
                edge_weights.append(weight)
        for i in range(1, num_nodes + 1):
            edge_index.append((i, i))
            edge_weights.append(1.0)
        edge_index = np.array(edge_index) if edge_index[0] else np.array([[], []])
        edge_weights = np.array(edge_weights, dtype=np.float32) if edge_weights else np.array([])
        return edge_index, edge_weights
    
    def _normalize_features(self, features):
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=0.0)
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-6] = 1.0
        return (features - f_min) / f_range
    
    def _gat_layer_numpy(self, features, edge_index, edge_weights, layer_idx):
        n, d_in = features.shape
        W = self.numpy_weights[layer_idx]
        d_out = W.shape[1]

        # 线性变换
        h = features @ W

        if edge_index.shape[1] == 0:
            return np.maximum(0, h)

        # 注意力分数
        h_i = h[edge_index[0]]
        h_j = h[edge_index[1]]
        e = np.sum(h_i * h_j, axis=1)

        # 边权重加权
        if edge_weights.size > 0:
            e = e * edge_weights

        # 稳定 softmax
        e_max = np.max(e) if e.size > 0 else 0.0
        e_exp = np.exp(e - e_max)

        attention_weights = {}
        for idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
            if i not in attention_weights:
                attention_weights[i] = []
            attention_weights[i].append((j, e_exp[idx]))

        output = np.zeros((n, d_out))
        for i in range(n):
            if i in attention_weights and len(attention_weights[i]) > 0:
                neighbors, weights = zip(*attention_weights[i])
                total = sum(weights)
                if total > 1e-6:
                    weights = [w / total for w in weights]
                    for idx, neighbor in enumerate(neighbors):
                        output[i] += weights[idx] * h[neighbor]
                else:
                    output[i] = h[i]
            else:
                output[i] = h[i]

        output = np.maximum(0, output)

        return output
    
    def encode_numpy(self, nodes, edges):
        if not nodes:
            return np.zeros(self.hidden_dim)

        num_nodes = len(nodes)

        # 预分配数组，避免列表append（提升性能）
        node_features = np.zeros((num_nodes, 3), dtype=np.float32)
        for i, node in enumerate(nodes):
            node_features[i] = np.array([
                node.get('remain_cpu', 0) / 100.0,
                node.get('remain_memory', 0) / 128.0,
                node.get('out_edge_count', 0) / max(num_nodes, 1),
            ])

        node_features = self._normalize_features(node_features)
        edge_index, edge_weights = self._build_edge_index(edges, num_nodes)

        h = node_features
        for layer_idx in range(self.num_layers):
            h = self._gat_layer_numpy(h, edge_index, edge_weights, layer_idx)

        # 度加权平均（与PyG对齐）
        if edge_index.shape[1] > 0:
            node_degree = np.bincount(edge_index[0], minlength=num_nodes)
        else:
            node_degree = np.ones(num_nodes)
        node_degree = node_degree / (node_degree.sum() + 1e-6)
        graph_embedding = np.sum(h * node_degree.reshape(-1, 1), axis=0)

        return graph_embedding

    def encode_pyg(self, nodes, edges):
        import torch

        if not nodes:
            return np.zeros(self.hidden_dim)

        num_nodes = len(nodes)

        # 预分配数组，避免列表append（消除性能警告）
        node_features = np.zeros((num_nodes, 3), dtype=np.float32)
        for i, node in enumerate(nodes):
            node_features[i] = np.array([
                node.get('remain_cpu', 0.0),
                node.get('remain_memory', 0.0),
                node.get('out_edge_count', 0.0),
            ])

        # 转换为张量并归一化
        x = torch.from_numpy(node_features).float()
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_range = x_max - x_min
        x_range[x_range < 1e-6] = 1.0
        x = (x - x_min) / x_range

        # 构建边索引和边权重（保持 [2, num_edges] 格式）
        edge_index, edge_weights = self._build_edge_index(edges, num_nodes)
        edge_index = torch.tensor(edge_index, dtype=torch.long) if edge_index.size > 0 else torch.empty((2, 0), dtype=torch.long)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32) if edge_weights.size > 0 else torch.empty(0, dtype=torch.float32)

        # 设备迁移
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weights = edge_weights.to(self.device)

        # 前向传播（传入边权重）
        with torch.no_grad():
            node_embeddings = self.pyg_model(x, edge_index, edge_attr=edge_weights)

        # 无边时的度加权平均
        if edge_index.numel() > 0:
            node_degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
        else:
            node_degree = torch.ones(num_nodes, device=self.device).float()
        node_degree = node_degree / (node_degree.sum() + 1e-6)

        graph_embedding = (node_embeddings * node_degree.unsqueeze(1)).sum(dim=0)

        return graph_embedding.cpu().numpy()

    def encode(self, nodes, edges):
        if self.use_pyg:
            return self.encode_pyg(nodes, edges)
        else:
            return self.encode_numpy(nodes, edges)


def test_gat_encoder():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from dataSet.data import Data
    from resource_graph import ResourceTopologyGraph

    print("=" * 70)
    print("ResourceAwareTaskGraphGenerator Test")
    print("=" * 70)

    data = Data('dataSet/data.xml')
    resource_graph = ResourceTopologyGraph(data)
    resource_graph.build_from_data()

    print(f"\nResource Graph: {resource_graph}")
    print(f"  Nodes: {resource_graph.get_node_count()}")
    
    print("Testing NumPy GAT Encoder:")
    encoder = GATEncoder(hidden_dim=32, num_layers=3, num_heads=4, use_pyg=False)
    embedding = encoder.encode(resource_graph.nodes, resource_graph.edges)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding: {embedding[:5]}...")
    
    print("\nTesting PyG GAT Encoder:")
    try:
        encoder_pyg = GATEncoder(hidden_dim=32, num_layers=3, num_heads=4, use_pyg=True)
        embedding_pyg = encoder_pyg.encode(resource_graph.nodes, resource_graph.edges)
        print(f"  Embedding shape: {embedding_pyg.shape}")
        print(f"  Embedding: {embedding_pyg}")
    except Exception as e:
        print(f"  PyG not available: {e}")


if __name__ == '__main__':
    test_gat_encoder()
