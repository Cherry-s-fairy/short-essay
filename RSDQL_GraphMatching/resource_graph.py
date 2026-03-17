#-*- coding: utf-8 -*-

import numpy as np
import math


class ResourceTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.nodes = []
        self.edges = []
        self.adjacency_matrix = None
        self.distance_matrix = None

    # 从加载的数据文件中构建资源拓扑图
    def build_from_data(self):
        self._build_nodes()
        self._build_edges()
        self._build_matrices()
        return self
        
    def _build_nodes(self):
        for node_id in sorted(self.data.uav_nodes.keys()):
            node = self.data.uav_nodes[node_id]
            out_bandwidth = sum(edge['bandwidth'] for edge in node.src_edges)
            out_latency = sum(edge['latency'] for edge in node.src_edges)
            out_loss = sum(edge['loss'] for edge in node.src_edges)
            edge_count = len(node.src_edges)
            self.nodes.append({
                'id': node.id,
                'total_cpu': node.total_cpu,
                'total_memory': node.total_memory,
                'remain_cpu': node.remain_cpu,
                'remain_memory': node.remain_memory,
                'out_edge_count': edge_count,
                'out_bandwidth': out_bandwidth,
                'out_latency': out_latency,
                'out_loss': out_loss,
                'src_edges': node.src_edges,
            })
            
    def _build_edges(self):
        edges_data = []
        for edge in self.data.uav_edges:
            weight = math.log(edge.bandwidth * edge.latency * (1 - edge.loss) + 1)
            edges_data.append({
                'src': edge.src,
                'dst': edge.dst,
                'bandwidth': edge.bandwidth,
                'latency': edge.latency,
                'loss': edge.loss if hasattr(edge, 'loss') else 0,
                'weight': weight
            })


        max_weight = max(e['weight'] for e in edges_data) if edges_data else 1
        min_weight = min(e['weight'] for e in edges_data) if edges_data else 0

        for e in edges_data:
            if max_weight > min_weight:
                weight = (e['weight'] - min_weight) / (max_weight - min_weight)
            else:
                weight = 0.5
            self.edges.append({
                'src': e['src'],
                'dst': e['dst'],
                'bandwidth': e['bandwidth'],
                'latency': e['latency'],
                'loss': e['loss'],
                'weight': weight
            })
        
        for node1_id in self.data.uav_nodes:
            for node2_id in self.data.uav_nodes:
                if node1_id != node2_id:
                    found = False
                    for edge in self.edges:
                        if edge['src'] == node1_id and edge['dst'] == node2_id:
                            found = True
                            break
                    if not found:
                        self.edges.append({
                            'src': node1_id,
                            'dst': node2_id,
                            'bandwidth': 0,
                            'latency': float('inf'),
                            'loss': 1,
                            'weight': float('0')
                        })
                        
    def _build_matrices(self):
        n = len(self.nodes)
        if n == 0:
            return
            
        self.distance_matrix = np.full((n, n), float('inf'))
        
        for edge in self.edges:
            src = edge['src'] - 1
            dst = edge['dst'] - 1
            if 0 <= src < n and 0 <= dst < n:
                self.distance_matrix[src][dst] = edge['weight']
                
        for i in range(n):
            self.distance_matrix[i][i] = 0
            
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distance_matrix[i][k] + self.distance_matrix[k][j] < self.distance_matrix[i][j]:
                        self.distance_matrix[i][j] = self.distance_matrix[i][k] + self.distance_matrix[k][j]
        
        self.adjacency_matrix = (self.distance_matrix < float('inf')).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
    def get_node_feature(self, node_id):
        if node_id < len(self.nodes):
            node = self.nodes[node_id]
            return np.array([node['total_cpu'], node['total_memory'], node['out_edge_count']])
        return np.array([0, 0, 0])
        
    def get_all_node_features(self):
        features = []
        for node in self.nodes:
            features.append([node['total_cpu'], node['total_memory'], node['out_edge_count']])
        return np.array(features)
        
    def get_edge_weight(self, src, dst):
        for edge in self.edges:
            if edge['src'] == src and edge['dst'] == dst:
                return np.array([edge['bandwidth'], edge['latency']])
        return np.array([float('inf'), float('inf')])
        
    def get_shortest_path_distance(self, src, dst):
        if src == dst:
            return 0
        n = len(self.nodes)
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return float('inf')
        dist = self.distance_matrix[src][dst]
        return dist
        
    def get_node_resource_capacity(self):
        capacities = []
        for node in self.nodes:
            capacities.append({
                'cpu': node['total_cpu'],
                'memory': node['total_memory'],
                'out_edge_count': node['out_edge_count']
            })
        return capacities
        
    def get_node_count(self):
        return len(self.nodes)
        
    def __repr__(self):
        return f"ResourceGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


def test_resource_graph():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from dataSet.data import Data
    
    print("="*60)
    print("ResourceTopologyGraph Test")
    print("="*60)
    
    data = Data('dataSet/data.xml')
    print(f"\nLoaded {len(data.uav_nodes)} UAV nodes")
    print(f"Loaded {len(data.uav_edges)} UAV edges")
    
    resource_graph = ResourceTopologyGraph(data)
    resource_graph.build_from_data()
    
    print(f"\n{resource_graph}")
    
    print("\n--- Nodes ---")
    for i, node in enumerate(resource_graph.nodes):
        print(f"  Node {i}: cpu={node['total_cpu']}, mem={node['total_memory']}, out_edges={node['out_edge_count']}, src_edges={node['src_edges']}")

    print("\n--- Edges ---")
    for edge in resource_graph.edges:
        print(f"  Edge: {edge['src']} -> {edge['dst']}, bandwidth={edge['bandwidth']}, latency={edge['latency']}, loss={edge['loss']}, weight={edge['weight']}")

    print("\n--- Distance Matrix ---")
    print(resource_graph.distance_matrix)
    
    print("\n--- Adjacency Matrix ---")
    print(resource_graph.adjacency_matrix)
    
    print("\n--- Node Features ---")
    for i in range(resource_graph.get_node_count()):
        feature = resource_graph.get_node_feature(i)
        print(f"  Node {i} feature: {feature}")
    
    print("\n--- Shortest Path Tests ---")
    test_pairs = [(0, 1), (0, 2), (1, 2), (0, 0)]
    for src, dst in test_pairs:
        dist = resource_graph.get_shortest_path_distance(src, dst)
        print(f"  Distance({src} -> {dst}): {dist}")
    
    print("\n--- Node Resource Capacity ---")
    capacities = resource_graph.get_node_resource_capacity()
    for i, cap in enumerate(capacities):
        print(f"  Node {i}: cpu={cap['cpu']}, mem={cap['memory']}, out_edges={cap['out_edge_count']}")
    
    print("\n--- Edge Weight Tests ---")
    test_edges = [(1, 2), (1, 1), (2, 1)]
    for src, dst in test_edges:
        weight = resource_graph.get_edge_weight(src, dst)
        print(f"  Weight({src} -> {dst}): {weight}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == '__main__':
    test_resource_graph()
