#-*- coding: utf-8 -*-

import numpy as np


class ResourceTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.nodes = []
        self.edges = []
        self.adjacency_matrix = None
        self.distance_matrix = None
        self._node_latency = {}
        
    def build_from_data(self):
        self._build_nodes()
        self._build_edges()
        self._build_matrices()
        return self
        
    def _build_nodes(self):
        for node_id, node in self.data.uav_nodes.items():
            self.nodes[node_id] = node

    def _build_edges(self):
        for node_id, node in self.data.uav_nodes.items():
            for edge in node.src_edges:
                self.edges.append({
                    'src': edge['src'],
                    'dst': edge['dst'],
                    'bandwidth': edge['bandwidth'],
                    'latency': edge['latency'],
                    'weight': edge['latency']
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
                            'bandwidth': 100,
                            'latency': 50,
                            'weight': 50
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
                self.distance_matrix[src][dst] = edge['latency']
                
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
            return np.array([node['cpu'], node['memory'], node['bandwidth']])
        return np.array([0, 0, 0])
        
    def get_all_node_features(self):
        features = []
        for node in self.nodes:
            features.append([node['cpu'], node['memory'], node['bandwidth']])
        return np.array(features)
        
    def get_edge_weight(self, src, dst):
        for edge in self.edges:
            if edge['src'] == src and edge['dst'] == dst:
                return edge['weight']
        return float('inf')
        
    def get_shortest_path_distance(self, src, dst):
        if src == dst:
            return 0
        n = len(self.nodes)
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return float('inf')
        dist = self.distance_matrix[src][dst]
        if dist < float('inf'):
            return dist
        return float('inf')
        
    def get_node_resource_capacity(self):
        capacities = []
        for node in self.nodes:
            capacities.append({
                'cpu': node['cpu'],
                'memory': node['memory'],
                'bandwidth': node['bandwidth']
            })
        return capacities
        
    def get_node_count(self):
        return len(self.nodes)
        
    def __repr__(self):
        return f"ResourceGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
