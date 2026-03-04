#-*- coding: utf-8 -*-

import numpy as np
from dataSet.data import Node, Edge


class ResourceTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.nodes = []
        self.edges = []
        self.adjacency_matrix = None
        self.distance_matrix = None
        
    def build_from_data(self):
        self._build_nodes()
        self._build_edges()
        self._build_matrices()
        return self
        
    def _build_nodes(self):
        for i in range(self.data.NodeNumber):
            resources = self.data.node_resources[i]
            node = Node(
                node_id=i,
                cpu=resources['cpu'],
                memory=resources['memory'],
                bandwidth=resources['bandwidth']
            )
            self.nodes.append(node)
            
    def _build_edges(self):
        for i in range(self.data.NodeNumber):
            for j in range(self.data.NodeNumber):
                if i != j:
                    distance = self.data.node_distances[i][j]
                    latency = distance
                    edge = Edge(src=i, dst=j, weight=distance, latency=latency)
                    self.edges.append(edge)
                    
    def _build_matrices(self):
        n = self.data.NodeNumber
        self.distance_matrix = np.array(self.data.node_distances)
        
        self.adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and self.distance_matrix[i][j] > 0:
                    self.adjacency_matrix[i][j] = 1
                    
    def get_node_feature(self, node_id):
        node = self.nodes[node_id]
        return np.array([node.cpu, node.memory, node.bandwidth])
        
    def get_all_node_features(self):
        features = []
        for node in self.nodes:
            features.append([node.cpu, node.memory, node.bandwidth])
        return np.array(features)
        
    def get_edge_weight(self, src, dst):
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                return edge.weight
        return float('inf')
        
    def get_shortest_path_distance(self, src, dst):
        if src == dst:
            return 0
        n = len(self.nodes)
        dist = self.distance_matrix[src][dst]
        if dist > 0:
            return dist
        return float('inf')
        
    def get_node_resource_capacity(self):
        capacities = []
        for node in self.nodes:
            capacities.append({
                'cpu': node.cpu,
                'memory': node.memory,
                'bandwidth': node.bandwidth
            })
        return capacities
        
    def get_node_count(self):
        return len(self.nodes)
        
    def __repr__(self):
        return f"ResourceGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
