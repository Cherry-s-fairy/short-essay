#-*- coding: utf-8 -*-

import numpy as np
from dataSet.data import Service, Edge


class TaskTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.services = []
        self.edges = []
        self.dependency_matrix = None
        self.communication_matrix = None
        self._service_resources = {}
        
    def build_from_data(self):
        self._build_services()
        self._build_dependencies()
        self._build_matrices()
        return self
        
    def _build_services(self):
        container_idx = 0
        for i in range(self.data.ServiceNumber):
            container_count = self.data.service_containernum[i]
            for j in range(container_count):
                if container_idx < len(self.data.container_state_queue) // 3:
                    cpu_demand = self.data.container_state_queue[container_idx * 3 + 1]
                    mem_demand = self.data.container_state_queue[container_idx * 3 + 2]
                    
                    service = Service(
                        service_id=i,
                        cpu_demand=cpu_demand,
                        memory_demand=mem_demand
                    )
                    self.services.append(service)
                    self._service_resources[i] = {
                        'cpu': cpu_demand,
                        'memory': mem_demand
                    }
                container_idx += 1
                
        if not self.services:
            for i in range(self.data.ServiceNumber):
                service = Service(
                    service_id=i,
                    cpu_demand=0.1,
                    memory_demand=0.1
                )
                self.services.append(service)
                self._service_resources[i] = {
                    'cpu': 0.1,
                    'memory': 0.1
                }
                
    def _build_dependencies(self):
        for i in range(self.data.ServiceNumber):
            for j in range(self.data.ServiceNumber):
                if i != j and self.data.service_weight[i][j] > 0:
                    weight = self.data.service_weight[i][j]
                    edge = Edge(src=i, dst=j, weight=weight)
                    self.edges.append(edge)
                    
                    if i < len(self.services) and j < len(self.services):
                        self.services[i].add_dependency(self.services[j], weight)
                        
    def _build_matrices(self):
        n = len(self.services)
        if n == 0:
            return
            
        self.communication_matrix = np.zeros((n, n))
        for edge in self.edges:
            if edge.src < n and edge.dst < n:
                self.communication_matrix[edge.src][edge.dst] = edge.weight
                
        self.dependency_matrix = (self.communication_matrix > 0).astype(int)
        
    def get_service_feature(self, service_id):
        if service_id < len(self.services):
            service = self.services[service_id]
            return np.array([
                service.cpu_demand,
                service.memory_demand,
                len(service.dependencies)
            ])
        return np.array([0, 0, 0])
        
    def get_all_service_features(self):
        features = []
        for service in self.services:
            features.append([
                service.cpu_demand,
                service.memory_demand,
                len(service.dependencies)
            ])
        if not features:
            features = [[0.1, 0.1, 0] for _ in range(self.data.ServiceNumber)]
        return np.array(features)
        
    def get_edge_weight(self, src, dst):
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                return edge.weight
        return 0
        
    def get_total_communication_weight(self):
        total = 0
        for edge in self.edges:
            total += edge.weight
        return total
        
    def get_service_resource_demand(self, service_id):
        return self._service_resources.get(service_id, {'cpu': 0, 'memory': 0})
        
    def get_service_count(self):
        return len(self.services)
        
    def adjust_topology(self, feedback):
        adjustment_type = feedback.get('adjustment_type', 'none')
        
        if adjustment_type == 'reduce_dependency':
            self._reduce_high_latency_dependencies(feedback.get('threshold', 0.5))
        elif adjustment_type == 'increase_capacity':
            self._increase_service_resources(feedback.get('scale_factor', 1.2))
        elif adjustment_type == 'decrease_capacity':
            self._decrease_service_resources(feedback.get('scale_factor', 0.8))
        elif adjustment_type == 'rebalance':
            self._rebalance_topology()
            
        self._build_matrices()
        
    def _reduce_high_latency_dependencies(self, threshold):
        new_edges = []
        for edge in self.edges:
            if edge.weight < threshold * self.get_total_communication_weight() / len(self.edges):
                new_edges.append(edge)
        self.edges = new_edges
        
    def _increase_service_resources(self, scale_factor):
        for service in self.services:
            service.cpu_demand *= scale_factor
            service.memory_demand *= scale_factor
            self._service_resources[service.id] = {
                'cpu': service.cpu_demand,
                'memory': service.memory_demand
            }
            
    def _decrease_service_resources(self, scale_factor):
        for service in self.services:
            service.cpu_demand = max(0.01, service.cpu_demand * scale_factor)
            service.memory_demand = max(0.01, service.memory_demand * scale_factor)
            self._service_resources[service.id] = {
                'cpu': service.cpu_demand,
                'memory': service.memory_demand
            }
            
    def _rebalance_topology():
        pass
        
    def __repr__(self):
        return f"TaskGraph(services={len(self.services)}, edges={len(self.edges)})"
