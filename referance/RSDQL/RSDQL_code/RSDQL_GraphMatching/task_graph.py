#-*- coding: utf-8 -*-

import numpy as np
import random


class TaskTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.services = []
        self.edges = []
        self.dependency_matrix = None
        self.communication_matrix = None
        self._service_resources = {}
        self.adjustment_history = []
        
    def build_from_data(self):
        self._build_services()
        self._build_dependencies()
        self._build_matrices()
        return self
        
    def _build_services(self):
        for service_id, service in self.data.service_nodes.items():
            self.services.append({
                'id': service.id,
                'cpu_demand': service.cpu_demand,
                'memory_demand': service.memory_demand,
                'dependencies': service.dependencies.copy()
            })
            self._service_resources[service.id] = {
                'cpu': service.cpu_demand,
                'memory': service.memory_demand
            }
            
    def _build_dependencies(self):
        for service_id, service in self.data.service_nodes.items():
            for dep in service.dependencies:
                self.edges.append({
                    'src': dep['src'],
                    'dst': dep['dst'],
                    'weight': dep['bandwidth'],
                    'latency': dep['latency']
                })
                
    def _build_matrices(self):
        n = len(self.services)
        if n == 0:
            return
            
        self.communication_matrix = np.zeros((n, n))
        for edge in self.edges:
            src_idx = edge['src'] - 1
            dst_idx = edge['dst'] - 1
            if src_idx < n and dst_idx < n:
                self.communication_matrix[src_idx][dst_idx] = edge['weight']
                
        self.dependency_matrix = (self.communication_matrix > 0).astype(int)
        
    def get_service_feature(self, service_id):
        if service_id < len(self.services):
            service = self.services[service_id]
            return np.array([
                service['cpu_demand'],
                service['memory_demand'],
                len(service['dependencies'])
            ])
        return np.array([0, 0, 0])
        
    def get_all_service_features(self):
        features = []
        for service in self.services:
            features.append([
                service['cpu_demand'],
                service['memory_demand'],
                len(service['dependencies'])
            ])
        if not features:
            features = [[0.1, 0.1, 0] for _ in range(len(self.data.service_nodes))]
        return np.array(features)
        
    def get_edge_weight(self, src, dst):
        for edge in self.edges:
            if edge['src'] == src and edge['dst'] == dst:
                return edge['weight']
        return 0
        
    def get_total_communication_weight(self):
        total = 0
        for edge in self.edges:
            total += edge['weight']
        return total
        
    def get_service_resource_demand(self, service_id):
        return self._service_resources.get(service_id, {'cpu': 0, 'memory': 0})
        
    def get_service_count(self):
        return len(self.services)
        
    def get_topology_stats(self):
        return {
            'service_count': len(self.services),
            'edge_count': len(self.edges),
            'avg_degree': np.mean([len(s['dependencies']) for s in self.services]) if self.services else 0,
            'total_communication': self.get_total_communication_weight(),
            'density': len(self.edges) / max(1, len(self.services) * (len(self.services) - 1))
        }
        
    def adjust_topology(self, feedback):
        adjustment_type = feedback.get('adjustment_type', 'none')
        
        if adjustment_type == 'reduce_dependency':
            self._reduce_high_latency_dependencies(feedback.get('threshold', 0.5))
        elif adjustment_type == 'increase_capacity':
            self._increase_service_resources(feedback.get('scale_factor', 1.2))
        elif adjustment_type == 'decrease_capacity':
            self._decrease_service_resources(feedback.get('scale_factor', 0.8))
        elif adjustment_type == 'optimize_dependency':
            self._optimize_dependencies(feedback.get('threshold', 0.7))
        elif adjustment_type == 'rebalance_topology':
            self._rebalance_topology(feedback.get('target_balance', 0.5))
        elif adjustment_type == 'fine_tune':
            self._fine_tune_resources(feedback.get('scale_factor', 1.05))
        elif adjustment_type == 'merge_services':
            self._merge_dependent_services(feedback.get('merge_dependent', True))
        elif adjustment_type == 'split_services':
            self._split_high_load_services()
            
        self._build_matrices()
        
        self.adjustment_history.append({
            'type': adjustment_type,
            'feedback': feedback,
            'stats': self.get_topology_stats()
        })
        
    def _reduce_high_latency_dependencies(self, threshold):
        if not self.edges:
            return
            
        avg_weight = self.get_total_communication_weight() / len(self.edges)
        
        new_edges = []
        for edge in self.edges:
            if edge['weight'] >= avg_weight * threshold:
                new_edges.append(edge)
        self.edges = new_edges
        
    def _increase_service_resources(self, scale_factor):
        for service in self.services:
            service['cpu_demand'] *= scale_factor
            service['memory_demand'] *= scale_factor
            self._service_resources[service['id']] = {
                'cpu': service['cpu_demand'],
                'memory': service['memory_demand']
            }
            
    def _decrease_service_resources(self, scale_factor):
        for service in self.services:
            service['cpu_demand'] = max(0.01, service['cpu_demand'] * scale_factor)
            service['memory_demand'] = max(0.01, service['memory_demand'] * scale_factor)
            self._service_resources[service['id']] = {
                'cpu': service['cpu_demand'],
                'memory': service['memory_demand']
            }
            
    def _optimize_dependencies(self, threshold):
        if not self.edges:
            return
            
        weights = [e['weight'] for e in self.edges]
        if not weights:
            return
            
        low_threshold = np.percentile(weights, threshold * 100)
        
        new_edges = [e for e in self.edges if e['weight'] >= low_threshold]
        self.edges = new_edges
        
    def _rebalance_topology(self, target_balance):
        for service in self.services:
            load = service['cpu_demand'] + service['memory_demand']
            if load > 2 * target_balance:
                scale = target_balance / load
                service['cpu_demand'] *= scale
                service['memory_demand'] *= scale
                self._service_resources[service['id']] = {
                    'cpu': service['cpu_demand'],
                    'memory': service['memory_demand']
                }
            elif load < 0.5 * target_balance:
                scale = 1.2
                service['cpu_demand'] *= scale
                service['memory_demand'] *= scale
                self._service_resources[service['id']] = {
                    'cpu': service['cpu_demand'],
                    'memory': service['memory_demand']
                }
                
    def _fine_tune_resources(self, scale_factor):
        for service in self.services:
            service['cpu_demand'] *= scale_factor
            service['memory_demand'] *= scale_factor
            self._service_resources[service['id']] = {
                'cpu': service['cpu_demand'],
                'memory': service['memory_demand']
            }
            
    def _merge_dependent_services(self, merge_flag):
        pass
        
    def _split_high_load_services(self):
        pass
        
    def generate_variation(self, variation_type='random', seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        new_task_graph = TaskTopologyGraph(self.data)
        new_task_graph.services = []
        new_task_graph._service_resources = {}
        
        if variation_type == 'random':
            for i, service in enumerate(self.services):
                cpu_var = np.random.uniform(0.8, 1.2)
                mem_var = np.random.uniform(0.8, 1.2)
                new_task_graph.services.append({
                    'id': i,
                    'cpu_demand': service['cpu_demand'] * cpu_var,
                    'memory_demand': service['memory_demand'] * mem_var,
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.services[-1]['cpu_demand'],
                    'memory': new_task_graph.services[-1]['memory_demand']
                }
                
        elif variation_type == 'scale_up':
            for i, service in enumerate(self.services):
                scale = np.random.uniform(1.0, 1.5)
                new_task_graph.services.append({
                    'id': i,
                    'cpu_demand': service['cpu_demand'] * scale,
                    'memory_demand': service['memory_demand'] * scale,
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.services[-1]['cpu_demand'],
                    'memory': new_task_graph.services[-1]['memory_demand']
                }
                
        elif variation_type == 'scale_down':
            for i, service in enumerate(self.services):
                scale = np.random.uniform(0.5, 1.0)
                new_task_graph.services.append({
                    'id': i,
                    'cpu_demand': max(0.01, service['cpu_demand'] * scale),
                    'memory_demand': max(0.01, service['memory_demand'] * scale),
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.services[-1]['cpu_demand'],
                    'memory': new_task_graph.services[-1]['memory_demand']
                }
                
        elif variation_type == 'topology_change':
            new_task_graph.services = [{
                'id': i,
                'cpu_demand': s['cpu_demand'],
                'memory_demand': s['memory_demand'],
                'dependencies': s['dependencies'].copy()
            } for i, s in enumerate(self.services)]
            new_task_graph._service_resources = dict(self._service_resources)
            
            if len(self.edges) > 0:
                num_changes = max(1, len(self.edges) // 5)
                changed_indices = random.sample(range(len(self.edges)), 
                                                min(num_changes, len(self.edges)))
                new_edges = []
                for i, edge in enumerate(self.edges):
                    if i in changed_indices:
                        new_weight = edge['weight'] * np.random.uniform(0.5, 1.5)
                        edge['weight'] = max(0.1, new_weight)
                    new_edges.append(edge)
                new_task_graph.edges = new_edges
            else:
                new_task_graph.edges = []
                
        new_task_graph._build_matrices()
        return new_task_graph
        
    def clone(self):
        new_graph = TaskTopologyGraph(self.data)
        new_graph.services = [{
            'id': s['id'],
            'cpu_demand': s['cpu_demand'],
            'memory_demand': s['memory_demand'],
            'dependencies': s['dependencies'].copy()
        } for s in self.services]
        new_graph._service_resources = dict(self._service_resources)
        new_graph.edges = [{
            'src': e['src'],
            'dst': e['dst'],
            'weight': e['weight'],
            'latency': e['latency']
        } for e in self.edges]
        new_graph._build_matrices()
        return new_graph
        
    def __repr__(self):
        return f"TaskGraph(services={len(self.services)}, edges={len(self.edges)})"
