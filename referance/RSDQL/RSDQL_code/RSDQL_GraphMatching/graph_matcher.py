#-*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment


class GraphMatcher:
    def __init__(self, resource_graph, task_graph):
        self.resource_graph = resource_graph
        self.task_graph = task_graph
        self.mapping = None
        self.match_score = 0.0
        
    def match(self, method='hungarian'):
        if method == 'hungarian':
            self.mapping = self._hungarian_matching()
        elif method == 'greedy':
            self.mapping = self._greedy_matching()
        elif method == 'learned':
            self.mapping = self._learned_matching()
        else:
            self.mapping = self._heuristic_matching()
            
        self.match_score = self.calculate_match_score()
        return self.mapping
        
    def _calculate_node_similarity(self, service_id, node_id):
        service_feature = self.task_graph.get_service_feature(service_id)
        node_feature = self.resource_graph.get_node_feature(node_id)
        
        if service_id < len(self.task_graph.services):
            service = self.task_graph.services[service_id]
            cpu_demand = service['cpu_demand']
            mem_demand = service['memory_demand']
        else:
            return 0.0
            
        if node_id < len(self.resource_graph.nodes):
            node = self.resource_graph.nodes[node_id]
            cpu_capacity = node.get('total_cpu', node.get('cpu', 0))
            mem_capacity = node.get('total_memory', node.get('memory', 0))
        else:
            return 0.0
            
        cpu_match = 1.0 - abs(cpu_demand - cpu_capacity) / max(cpu_capacity, 1)
        mem_match = 1.0 - abs(mem_demand - mem_capacity) / max(mem_capacity, 1)
        
        similarity = 0.6 * max(0, cpu_match) + 0.4 * max(0, mem_match)
        return similarity
        
    def _calculate_edge_similarity(self, service_pairs, node_pairs):
        if not service_pairs or not node_pairs:
            return 0.0
            
        total_similarity = 0.0
        count = 0
        
        for s1, s2 in service_pairs:
            s1_node = node_pairs.get(s1, -1)
            s2_node = node_pairs.get(s2, -1)
            
            if s1_node >= 0 and s2_node >= 0:
                service_dist = self.task_graph.get_edge_weight(s1, s2)
                node_dist = self.resource_graph.get_shortest_path_distance(s1_node, s2_node)
                
                if node_dist > 0 and node_dist < float('inf'):
                    similarity = service_dist / (service_dist + node_dist)
                    total_similarity += similarity
                    count += 1
                    
        return total_similarity / count if count > 0 else 0.0
        
    def _build_cost_matrix(self):
        n_services = self.task_graph.get_service_count()
        n_nodes = self.resource_graph.get_node_count()
        
        if n_services == 0 or n_nodes == 0:
            return np.zeros((1, 1))
        
        cost_matrix = np.zeros((n_services, n_nodes))
        
        for i in range(n_services):
            for j in range(n_nodes):
                similarity = self._calculate_node_similarity(i, j)
                cost_matrix[i][j] = 1.0 - similarity
                
        return cost_matrix
        
    def _hungarian_matching(self):
        cost_matrix = self._build_cost_matrix()
        
        n_services = cost_matrix.shape[0]
        n_nodes = cost_matrix.shape[1]
        
        if n_services > n_nodes:
            larger = n_services
            cost_matrix = np.pad(cost_matrix, ((0, larger - n_services), (0, larger - n_nodes)), mode='constant', constant_values=1000)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            if i < self.task_graph.get_service_count() and j < self.resource_graph.get_node_count():
                mapping[i] = j
            
        return mapping
        
    def _greedy_matching(self):
        n_services = self.task_graph.get_service_count()
        n_nodes = self.resource_graph.get_node_count()
        
        if n_services > n_nodes:
            return None
            
        similarity_matrix = 1.0 - self._build_cost_matrix()
        
        mapping = {}
        used_nodes = set()
        
        for _ in range(n_services):
            max_sim = -1
            best_service = -1
            best_node = -1
            
            for i in range(n_services):
                if i in mapping:
                    continue
                for j in range(n_nodes):
                    if j in used_nodes:
                        continue
                    if similarity_matrix[i][j] > max_sim:
                        max_sim = similarity_matrix[i][j]
                        best_service = i
                        best_node = j
                        
            if best_service >= 0 and best_node >= 0:
                mapping[best_service] = best_node
                used_nodes.add(best_node)
                
        return mapping
        
    def _learned_matching(self):
        return self._hungarian_matching()
        
    def _heuristic_matching(self):
        n_services = self.task_graph.get_service_count()
        n_nodes = self.resource_graph.get_node_count()
        
        if n_services > n_nodes:
            return None
            
        mapping = {}
        available_nodes = list(range(n_nodes))
        
        for service_id in range(n_services):
            best_node = -1
            best_score = -1
            
            for node_id in available_nodes:
                score = self._calculate_node_similarity(service_id, node_id)
                if score > best_score:
                    best_score = score
                    best_node = node_id
                    
            if best_node >= 0:
                mapping[service_id] = best_node
                available_nodes.remove(best_node)
                
        return mapping
        
    def calculate_match_score(self):
        if not self.mapping:
            return 0.0
            
        node_score = 0.0
        for service_id, node_id in self.mapping.items():
            node_score += self._calculate_node_similarity(service_id, node_id)
        node_score /= len(self.mapping)
        
        service_pairs = []
        service_ids = list(self.mapping.keys())
        for i in range(len(service_ids)):
            for j in range(i + 1, len(service_ids)):
                service_pairs.append((service_ids[i], service_ids[j]))
        
        edge_score = self._calculate_edge_similarity(
            service_pairs,
            self.mapping
        )
        
        self.match_score = 0.7 * node_score + 0.3 * edge_score
        return self.match_score
        
    def validate_mapping(self):
        if not self.mapping:
            return False, "No mapping available"
            
        n_services = self.task_graph.get_service_count()
        n_nodes = self.resource_graph.get_node_count()
        
        if len(self.mapping) != n_services:
            return False, f"Mapping incomplete: {len(self.mapping)}/{n_services}"
            
        if len(set(self.mapping.values())) != len(self.mapping):
            return False, "Duplicate node assignment"
            
        for service_id, node_id in self.mapping.items():
            service_demand = self.task_graph.get_service_resource_demand(service_id)
            if node_id < len(self.resource_graph.nodes):
                node_capacity = self.resource_graph.nodes[node_id]
                
                if service_demand['cpu'] > node_capacity.get('total_cpu', node_capacity.get('cpu', 0)):
                    return False, f"Service {service_id} CPU demand exceeds Node {node_id} capacity"
                if service_demand['memory'] > node_capacity.get('total_memory', node_capacity.get('memory', 0)):
                    return False, f"Service {service_id} Memory demand exceeds Node {node_id} capacity"
                
        return True, "Valid mapping"
        
    def get_mapping(self):
        return self.mapping
        
    def get_deployment_plan(self):
        if not self.mapping:
            return []
            
        deployment_plan = []
        
        for service_id, node_id in self.mapping.items():
            if service_id < len(self.task_graph.services):
                service = self.task_graph.services[service_id]
                deployment_plan.append({
                    'service_id': service_id,
                    'node_id': node_id,
                    'cpu_demand': service['cpu_demand'],
                    'memory_demand': service['memory_demand']
                })
                
        return deployment_plan
