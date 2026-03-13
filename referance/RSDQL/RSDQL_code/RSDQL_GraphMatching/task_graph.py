#-*- coding: utf-8 -*-
import sys
import os
import numpy as np
from numpy.linalg import norm
import random
import torch
import torch.nn as nn

from resources_gat_encoder import ResourcesGATEncoder
from tasks_gat_encoder import TasksGATEncoder
from dataSet.data import Data
from resource_graph import ResourceTopologyGraph


class EdgeFeasibilityNet(nn.Module):
    def __init__(self, resource_dim=32, edge_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(resource_dim + edge_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class ResourceAwareTaskGraphGenerator:
    def __init__(self, resource_graph, data=None, use_pyg=False):
        self.resource_graph = resource_graph
        self.data = data
        self.Z_R = None
        self.Z_T = None
        self.iteration = 0
        self.max_iterations = 10
        self.convergence_threshold = 0.05
        self.history = []

        self.gnn_hidden_dim = 32
        self.gnn_num_layers = 3
        self.num_heads = 4
        
        self.resources_gat_encoder = ResourcesGATEncoder(
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
            num_heads=self.num_heads,
            use_pyg=use_pyg
        )

        self.tasks_gat_encoder = TasksGATEncoder(
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
            num_heads=self.num_heads,
            use_pyg=use_pyg
        )

        self.edge_model = EdgeFeasibilityNet()

    # gat对资源图编码
    def encode_resource_graph(self):
        nodes = self.resource_graph.nodes
        edges = self.resource_graph.edges if hasattr(self.resource_graph, 'edges') else []
        
        if not nodes:
            self.Z_R = np.zeros(self.gnn_hidden_dim)
            return self.Z_R
        
        self.Z_R = self.resources_gat_encoder.encode(nodes, edges)
        
        return self.Z_R

    # 为每个任务（service）生成特征向量，同时评估任务与资源节点匹配的可行性
    def generate_task_features(self, task):
        if self.Z_R is None:
            self.encode_resource_graph()
            
        cpu_demand = task.get('cpu_demand', 0)
        memory_demand = task.get('memory_demand', 0)
        
        resource_match = 0
        feasible_neighbors = []
        
        if len(self.resource_graph.nodes) > 0:
            for i, node in enumerate(self.resource_graph.nodes):
                cpu_capacity = node.get('remain_cpu', 0)
                mem_capacity = node.get('remain_memory', 0)
                
                if cpu_demand <= cpu_capacity and memory_demand <= mem_capacity:
                    feasible_neighbors.append(i)
                    
                    cpu_ratio = min(cpu_demand / max(cpu_capacity, 1), 1.0)
                    mem_ratio = min(memory_demand / max(mem_capacity, 1), 1.0)
                    match = (cpu_ratio + mem_ratio) / 2
                    resource_match = max(resource_match, match)
        
        task_embedding = np.array([
            cpu_demand,
            memory_demand,
            len(feasible_neighbors),
            resource_match,
            self.Z_R[0] if len(self.Z_R) > 0 else 0,
            self.Z_R[1] if len(self.Z_R) > 1 else 0
        ])

        # 将资源匹配特征挂载到任务对象，供GAT编码使用
        task['feasible_nodes'] = len(feasible_neighbors)
        task['resource_match'] = resource_match
        
        return task_embedding, feasible_neighbors
        
    def filter_dependency_edges(self, edges, resource_aware=True):
        if not resource_aware:
            return edges
        if self.Z_R is None:
            self.encode_resource_graph()

        filtered_edges = []
        Z_R = torch.tensor(self.Z_R, dtype=torch.float32)
        print("=== filter_dependency_edges ===")
        print(f"uavs count = {self.resource_graph.get_node_count()}")
        for edge in edges:
            src = edge.get('src', 1) - 1
            dst = edge.get('dst', 1) - 1
            bandwidth = max(edge.get('bandwidth', 0), 1e-6)
            latency = edge.get('latency', float('inf'))
            loss = edge.get('loss', 1.0)
            data = edge.get('data', 0)
            edge_feature = torch.tensor(
                [bandwidth, latency, loss, data],
                dtype=torch.float32
            )
            x = torch.cat([edge_feature, Z_R])
            feasibility = self.edge_model(x).item()
            if feasibility > 0.5:
                edge['comm_feasibility'] = feasibility
                filtered_edges.append(edge)
            print(f"边 {src}→{dst}：bandwidth={bandwidth}, latency={latency}, loss={loss}, data={data}, weight={edge.get('weight')}, comm_feasibility={feasibility}")

        return filtered_edges
        
    def encode_task_graph(self, task_graph):
        tasks = task_graph.tasks
        if not tasks:
            self.Z_T = np.zeros(self.gnn_hidden_dim)  # 改为GAT隐藏维度
            return self.Z_T

        print("=== 任务节点原始特征 ===")
        for task in tasks:
            self.generate_task_features(task)
            print(f"任务：{task}")


        edges = task_graph.edges if hasattr(task_graph, 'edges') else []
        filtered_edges = self.filter_dependency_edges(edges, resource_aware=True)
        print("\n=== 过滤后的边特征 ===")
        for edge in filtered_edges:
            print(f"边 {edge.get('src')}→{edge.get('dst')}：comm_feasibility={edge.get('comm_feasibility')}, weight={edge.get('weight')}")

        self.Z_T = self.tasks_gat_encoder.encode(tasks, filtered_edges)
        return self.Z_T
        
    def decode_combined_embedding(self):
        if self.Z_R is None or self.Z_T is None:
            return {}
    
        # 分别从资源和任务嵌入中提取指标
        resource_adaptation_score = float(np.mean(self.Z_R)) if len(self.Z_R) > 0 else 0
        task_complexity = float(np.mean(self.Z_T)) if len(self.Z_T) > 0 else 0

        # 计算嵌入的方差（表示多样性/复杂度）
        resource_variance = float(np.var(self.Z_R)) if len(self.Z_R) > 0 else 0
        task_variance = float(np.var(self.Z_T)) if len(self.Z_T) > 0 else 0

        # 任务-资源嵌入相似度（图匹配关键分数）
        cos_sim = 0.0
        if norm(self.Z_R) > 0 and norm(self.Z_T) > 0:
            cos_sim = float(np.dot(self.Z_R, self.Z_T) / (norm(self.Z_R) * norm(self.Z_T)))
        # 计算两个嵌入之间的相似度（表示匹配度）
        combined = np.concatenate([self.Z_T, self.Z_R])
        combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-6)

        decoded = {
            'resource_adaptation_score': resource_adaptation_score,      # 基于Z_R
            'task_complexity': task_complexity,                           # 基于Z_T
            'resource_variance': resource_variance,                       # 资源多样性
            'task_variance': task_variance,                               # 任务多样性
            'resource_task_similarity': round(cos_sim, 4),
            'embedding_dimension': len(combined),
            'embedding': combined.tolist()
        }

        return decoded
        
    def evaluate_deployment(self, task_graph):
        if not self.resource_graph or not task_graph.tasks:
            return {
                'R_success': 0.0,
                'T_latency': 0.0,
                'F_resched': 0,
                'match_score': 0.0,
                'cpu_util': 0.0,
                'mem_util': 0.0
            }

        # ===================== 1. 计算资源利用率 =====================
        total_cpu_demand = sum(t.get('cpu_demand', 0) for t in task_graph.tasks)
        total_mem_demand = sum(t.get('memory_demand', 0) for t in task_graph.tasks)
        
        total_cpu_capacity = sum(n.get('remain_cpu', 0) for n in self.resource_graph.nodes)
        total_mem_capacity = sum(n.get('remain_memory', 0) for n in self.resource_graph.nodes)

        cpu_util = min(1.0, total_cpu_demand / max(total_cpu_capacity, 1))
        mem_util = min(1.0, total_mem_demand / max(total_mem_capacity, 1))

        cpu_satisfy = 1.0 if total_cpu_demand <= total_cpu_capacity else max(0.0, total_cpu_capacity / total_cpu_demand)
        mem_satisfy = 1.0 if total_mem_demand <= total_mem_capacity else max(0.0, total_mem_capacity / total_mem_demand)
        R_success = (cpu_satisfy + mem_satisfy) / 2  # 均衡CPU/内存满足度

        # ===================== 2. 计算通信质量（用comm_feasibility） =====================
        comm_scores = []
        if hasattr(task_graph, 'edges') and task_graph.edges:
            for edge in task_graph.edges:
                comm = edge.get('comm_feasibility', 0.05)
                comm_scores.append(comm)

        # 通信质量得分（越高越好）
        comm_score = np.mean(comm_scores) if comm_scores else 0.5

        # ===================== 3. 重调度次数 =====================
        if R_success < 0.6:
            F_resched = min(5, int((0.6 - R_success) * 10))  # 限制最大重调度次数
        else:
            F_resched = 0

        # ===================== 4. 综合匹配得分（可配置权重） =====================
        w_resource = 0.4  # 资源权重
        w_comm = 0.4  # 通信权重
        w_util = 0.2  # 利用率权重
        match_score = (
                R_success * w_resource +
                comm_score * w_comm +
                (1 - (cpu_util + mem_util) / 2) * w_util
        )
        match_score = round(min(1.0, max(0.0, match_score)), 4)
        
        return {
            'R_success': round(R_success, 4),
            'T_latency': round(comm_score, 4),
            'F_resched': F_resched,
            'match_score': match_score,
            'cpu_utilization': round(cpu_util, 4),
            'memory_utilization': round(mem_util, 4)
        }
        
    def generate_resource_aware_task_graph(self, task_graph):
        self.encode_resource_graph()
        
        self.Z_T = None
        self.encode_task_graph(task_graph)
        
        decoded = self.decode_combined_embedding()
        
        metrics = self.evaluate_deployment(task_graph)
        
        self.history.append({
            'iteration': self.iteration,
            'metrics': metrics,
            'embedding': decoded
        })
        
        return task_graph, metrics
        
    def optimize_task_graph(self, task_graph):
        self.iteration = 0
        self.history = []
        
        original_graph = task_graph.clone()
        
        total_cpu_capacity = sum(n.get('total_cpu', 0) for n in self.resource_graph.nodes)
        total_mem_capacity = sum(n.get('total_memory', 0) for n in self.resource_graph.nodes)
        
        current_graph = original_graph.clone()
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            current_graph, metrics = self.evaluate_with_adjustment(
                current_graph.clone(), total_cpu_capacity, total_mem_capacity
            )
            
            if len(self.history) >= 2:
                prev_metrics = self.history[-2]['metrics']
                curr_metrics = self.history[-1]['metrics']
                
                success_diff = abs(curr_metrics['R_success'] - prev_metrics['R_success'])
                latency_diff = abs(curr_metrics['T_latency'] - prev_metrics['T_latency'])
                
                if success_diff < self.convergence_threshold and latency_diff < 1.0:
                    break                    
                    
            if metrics['R_success'] >= 0.85 and metrics['T_latency'] < 80:
                break                
                
        return current_graph, self.history
        
    def evaluate_with_adjustment(self, task_graph, total_cpu_capacity, total_mem_capacity):
        self.encode_resource_graph()
        self.encode_task_graph(task_graph)
        decoded = self.decode_combined_embedding()
        metrics = self.evaluate_deployment(task_graph)
        
        adjusted = False
        
        if metrics['cpu_utilization'] > 1.0 or metrics['memory_utilization'] > 1.0:
            scale_factor = min(0.7, min(total_cpu_capacity, total_mem_capacity) / max(
                sum(s.get('cpu_demand', 0) for s in task_graph.tasks),
                sum(s.get('memory_demand', 0) for s in task_graph.tasks)
            ) * 0.9)
            
            for service in task_graph.tasks:
                service['cpu_demand'] = max(1.0, service['cpu_demand'] * scale_factor)
                service['memory_demand'] = max(64.0, service['memory_demand'] * scale_factor)
                
            adjusted = True
            
        elif metrics['T_latency'] > 50 and task_graph.edges:
            feedback = {
                'adjustment_type': 'split_high_latency',
                'latency_threshold': 50
            }
            task_graph.adjust_topology(feedback)
            adjusted = True
            
        if adjusted:
            metrics = self.evaluate_deployment(task_graph)
            
        self.history.append({
            'iteration': self.iteration,
            'metrics': metrics,
            'embedding': decoded
        })
            
        return task_graph, metrics


class TaskTopologyGraph:
    def __init__(self, data):
        self.data = data
        self.tasks = []
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
        for task_id, task in self.data.task_nodes.items():
            self.tasks.append({
                'id': task.id,
                'cpu_demand': task.cpu_demand,
                'memory_demand': task.memory_demand,
                'priority': task.priority,
                'dependencies': task.dependencies.copy()
            }) 
            self._service_resources[task.id] = {
                'cpu': task.cpu_demand,
                'memory': task.memory_demand
            }
            
    def _build_dependencies(self):
        for task_id, task in self.data.task_nodes.items():
            for dep in task.dependencies:
                basic = dep['data'] * 8 / dep['bandwidth'] + dep['latency']
                weight = basic * (1 + dep['loss'] + 5)
                self.edges.append({
                    'src': dep['src'],
                    'dst': dep['dst'],
                    'bandwidth': dep['bandwidth'],
                    'latency': dep['latency'],
                    'loss': dep['loss'],
                    'data': dep['data'],
                    'weight': weight
                })
                
    def _build_matrices(self):
        n = len(self.tasks)
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
        if service_id < len(self.tasks):
            service = self.tasks[service_id]
            return np.array([
                service['cpu_demand'],
                service['memory_demand'],
                len(service['dependencies'])
            ])
        return np.array([0, 0, 0])
        
    def get_all_service_features(self):
        features = []
        for service in self.tasks:
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
        return len(self.tasks)
        
    def get_topology_stats(self):
        return {
            'service_count': len(self.tasks),
            'edge_count': len(self.edges),
            'avg_degree': np.mean([len(s['dependencies']) for s in self.tasks]) if self.tasks else 0,
            'total_communication': self.get_total_communication_weight(),
            'density': len(self.edges) / max(1, len(self.tasks) * (len(self.tasks) - 1))
        }


    def adjust_topology(self, feedback):
        adjustment_type = feedback.get('adjustment_type', 'none')
        
        if adjustment_type == 'split_high_latency':
            self._split_critical_tasks(
                feedback.get('latency_threshold', 100),
                feedback.get('success_threshold', 0.8)
            )
        elif adjustment_type == 'merge_communication':
            self._merge_neighboring_tasks(
                feedback.get('comm_cost_threshold', 50)
            )
        elif adjustment_type == 'adjust_priority':
            self._adjust_task_priority(
                feedback.get('lambda1', 0.5),
                feedback.get('lambda2', 0.5)
            )
        elif adjustment_type == 'modify_dependency':
            self._modify_dependency_edges(
                feedback.get('threshold', 0.5)
            )
        elif adjustment_type == 'reduce_dependency':
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
        for service in self.tasks:
            service['cpu_demand'] *= scale_factor
            service['memory_demand'] *= scale_factor
            self._service_resources[service['id']] = {
                'cpu': service['cpu_demand'],
                'memory': service['memory_demand']
            }
            
    def _decrease_service_resources(self, scale_factor):
        for service in self.tasks:
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
        for service in self.tasks:
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
        for service in self.tasks:
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
        
    def _split_critical_tasks(self, latency_threshold, success_threshold):
        if not self.tasks:
            return
            
        critical_tasks = []
        for service in self.tasks:
            latency = 0
            for edge in self.edges:
                if edge['src'] == service['id'] or edge['dst'] == service['id']:
                    latency += edge.get('latency', 0)
            
            if latency > latency_threshold:
                critical_tasks.append({
                    'id': service['id'],
                    'latency': latency,
                    'cpu': service['cpu_demand'],
                    'memory': service['memory_demand']
                })
        
        for task in critical_tasks:
            original_id = task['id']
            original_cpu = task['cpu']
            original_memory = task['memory']
            
            sub_task_1 = {
                'id': original_id,
                'cpu_demand': original_cpu / 2,
                'memory_demand': original_memory / 2,
                'dependencies': []
            }
            
            sub_task_2 = {
                'id': len(self.tasks),
                'cpu_demand': original_cpu / 2,
                'memory_demand': original_memory / 2,
                'dependencies': [{'src': original_id, 'dst': len(self.tasks), 'bandwidth': 10, 'latency': 5}]
            }
            
            for service in self.tasks:
                if service['id'] == original_id:
                    for dep in service['dependencies']:
                        if dep['src'] == original_id:
                            sub_task_2['dependencies'].append(dep)
            
            self.tasks = [s for s in self.tasks if s['id'] != original_id]
            self.tasks.append(sub_task_1)
            self.tasks.append(sub_task_2)
            
            self._service_resources[sub_task_1['id']] = {
                'cpu': sub_task_1['cpu_demand'],
                'memory': sub_task_1['memory_demand']
            }
            self._service_resources[sub_task_2['id']] = {
                'cpu': sub_task_2['cpu_demand'],
                'memory': sub_task_2['memory_demand']
            }
            
            new_edges = []
            for edge in self.edges:
                if edge['src'] != original_id and edge['dst'] != original_id:
                    new_edges.append(edge)
            self.edges = new_edges
            
            self.edges.append({
                'src': sub_task_1['id'],
                'dst': sub_task_2['id'],
                'bandwidth': 10,
                'latency': 5
            })
            
    def _merge_neighboring_tasks(self, comm_cost_threshold):
        if len(self.tasks) < 2:
            return
            
        for i in range(len(self.tasks)):
            for j in range(i + 1, len(self.tasks)):
                service_i = self.tasks[i]
                service_j = self.tasks[j]
                
                comm_cost = 0
                for edge in self.edges:
                    if (edge['src'] == service_i['id'] and edge['dst'] == service_j['id']) or \
                       (edge['src'] == service_j['id'] and edge['dst'] == service_i['id']):
                        comm_cost = edge.get('bandwidth', 0) * edge.get('latency', 0)
                        break
                
                if comm_cost > comm_cost_threshold:
                    merged_service = {
                        'id': service_i['id'],
                        'cpu_demand': service_i['cpu_demand'] + service_j['cpu_demand'],
                        'memory_demand': service_i['memory_demand'] + service_j['memory_demand'],
                        'dependencies': []
                    }
                    
                    for dep in service_i['dependencies']:
                        if dep['dst'] != service_j['id']:
                            merged_service['dependencies'].append(dep)
                    for dep in service_j['dependencies']:
                        if dep['dst'] != service_i['id'] and dep['src'] != service_i['id']:
                            merged_service['dependencies'].append({
                                'src': service_i['id'],
                                'dst': dep['dst'],
                                'bandwidth': dep.get('bandwidth', 0),
                                'latency': dep.get('latency', 0)
                            })
                    
                    self.tasks = [s for s in self.tasks if s['id'] not in [service_i['id'], service_j['id']]]
                    self.tasks.append(merged_service)
                    
                    self._service_resources[merged_service['id']] = {
                        'cpu': merged_service['cpu_demand'],
                        'memory': merged_service['memory_demand']
                    }
                    
                    new_edges = []
                    for edge in self.edges:
                        if edge['src'] not in [service_i['id'], service_j['id']] and \
                           edge['dst'] not in [service_i['id'], service_j['id']]:
                            new_edges.append(edge)
                    self.edges = new_edges
                    
                    break
                    
    def _adjust_task_priority(self, lambda1, lambda2):
        for service in self.tasks:
            success_rate = service.get('success_rate', 0.9)
            latency = 0
            for edge in self.edges:
                if edge['src'] == service['id']:
                    latency += edge.get('latency', 0)
            
            priority = lambda1 / (latency + 1) + lambda2 * success_rate
            service['priority'] = priority
            
    def _modify_dependency_edges(self, threshold):
        if not self.edges:
            return
            
        weights = []
        for edge in self.edges:
            weight = edge.get('bandwidth', 0) * edge.get('latency', 0)
            weights.append(weight)
            
        if not weights:
            return
            
        threshold_value = np.percentile(weights, threshold * 100)
        
        new_edges = []
        for edge in self.edges:
            weight = edge.get('bandwidth', 0) * edge.get('latency', 0)
            if weight >= threshold_value:
                new_edges.append(edge)
                
        self.edges = new_edges
        
    def generate_variation(self, variation_type='random', seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        new_task_graph = TaskTopologyGraph(self.data)
        new_task_graph.tasks = []
        new_task_graph._service_resources = {}
        
        if variation_type == 'random':
            for i, service in enumerate(self.tasks):
                cpu_var = np.random.uniform(0.8, 1.2)
                mem_var = np.random.uniform(0.8, 1.2)
                new_task_graph.tasks.append({
                    'id': i,
                    'cpu_demand': service['cpu_demand'] * cpu_var,
                    'memory_demand': service['memory_demand'] * mem_var,
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.tasks[-1]['cpu_demand'],
                    'memory': new_task_graph.tasks[-1]['memory_demand']
                }
                
        elif variation_type == 'scale_up':
            for i, service in enumerate(self.tasks):
                scale = np.random.uniform(1.0, 1.5)
                new_task_graph.tasks.append({
                    'id': i,
                    'cpu_demand': service['cpu_demand'] * scale,
                    'memory_demand': service['memory_demand'] * scale,
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.tasks[-1]['cpu_demand'],
                    'memory': new_task_graph.tasks[-1]['memory_demand']
                }
                
        elif variation_type == 'scale_down':
            for i, service in enumerate(self.tasks):
                scale = np.random.uniform(0.5, 1.0)
                new_task_graph.tasks.append({
                    'id': i,
                    'cpu_demand': max(0.01, service['cpu_demand'] * scale),
                    'memory_demand': max(0.01, service['memory_demand'] * scale),
                    'dependencies': service['dependencies'].copy()
                })
                new_task_graph._service_resources[i] = {
                    'cpu': new_task_graph.tasks[-1]['cpu_demand'],
                    'memory': new_task_graph.tasks[-1]['memory_demand']
                }
                
        elif variation_type == 'topology_change':
            new_task_graph.tasks = [{
                'id': i,
                'cpu_demand': s['cpu_demand'],
                'memory_demand': s['memory_demand'],
                'dependencies': s['dependencies'].copy()
            } for i, s in enumerate(self.tasks)]
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
        new_graph.tasks = [{
            'id': s['id'],
            'cpu_demand': s['cpu_demand'],
            'memory_demand': s['memory_demand'],
            'dependencies': s['dependencies'].copy()
        } for s in self.tasks]
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
        return f"TaskGraph(services={len(self.tasks)}, edges={len(self.edges)})"

def load_data():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("Loading data")
    print("=" * 70)

    data = Data('dataSet/data.xml')

    resource_graph = ResourceTopologyGraph(data)
    resource_graph.build_from_data()
    print(f"\nResource Graph: {resource_graph}")
    print(f"  Nodes: {resource_graph.get_node_count()}")

    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    print(f"Task Graph: {task_graph}")
    print(f"  Nodes: {task_graph.get_service_count()}")

    return data, resource_graph, task_graph

def test_resource_aware_generator():
    data, resource_graph, task_graph = load_data()
    
    generator = ResourceAwareTaskGraphGenerator(resource_graph, data)
    
    print("\n--- Test 1: Encode Resource Graph ---")
    Z_R = generator.encode_resource_graph()
    print(f"  Z_R (resource embedding): {Z_R}")
    print(f"  Shape: {Z_R.shape}")
    
    print("\n--- Test 2: Generate Task Features ---")
    for i, service in enumerate(task_graph.tasks[:3]):
        task_embedding, feasible_neighbors = generator.generate_task_features(service)
        print(f"  Task {i}:")
        print(f"    Embedding: {task_embedding}")
        print(f"    Feasible UAV neighbors: {feasible_neighbors}")
    
    print("\n--- Test 3: Filter Dependency Edges ---")
    print(f"  Original edges: {len(task_graph.edges)}")
    filtered_edges = generator.filter_dependency_edges(task_graph.edges.copy())
    print(f"  Filtered edges: {len(filtered_edges)}")
    for edge in filtered_edges:
        comm_feas = edge.get('comm_feasibility', 'N/A')
        print(f"    {edge['src']} -> {edge['dst']}: comm_feasibility = {comm_feas}")
    
    print("\n--- Test 4: Encode Task Graph ---")
    Z_T = generator.encode_task_graph(task_graph)
    print(f"  Z_T (task embedding): {Z_T}")
    print(f"  Shape: {Z_T.shape}")
    
    print("\n--- Test 5: Decode Combined Embedding ---")
    decoded = generator.decode_combined_embedding()
    print(f"  Resource adaptation score: {decoded.get('resource_adaptation_score', 'N/A'):.4f}")
    print(f"  Task complexity: {decoded.get('task_complexity', 'N/A'):.4f}")
    print(f"  Combined embedding length: {len(decoded.get('embedding', []))}")
    
    print("\n--- Test 6: Evaluate Deployment ---")
    metrics = generator.evaluate_deployment(task_graph)
    print(f"  R_success (success rate): {metrics['R_success']:.4f}")
    print(f"  T_latency (avg latency): {metrics['T_latency']:.2f} ms")
    print(f"  F_resched (reschedule count): {metrics['F_resched']}")
    print(f"  Match score: {metrics['match_score']:.4f}")
    print(f"  CPU utilization: {metrics['cpu_utilization']:.4f}")
    print(f"  Memory utilization: {metrics['memory_utilization']:.4f}")
    
    print("\n--- Test 7: Generate Resource-Aware Task Graph ---")
    generator.history = []
    adapted_graph, gen_metrics = generator.generate_resource_aware_task_graph(task_graph)
    print(f"  Adapted graph: {adapted_graph}")
    print(f"  Metrics: R_success={gen_metrics['R_success']:.4f}, T_latency={gen_metrics['T_latency']:.2f}")
    
    print("\n--- Test 8: Optimize Task Graph (Iterative) ---")
    generator.history = []
    generator.iteration = 0
    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    
    optimized_graph, history = generator.optimize_task_graph(task_graph)
    print(f"  Original: {task_graph}")
    print(f"  Optimized: {optimized_graph}")
    print(f"  Iterations: {len(history)}")
    
    print("\n  --- Optimization History ---")
    for h in history:
        m = h['metrics']
        print(f"    Iter {h['iteration']}: R_success={m['R_success']:.4f}, "
              f"T_latency={m['T_latency']:.2f}, "
              f"F_resched={m['F_resched']}, "
              f"match={m['match_score']:.4f}")
    
    print("\n--- Test 9: Compare Original vs Optimized ---")
    original_metrics = generator.evaluate_deployment(task_graph)
    optimized_metrics = generator.evaluate_deployment(optimized_graph)
    
    print(f"  Original Task Graph:")
    print(f"    Services: {len(task_graph.tasks)}, Edges: {len(task_graph.edges)}")
    print(f"    R_success: {original_metrics['R_success']:.4f}")
    print(f"    T_latency: {original_metrics['T_latency']:.2f} ms")
    print(f"    Match score: {original_metrics['match_score']:.4f}")
    
    print(f"\n  Optimized Task Graph:")
    print(f"    Services: {len(optimized_graph.services)}, Edges: {len(optimized_graph.edges)}")
    print(f"    R_success: {optimized_metrics['R_success']:.4f}")
    print(f"    T_latency: {optimized_metrics['T_latency']:.2f} ms")
    print(f"    Match score: {optimized_metrics['match_score']:.4f}")
    
    improvement = optimized_metrics['match_score'] - original_metrics['match_score']
    print(f"\n  Improvement: {improvement:+.4f} ({improvement/max(original_metrics['match_score'],0.001)*100:+.2f}%)")
    
    print("\n" + "="*70)
    print("ResourceAwareTaskGraphGenerator Test Complete!")
    print("="*70)


def test_task_graph():
    data, _, task_graph = load_data()
    print("\n--- Services ---")
    for i, service in enumerate(task_graph.tasks):
        print(f"  Task {i}: cpu={service['cpu_demand']}, mem={service['memory_demand']}, deps={len(service['dependencies'])}")
    
    print("\n--- Dependencies ---")
    for edge in task_graph.edges:
        print(f"  {edge['src']} -> {edge['dst']}: bw={edge.get('bandwidth', 0)}, lat={edge.get('latency', 0)}")
    
    print("\n--- Communication Matrix ---")
    print(task_graph.communication_matrix)
    
    print("\n--- Task Features ---")
    for i in range(task_graph.get_service_count()):
        feature = task_graph.get_service_feature(i)
        print(f"  Task {i}: {feature}")
    
    print("\n--- Topology Stats ---")
    stats = task_graph.get_topology_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Testing Adjustment Methods")
    print("="*60)
    
    print("\n--- Test 1: Split Critical Tasks ---")
    feedback1 = {
        'adjustment_type': 'split_high_latency',
        'latency_threshold': 50,
        'success_threshold': 0.8
    }
    task_graph.adjust_topology(feedback1)
    print(f"After split: {task_graph}")
    print(f"  Services: {len(task_graph.tasks)}")
    print(f"  Edges: {len(task_graph.edges)}")
    
    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    
    print("\n--- Test 2: Merge Neighboring Tasks ---")
    feedback2 = {
        'adjustment_type': 'merge_communication',
        'comm_cost_threshold': 50
    }
    task_graph.adjust_topology(feedback2)
    print(f"After merge: {task_graph}")
    print(f"  Services: {len(task_graph.tasks)}")
    print(f"  Edges: {len(task_graph.edges)}")
    
    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    
    print("\n--- Test 3: Adjust Priority ---")
    feedback3 = {
        'adjustment_type': 'adjust_priority',
        'lambda1': 0.5,
        'lambda2': 0.5
    }
    task_graph.adjust_topology(feedback3)
    print(f"After priority adjust: {task_graph}")
    for i, service in enumerate(task_graph.tasks):
        priority = service.get('priority', 'N/A')
        print(f"  Task {i} priority: {priority}")
    
    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    
    print("\n--- Test 4: Modify Dependency Edges ---")
    feedback4 = {
        'adjustment_type': 'modify_dependency',
        'threshold': 0.5
    }
    task_graph.adjust_topology(feedback4)
    print(f"After modify dependency: {task_graph}")
    print(f"  Edges: {len(task_graph.edges)}")
    for edge in task_graph.edges:
        print(f"    {edge['src']} -> {edge['dst']}")
    
    task_graph = TaskTopologyGraph(data)
    task_graph.build_from_data()
    
    print("\n--- Test 5: Generate Variation (random) ---")
    variation = task_graph.generate_variation('random', seed=42)
    print(f"Original: {task_graph}")
    print(f"Variation: {variation}")
    for i, service in enumerate(variation.tasks):
        print(f"  Task {i}: cpu={service['cpu_demand']:.2f}, mem={service['memory_demand']:.2f}")
    
    print("\n--- Test 6: Clone ---")
    cloned = task_graph.clone()
    print(f"Original: {task_graph}")
    print(f"Cloned: {cloned}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

def test_Z_R_Z_T():
    data, resource_graph, task_graph = load_data()

    generator = ResourceAwareTaskGraphGenerator(resource_graph, data)

    print("\n--- Test 1: Encode Resource Graph ---")
    Z_R = generator.encode_resource_graph()
    print(f"  Z_R (resource embedding): {Z_R}")
    print(f"  Shape: {Z_R.shape}")

    print("\n--- Test 2: Encode Task Graph ---")
    Z_T = generator.encode_task_graph(task_graph)
    print(f"  Z_T (task embedding): {Z_T}")
    print(f"  Shape: {Z_T.shape}")

def test_generate_resource_aware_task_graph():
    data, resource_graph, task_graph = load_data()

    print("\n[Step 1] 初始化 ResourceAwareTaskGraphGenerator...")
    generator = ResourceAwareTaskGraphGenerator(resource_graph, data)
    print(f"  GAT隐藏维度: {generator.gnn_hidden_dim}")
    print(f"  GAT层数: {generator.gnn_num_layers}")
    print(f"  注意力头数: {generator.num_heads}")

    print("\n[Step 2] 调用 generate_resource_aware_task_graph()...")
    result_graph, metrics = generator.generate_resource_aware_task_graph(task_graph)

    print("\n" + "=" * 70)
    print("                        测试结果")
    print("=" * 70)

    print("\n[1] 返回的任务图:")
    print(f"    任务数量: {len(result_graph.tasks)}")
    print(f"    边数量: {len(result_graph.edges)}")

    print("\n[2] 部署评估指标 (metrics):")
    print(f"    R_success (成功率): {metrics.get('R_success', 0):.4f}")
    print(f"    T_latency (时延):   {metrics.get('T_latency', 0):.2f} ms")
    print(f"    F_resched (重调):   {metrics.get('F_resched', 0)}")
    print(f"    match_score (匹配分): {metrics.get('match_score', 0):.4f}")
    print(f"    cpu_utilization:    {metrics.get('cpu_utilization', 0):.4f}")
    print(f"    memory_utilization: {metrics.get('memory_utilization', 0):.4f}")

    print("\n[3] 编码向量:")
    print(f"    Z_R (资源嵌入): shape={generator.Z_R.shape}")
    print(f"    Z_R: {generator.Z_R}")
    print(f"    Z_T (任务嵌入): shape={generator.Z_T.shape}")
    print(f"    Z_T: {generator.Z_T}")

    print("\n[4] 历史记录:")
    print(f"    迭代次数: {len(generator.history)}")
    for i, record in enumerate(generator.history):
        print(f"    迭代 {i + 1}: {record.get('metrics', {})}")

    print("\n" + "=" * 70)
    print("                      测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    # test_task_graph()
    # print("\n\n")
    # test_resource_aware_generator()
    # test_Z_R_Z_T()
    test_generate_resource_aware_task_graph()
