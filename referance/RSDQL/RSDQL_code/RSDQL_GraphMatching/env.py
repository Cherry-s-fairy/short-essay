#-*- coding: utf-8 -*-

import numpy as np
from dataSet.data import Data
from resource_graph import ResourceTopologyGraph
from task_graph import TaskTopologyGraph
from graph_matcher import GraphMatcher
from feedback_optimizer import FeedbackOptimizer, TaskGraphGenerator


class Env:
    def __init__(self, xml_path='./dataSet/data.xml'):
        self.data = Data(xml_path)
        self.resource_graph = ResourceTopologyGraph(self.data).build_from_data()
        self.task_graph_generator = TaskGraphGenerator(self.data)
        self.task_graph = self.task_graph_generator.generate_initial_graph()
        self.matcher = GraphMatcher(self.resource_graph, self.task_graph)
        self.feedback_optimizer = FeedbackOptimizer(self.task_graph, self.resource_graph)
        
        self.state = None
        self.current_mapping = None
        self.episode_count = 0
        self.iteration_count = 0
        self.max_iterations = 20
        self.convergence_threshold = 0.05
        
    def reset(self):
        self.episode_count += 1
        self.iteration_count = 0
        self.matcher = GraphMatcher(self.resource_graph, self.task_graph)
        self.current_mapping = self.matcher.match(method='hungarian')
        
        valid, message = self.matcher.validate_mapping()
        
        self.state = self._build_state()
        
        return self.state, valid, message
        
    def _build_state(self):
        resource_features = self.resource_graph.get_all_node_features()
        task_features = self.task_graph.get_all_service_features()
        
        state = {
            'resource_features': resource_features,
            'task_features': task_features,
            'communication_matrix': self.task_graph.communication_matrix,
            'distance_matrix': self.resource_graph.distance_matrix,
            'mapping': self.current_mapping,
            'match_score': self.matcher.match_score,
            'task_topology_stats': self.task_graph.get_topology_stats(),
            'iteration': self.iteration_count
        }
        
        return state
        
    def step(self, action=None):
        if self.current_mapping is None:
            return self.state, -100, True, {}
            
        deployment_plan = self.matcher.get_deployment_plan()
        
        cost, comm_cost, var_cost = self._calculate_cost(deployment_plan)
        
        reward = self._calculate_reward(cost)
        
        metrics = self._simulate_deployment_metrics(deployment_plan)
        
        self.feedback_optimizer.collect_metrics(metrics)
        
        feedback = self.feedback_optimizer.generate_feedback()
        
        if feedback.get('adjustment_type') != 'none' and not feedback.get('convergence', False):
            if self.iteration_count < self.max_iterations:
                self.task_graph = self.feedback_optimizer.adjust_task_graph(feedback)
                self.matcher = GraphMatcher(self.resource_graph, self.task_graph)
                self.current_mapping = self.matcher.match(method='hungarian')
                self.iteration_count += 1
        
        done = True
        
        info = {
            'match_score': self.matcher.match_score,
            'cost': cost,
            'comm_cost': comm_cost,
            'var_cost': var_cost,
            'metrics': metrics,
            'feedback': feedback,
            'deployment_plan': deployment_plan,
            'iteration': self.iteration_count,
            'converged': feedback.get('convergence', False)
        }
        
        self.state = self._build_state()
        
        return self.state, reward, done, info
        
    def _calculate_cost(self, deployment_plan):
        comm_cost = 0.0
        
        for i, service1 in enumerate(deployment_plan):
            for j, service2 in enumerate(deployment_plan):
                if i != j:
                    weight = self.task_graph.get_edge_weight(i, j)
                    if weight > 0:
                        node1 = service1['node_id']
                        node2 = service2['node_id']
                        distance = self.resource_graph.get_shortest_path_distance(node1, node2)
                        if distance < float('inf'):
                            comm_cost += weight * distance
                        
        comm_cost = comm_cost / 2.0
        
        var_cost = self._calculate_variance_cost(deployment_plan)
        
        total_cost = 0.5 * comm_cost + 0.5 * var_cost
        
        return total_cost, comm_cost, var_cost
        
    def _calculate_variance_cost(self, deployment_plan):
        node_loads = {}
        
        for plan in deployment_plan:
            node_id = plan['node_id']
            if node_id not in node_loads:
                node_loads[node_id] = {'cpu': 0, 'memory': 0}
            node_loads[node_id]['cpu'] += plan['cpu_demand']
            node_loads[node_id]['memory'] += plan['memory_demand']
            
        cpu_loads = [load['cpu'] for load in node_loads.values()]
        mem_loads = [load['memory'] for load in node_loads.values()]
        
        cpu_var = np.var(cpu_loads) if len(cpu_loads) > 1 else 0
        mem_var = np.var(mem_loads) if len(mem_loads) > 1 else 0
        
        return cpu_var + mem_var
        
    def _calculate_reward(self, cost):
        max_cost = 500.0
        reward = -cost
        
        if cost < max_cost:
            reward += (max_cost - cost) / max_cost * 50
            
        return reward
        
    def _simulate_deployment_metrics(self, deployment_plan):
        match_score = self.matcher.match_score
        
        success_rate = match_score
        
        if match_score < 0.5:
            success_rate = match_score * 0.8
        elif match_score < 0.7:
            success_rate = match_score * 0.9
        else:
            success_rate = 0.85 + match_score * 0.15
            
        success_rate = min(1.0, success_rate)
        
        total_distance = 0
        edge_count = 0
        latency_samples = []
        
        for i in range(len(deployment_plan)):
            for j in range(len(deployment_plan)):
                weight = self.task_graph.get_edge_weight(i, j)
                if weight > 0:
                    node_i = deployment_plan[i]['node_id']
                    node_j = deployment_plan[j]['node_id']
                    distance = self.resource_graph.get_shortest_path_distance(node_i, node_j)
                    if distance < float('inf'):
                        total_distance += distance
                        latency_samples.append(distance * 10)
                        edge_count += 1
                    
        avg_distance = total_distance / edge_count if edge_count > 0 else 0
        avg_latency = avg_distance * 10
        
        latency_variance = np.var(latency_samples) if len(latency_samples) > 1 else 0
        
        reschedule_count = 0
        if match_score < 0.7:
            reschedule_count = int((1 - match_score) * 10)
        if match_score < 0.5:
            reschedule_count += 3
            
        node_resources = self.resource_graph.get_node_resource_capacity()
        used_resources = {'cpu': 0, 'memory': 0}
        
        node_loads = {}
        for plan in deployment_plan:
            node_id = plan['node_id']
            if node_id not in node_loads:
                node_loads[node_id] = {'cpu': 0, 'memory': 0}
            node_loads[node_id]['cpu'] += plan['cpu_demand']
            node_loads[node_id]['memory'] += plan['memory_demand']
            used_resources['cpu'] += plan['cpu_demand']
            used_resources['memory'] += plan['memory_demand']
            
        total_capacity = sum(n['cpu'] + n['memory'] for n in node_resources) if node_resources else 1
        resource_utilization = (used_resources['cpu'] + used_resources['memory']) / total_capacity if total_capacity > 0 else 0
        
        cpu_loads = [l['cpu'] for l in node_loads.values()]
        mem_loads = [l['memory'] for l in node_loads.values()]
        
        if cpu_loads and mem_loads:
            total_load = [cpu_loads[i] + mem_loads[i] for i in range(len(cpu_loads))]
            if len(total_load) > 1:
                load_balance_score = 1.0 - min(1.0, np.std(total_load) / (np.mean(total_load) + 0.001))
            else:
                load_balance_score = 1.0
        else:
            load_balance_score = 0.0
            
        network_load = total_distance / max(1, edge_count)
        
        edge_count_deployed = 0
        cloud_count = 0
        edge_node_count = self._count_edge_nodes()
        
        for plan in deployment_plan:
            node_id = plan['node_id']
            if node_id < edge_node_count:
                edge_count_deployed += 1
            else:
                cloud_count += 1
                
        total_services = len(deployment_plan)
        edge_cloud_ratio = edge_count_deployed / total_services if total_services > 0 else 0.5
        
        service_availability = success_rate * (1.0 - reschedule_count / max(1, self.max_iterations))
        
        deployment_cost = self._calculate_deployment_cost(deployment_plan, total_distance)
        
        return {
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'latency_variance': latency_variance,
            'reschedule_count': reschedule_count,
            'resource_utilization': resource_utilization,
            'communication_cost': total_distance,
            'match_score': match_score,
            'deployment_cost': deployment_cost,
            'service_availability': service_availability,
            'network_load': network_load,
            'load_balance_score': load_balance_score,
            'edge_cloud_ratio': edge_cloud_ratio
        }
        
    def _count_edge_nodes(self):
        return max(1, self.resource_graph.get_node_count() // 2)
    
    def _calculate_deployment_cost(self, deployment_plan, total_distance):
        var_cost = self._calculate_variance_cost(deployment_plan)
        return total_distance * 0.5 + var_cost * 0.5
        
    def run_full_experiment(self, max_iterations=None):
        if max_iterations is not None:
            self.max_iterations = max_iterations
            
        experiment_results = []
        
        state, valid, message = self.reset()
        
        while self.iteration_count < self.max_iterations:
            next_state, reward, done, info = self.step()
            
            experiment_results.append({
                'iteration': self.iteration_count,
                'match_score': info.get('match_score', 0),
                'metrics': info.get('metrics', {}),
                'feedback': info.get('feedback', {}),
                'cost': info.get('cost', 0)
            })
            
            if info.get('converged', False):
                break
                
            state = next_state
            
        summary = self.feedback_optimizer.get_experiment_summary()
        
        return {
            'iterations': experiment_results,
            'summary': summary,
            'final_state': state,
            'converged': self.iteration_count < self.max_iterations
        }
        
    def get_resource_graph(self):
        return self.resource_graph
        
    def get_task_graph(self):
        return self.task_graph
        
    def get_matcher(self):
        return self.matcher
        
    def get_feedback_optimizer(self):
        return self.feedback_optimizer


def test_env():
    env = Env()
    print("Resource Graph:", env.resource_graph)
    print("Task Graph:", env.task_graph)
    
    state, valid, message = env.reset()
    print(f"Reset: valid={valid}, message={message}")
    print(f"Match Score: {state['match_score']:.4f}")
    
    next_state, reward, done, info = env.step()
    print(f"Step: reward={reward:.2f}, done={done}")
    print(f"Cost: {info['cost']:.2f}")
    print(f"Metrics: {info['metrics']}")
    
    print("\n=== Running Full Experiment ===")
    result = env.run_full_experiment(max_iterations=10)
    print(f"Converged: {result['converged']}")
    print(f"Summary: {result['summary']}")


if __name__ == '__main__':
    test_env()
