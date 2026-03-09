#-*- coding: utf-8 -*-

import numpy as np
import json
import os
from datetime import datetime


class DeploymentMetrics:
    def __init__(self):
        self.success_rate = 0.0
        self.avg_latency = 0.0
        self.reschedule_count = 0
        self.resource_utilization = 0.0
        self.communication_cost = 0.0
        self.match_score = 0.0
        self.deployment_cost = 0.0
        self.latency_variance = 0.0
        self.service_availability = 0.0
        self.network_load = 0.0
        self.load_balance_score = 0.0
        self.edge_cloud_ratio = 0.0
        self.iteration = 0
        
    def to_dict(self):
        return {
            'success_rate': self.success_rate,
            'avg_latency': self.avg_latency,
            'reschedule_count': self.reschedule_count,
            'resource_utilization': self.resource_utilization,
            'communication_cost': self.communication_cost,
            'match_score': self.match_score,
            'deployment_cost': self.deployment_cost,
            'latency_variance': self.latency_variance,
            'service_availability': self.service_availability,
            'network_load': self.network_load,
            'load_balance_score': self.load_balance_score,
            'edge_cloud_ratio': self.edge_cloud_ratio,
            'iteration': self.iteration
        }
        
    def __repr__(self):
        return (f"Metrics(success={self.success_rate:.2%}, "
                f"latency={self.avg_latency:.2f}ms, "
                f"reschedule={self.reschedule_count}, "
                f"util={self.resource_utilization:.2%}, "
                f"match={self.match_score:.3f})")


class FeedbackOptimizer:
    def __init__(self, task_graph, resource_graph=None):
        self.task_graph = task_graph
        self.resource_graph = resource_graph
        self.history = []
        self.current_iteration = 0
        self.adaptation_threshold = {
            'success_rate': 0.8,
            'latency': 100.0,
            'reschedule': 3,
            'utilization': 0.5,
            'match_score': 0.6,
            'r_max': 0.9,
            'tau_max': 100.0,
            'comm_cost_threshold': 50.0
        }
        self.weight_config = {
            'success_rate': 0.3,
            'latency': 0.25,
            'reschedule': 0.2,
            'utilization': 0.15,
            'load_balance': 0.1,
            'lambda1': 0.5,
            'lambda2': 0.5
        }
        self.best_metrics = None
        self.convergence_count = 0
        self.max_convergence_iterations = 5
        self.critical_tasks = []
        
    def collect_metrics(self, deployment_result):
        metrics = DeploymentMetrics()
        metrics.iteration = self.current_iteration
        
        if isinstance(deployment_result, dict):
            metrics.success_rate = deployment_result.get('success_rate', 0.0)
            metrics.avg_latency = deployment_result.get('avg_latency', 0.0)
            metrics.reschedule_count = deployment_result.get('reschedule_count', 0)
            metrics.resource_utilization = deployment_result.get('resource_utilization', 0.0)
            metrics.communication_cost = deployment_result.get('communication_cost', 0.0)
            metrics.match_score = deployment_result.get('match_score', 0.0)
            metrics.deployment_cost = deployment_result.get('deployment_cost', 0.0)
            metrics.latency_variance = deployment_result.get('latency_variance', 0.0)
            metrics.service_availability = deployment_result.get('service_availability', 1.0)
            metrics.network_load = deployment_result.get('network_load', 0.0)
            metrics.load_balance_score = deployment_result.get('load_balance_score', 0.0)
            metrics.edge_cloud_ratio = deployment_result.get('edge_cloud_ratio', 0.5)
        else:
            metrics.success_rate = 1.0 if deployment_result else 0.0
            
        if self.best_metrics is None or metrics.success_rate > self.best_metrics.success_rate:
            self.best_metrics = metrics
            
        self.history.append({
            'iteration': self.current_iteration,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        self.current_iteration += 1
        
    def generate_feedback(self):
        if not self.history:
            return {'adjustment_type': 'none', 'reason': 'no_history'}
            
        recent_metrics = self.history[-1]['metrics']
        
        adjustment_type = 'none'
        params = {}
        reasons = []
        
        r_success = recent_metrics.success_rate
        t_latency = recent_metrics.avg_latency
        f_resched = recent_metrics.reschedule_count
        r_max = self.adaptation_threshold['r_max']
        tau_max = self.adaptation_threshold['tau_max']
        comm_cost_threshold = self.adaptation_threshold['comm_cost_threshold']
        
        self._identify_critical_tasks(recent_metrics)
        
        if r_success < r_max and t_latency > tau_max:
            for task in self.critical_tasks:
                task_latency = task.get('latency', 0)
                task_success = task.get('success_rate', 0)
                
                if task_success > r_max and task_latency > comm_cost_threshold:
                    adjustment_type = 'split_high_latency'
                    params['latency_threshold'] = tau_max
                    params['success_threshold'] = r_max
                    params['scale_factor'] = 0.5
                    reasons.append('critical_task_split')
                    break
                    
        if t_latency > tau_max and recent_metrics.communication_cost > comm_cost_threshold:
            adjustment_type = 'merge_communication'
            params['comm_cost_threshold'] = comm_cost_threshold
            params['merge_dependent'] = True
            reasons.append('merge_high_communication')
            
        if adjustment_type == 'none':
            adjustment_type = 'adjust_priority'
            params['lambda1'] = self.weight_config['lambda1']
            params['lambda2'] = self.weight_config['lambda2']
            params['threshold'] = 0.5
            reasons.append('adjust_priority')
            
        if recent_metrics.match_score < self.adaptation_threshold['match_score']:
            if recent_metrics.success_rate < self.adaptation_threshold['success_rate']:
                params['threshold'] = 0.5
                params['scale_factor'] = 0.9
                reasons.append('low_match_and_success')
            else:
                params['threshold'] = 0.7
                reasons.append('low_match')
                
        if t_latency > tau_max:
            adjustment_type = 'modify_dependency'
            params['threshold'] = max(params.get('threshold', 0.5) - 0.2, 0.2)
            reasons.append('high_latency')
            
        if f_resched > self.adaptation_threshold['reschedule']:
            if adjustment_type == 'none':
                adjustment_type = 'decrease_capacity'
            params['scale_factor'] = max(0.7, 1.0 - f_resched * 0.1)
            params['strategy'] = 'uniform'
            reasons.append('high_reschedule')
            
        if recent_metrics.resource_utilization < self.adaptation_threshold['utilization']:
            if adjustment_type == 'none':
                adjustment_type = 'decrease_capacity'
            params['scale_factor'] = min(params.get('scale_factor', 1.0), 0.8)
            params['strategy'] = 'selective'
            reasons.append('low_utilization')
            
        if recent_metrics.load_balance_score < 0.3:
            adjustment_type = 'rebalance_topology'
            params['target_balance'] = 0.5
            reasons.append('poor_load_balance')
            
        if adjustment_type == 'none':
            if recent_metrics.match_score > 0.8:
                adjustment_type = 'fine_tune'
                params['scale_factor'] = 1.05
                reasons.append('good_match_optimize')
        
        self._check_convergence(recent_metrics)
        
        feedback = {
            'adjustment_type': adjustment_type,
            'reasons': reasons,
            'r_success': r_success,
            't_latency': t_latency,
            'f_resched': f_resched,
            'success_rate': recent_metrics.success_rate,
            'latency': recent_metrics.avg_latency,
            'reschedule_count': recent_metrics.reschedule_count,
            'utilization': recent_metrics.resource_utilization,
            'match_score': recent_metrics.match_score,
            'communication_cost': recent_metrics.communication_cost,
            'load_balance_score': recent_metrics.load_balance_score,
            'iteration': recent_metrics.iteration,
            'critical_tasks': self.critical_tasks,
            'convergence': self.convergence_count >= self.max_convergence_iterations,
            **params
        }
        
        return feedback
        
    def _identify_critical_tasks(self, metrics):
        self.critical_tasks = []
        
        if not self.task_graph or not self.task_graph.services:
            return
            
        task_count = len(self.task_graph.services)
        
        for i, service in enumerate(self.task_graph.services):
            service_latency = 0
            for edge in self.task_graph.edges:
                if edge['src'] == service['id'] or edge['dst'] == service['id']:
                    service_latency += edge.get('latency', 0)
            
            task_success = metrics.success_rate * (1 - service_latency / max(1, self.adaptation_threshold['tau_max']))
            task_success = max(0, min(1, task_success))
            
            if service_latency > self.adaptation_threshold['tau_max'] or task_success < self.adaptation_threshold['success_rate']:
                self.critical_tasks.append({
                    'id': service['id'],
                    'latency': service_latency,
                    'success_rate': task_success,
                    'cpu_demand': service['cpu_demand'],
                    'memory_demand': service['memory_demand']
                })
        
    def _check_convergence(self, metrics):
        if len(self.history) < 2:
            return
            
        prev_metrics = self.history[-2]['metrics']
        
        success_diff = abs(metrics.success_rate - prev_metrics.success_rate)
        latency_diff = abs(metrics.avg_latency - prev_metrics.avg_latency)
        match_diff = abs(metrics.match_score - prev_metrics.match_score)
        
        if success_diff < 0.01 and latency_diff < 1.0 and match_diff < 0.01:
            self.convergence_count += 1
        else:
            self.convergence_count = 0
            
    def adjust_task_graph(self, feedback):
        adjustment_type = feedback.get('adjustment_type', 'none')
        
        if adjustment_type != 'none':
            self.task_graph.adjust_topology(feedback)
            
        return self.task_graph
        
    def get_best_feedback(self):
        if not self.history:
            return None
            
        best = max(self.history, 
                   key=lambda x: (x['metrics'].success_rate * 100 
                               - x['metrics'].avg_latency 
                               - x['metrics'].reschedule_count * 10
                               + x['metrics'].match_score * 50))
        return best['metrics']
        
    def get_adaptation_rate(self):
        if len(self.history) < 2:
            return 0.0
            
        recent = self.history[-1]['metrics']
        initial = self.history[0]['metrics']
        
        success_diff = recent.success_rate - initial.success_rate
        latency_diff = initial.avg_latency - recent.avg_latency if initial.avg_latency > 0 else 0
        
        adaptation = (0.5 * success_diff + 
                     0.3 * (latency_diff / max(initial.avg_latency, 1)) +
                     0.2 * (recent.match_score - initial.match_score))
        return max(0, min(1.0, adaptation))
        
    def get_convergence_rate(self):
        if len(self.history) < 3:
            return 0.0
            
        convergence_points = sum(1 for h in self.history 
                                if h['iteration'] > 0 and 
                                h['iteration'] < len(self.history) - 1 and
                                abs(h['metrics'].match_score - self.history[-1]['metrics'].match_score) < 0.05)
        return convergence_points / max(1, len(self.history) - 2)
        
    def get_experiment_summary(self):
        if not self.history:
            return {}
            
        metrics_list = [h['metrics'] for h in self.history]
        
        return {
            'total_iterations': len(self.history),
            'convergence_iterations': self.convergence_count,
            'convergence_rate': self.get_convergence_rate(),
            'adaptation_rate': self.get_adaptation_rate(),
            'initial_metrics': metrics_list[0].to_dict() if metrics_list else {},
            'final_metrics': metrics_list[-1].to_dict() if metrics_list else {},
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else {},
            'avg_success_rate': np.mean([m.success_rate for m in metrics_list]),
            'avg_latency': np.mean([m.avg_latency for m in metrics_list]),
            'avg_match_score': np.mean([m.match_score for m in metrics_list]),
            'total_reschedule': sum(m.reschedule_count for m in metrics_list)
        }
        
    def export_metrics(self, filepath):
        summary = self.get_experiment_summary()
        
        detailed_history = []
        for h in self.history:
            detailed_history.append({
                'iteration': h['iteration'],
                'timestamp': h['timestamp'],
                **h['metrics'].to_dict()
            })
            
        data = {
            'summary': summary,
            'history': detailed_history
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
        
    def reset(self):
        self.history = []
        self.current_iteration = 0
        self.best_metrics = None
        self.convergence_count = 0


class TaskGraphGenerator:
    def __init__(self, data):
        self.data = data
        self.current_task_graph = None
        self._optimizer = None
        
    def generate_initial_graph(self):
        from task_graph import TaskTopologyGraph
        self.current_task_graph = TaskTopologyGraph(self.data).build_from_data()
        self._optimizer = FeedbackOptimizer(self.current_task_graph)
        return self.current_task_graph
        
    def adjust_graph(self, feedback):
        if self.current_task_graph is None:
            return self.generate_initial_graph()
            
        if self._optimizer is None:
            self._optimizer = FeedbackOptimizer(self.current_task_graph)
            
        self._optimizer.adjust_task_graph(feedback)
        self.current_task_graph = self._optimizer.task_graph
        
        return self.current_task_graph
        
    def get_current_graph(self):
        return self.current_task_graph
