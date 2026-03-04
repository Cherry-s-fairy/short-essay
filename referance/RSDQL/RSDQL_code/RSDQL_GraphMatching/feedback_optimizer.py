#-*- coding: utf-8 -*-

import numpy as np


class DeploymentMetrics:
    def __init__(self):
        self.success_rate = 0.0
        self.avg_latency = 0.0
        self.reschedule_count = 0
        self.resource_utilization = 0.0
        self.communication_cost = 0.0
        
    def __repr__(self):
        return (f"Metrics(success={self.success_rate:.2%}, "
                f"latency={self.avg_latency:.2f}ms, "
                f"reschedule={self.reschedule_count}, "
                f"util={self.resource_utilization:.2%})")


class FeedbackOptimizer:
    def __init__(self, task_graph):
        self.task_graph = task_graph
        self.history = []
        self.current_iteration = 0
        self.adaptation_threshold = {
            'success_rate': 0.8,
            'latency': 100.0,
            'reschedule': 3,
            'utilization': 0.5
        }
        
    def collect_metrics(self, deployment_result):
        metrics = DeploymentMetrics()
        
        if isinstance(deployment_result, dict):
            metrics.success_rate = deployment_result.get('success_rate', 0.0)
            metrics.avg_latency = deployment_result.get('avg_latency', 0.0)
            metrics.reschedule_count = deployment_result.get('reschedule_count', 0)
            metrics.resource_utilization = deployment_result.get('resource_utilization', 0.0)
            metrics.communication_cost = deployment_result.get('communication_cost', 0.0)
        else:
            metrics.success_rate = 1.0 if deployment_result else 0.0
            
        self.history.append({
            'iteration': self.current_iteration,
            'metrics': metrics
        })
        
        self.current_iteration += 1
        
    def generate_feedback(self):
        if not self.history:
            return {'adjustment_type': 'none'}
            
        recent_metrics = self.history[-1]['metrics']
        
        adjustment_type = 'none'
        params = {}
        
        if recent_metrics.success_rate < self.adaptation_threshold['success_rate']:
            adjustment_type = 'reduce_dependency'
            params['threshold'] = 0.5
        elif recent_metrics.avg_latency > self.adaptation_threshold['latency']:
            adjustment_type = 'reduce_dependency'
            params['threshold'] = 0.3
        elif recent_metrics.reschedule_count > self.adaptation_threshold['reschedule']:
            adjustment_type = 'decrease_capacity'
            params['scale_factor'] = 0.9
        elif recent_metrics.resource_utilization < self.adaptation_threshold['utilization']:
            adjustment_type = 'decrease_capacity'
            params['scale_factor'] = 0.8
        else:
            adjustment_type = 'none'
            
        feedback = {
            'adjustment_type': adjustment_type,
            'success_rate': recent_metrics.success_rate,
            'latency': recent_metrics.avg_latency,
            'reschedule_count': recent_metrics.reschedule_count,
            'utilization': recent_metrics.resource_utilization,
            'communication_cost': recent_metrics.communication_cost,
            **params
        }
        
        return feedback
        
    def adjust_task_graph(self, feedback):
        adjustment_type = feedback.get('adjustment_type', 'none')
        
        if adjustment_type != 'none':
            self.task_graph.adjust_topology(feedback)
            
        return self.task_graph
        
    def get_best_feedback(self):
        if not self.history:
            return None
            
        best = max(self.history, 
                   key=lambda x: x['metrics'].success_rate * 100 
                               - x['metrics'].avg_latency 
                               - x['metrics'].reschedule_count * 10)
        return best['metrics']
        
    def get_adaptation_rate(self):
        if len(self.history) < 2:
            return 0.0
            
        recent = self.history[-1]['metrics']
        initial = self.history[0]['metrics']
        
        success_diff = recent.success_rate - initial.success_rate
        latency_diff = initial.avg_latency - recent.avg_latency
        
        adaptation = 0.5 * success_diff + 0.5 * (latency_diff / max(initial.avg_latency, 1))
        return max(0, adaptation)
        
    def reset(self):
        self.history = []
        self.current_iteration = 0


class TaskGraphGenerator:
    def __init__(self, data):
        self.data = data
        self.current_task_graph = None
        
    def generate_initial_graph(self):
        from task_graph import TaskTopologyGraph
        self.current_task_graph = TaskTopologyGraph(self.data).build_from_data()
        return self.current_task_graph
        
    def adjust_graph(self, feedback):
        if self.current_task_graph is None:
            return self.generate_initial_graph()
            
        optimizer = FeedbackOptimizer(self.current_task_graph)
        self.current_task_graph = optimizer.adjust_task_graph(feedback)
        
        return self.current_task_graph
        
    def get_current_graph(self):
        return self.current_task_graph
