#-*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Env
from dataSet.data import Data


class ExperimentRunner:
    def __init__(self, xml_path='./dataSet/data.xml'):
        self.xml_path = xml_path
        self.results = []
        
    def run_single_experiment(self, max_iterations=20, experiment_id=0):
        print(f"\n{'='*60}")
        print(f"Running Experiment {experiment_id}")
        print(f"{'='*60}")
        
        env = Env(self.xml_path)
        result = env.run_full_experiment(max_iterations=max_iterations)
        
        summary = result['summary']
        
        print(f"\nExperiment {experiment_id} Results:")
        print(f"  Converged: {result['converged']}")
        print(f"  Total Iterations: {summary.get('total_iterations', 0)}")
        print(f"  Final Success Rate: {summary.get('final_metrics', {}).get('success_rate', 0):.4f}")
        print(f"  Final Match Score: {summary.get('final_metrics', {}).get('match_score', 0):.4f}")
        print(f"  Final Latency: {summary.get('final_metrics', {}).get('avg_latency', 0):.2f}ms")
        print(f"  Total Reschedule: {summary.get('total_reschedule', 0)}")
        print(f"  Adaptation Rate: {summary.get('adaptation_rate', 0):.4f}")
        print(f"  Convergence Rate: {summary.get('convergence_rate', 0):.4f}")
        
        return {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'converged': result['converged'],
            'iterations': result['iterations'],
            'summary': summary
        }
        
    def run_multiple_experiments(self, num_experiments=10, max_iterations=20):
        print(f"\n{'#'*60}")
        print(f"# Running {num_experiments} Experiments")
        print(f"# Max Iterations per Experiment: {max_iterations}")
        print(f"{'#'*60}")
        
        all_results = []
        
        for i in range(num_experiments):
            result = self.run_single_experiment(max_iterations=max_iterations, 
                                               experiment_id=i)
            all_results.append(result)
            
        self.results = all_results
        return all_results
        
    def analyze_results(self):
        if not self.results:
            print("No results to analyze!")
            return {}
            
        success_rates = []
        match_scores = []
        latencies = []
        reschedule_counts = []
        adaptation_rates = []
        convergence_rates = []
        iterations_to_converge = []
        
        for result in self.results:
            summary = result['summary']
            success_rates.append(summary.get('final_metrics', {}).get('success_rate', 0))
            match_scores.append(summary.get('final_metrics', {}).get('match_score', 0))
            latencies.append(summary.get('final_metrics', {}).get('avg_latency', 0))
            reschedule_counts.append(summary.get('total_reschedule', 0))
            adaptation_rates.append(summary.get('adaptation_rate', 0))
            convergence_rates.append(summary.get('convergence_rate', 0))
            
            total_iters = summary.get('total_iterations', 0)
            iterations_to_converge.append(total_iters)
            
        analysis = {
            'success_rate': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': np.min(success_rates),
                'max': np.max(success_rates)
            },
            'match_score': {
                'mean': np.mean(match_scores),
                'std': np.std(match_scores),
                'min': np.min(match_scores),
                'max': np.max(match_scores)
            },
            'latency': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'reschedule_count': {
                'mean': np.mean(reschedule_counts),
                'std': np.std(reschedule_counts),
                'min': np.min(reschedule_counts),
                'max': np.max(reschedule_counts)
            },
            'adaptation_rate': {
                'mean': np.mean(adaptation_rates),
                'std': np.std(adaptation_rates)
            },
            'convergence_rate': {
                'mean': np.mean(convergence_rates),
                'std': np.std(convergence_rates)
            },
            'iterations_to_converge': {
                'mean': np.mean(iterations_to_converge),
                'std': np.std(iterations_to_converge)
            }
        }
        
        return analysis
        
    def print_analysis(self):
        analysis = self.analyze_results()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n1. Task Success Rate:")
        print(f"   Mean: {analysis['success_rate']['mean']:.4f} ± {analysis['success_rate']['std']:.4f}")
        print(f"   Range: [{analysis['success_rate']['min']:.4f}, {analysis['success_rate']['max']:.4f}]")
        
        print(f"\n2. Graph Match Score:")
        print(f"   Mean: {analysis['match_score']['mean']:.4f} ± {analysis['match_score']['std']:.4f}")
        print(f"   Range: [{analysis['match_score']['min']:.4f}, {analysis['match_score']['max']:.4f}]")
        
        print(f"\n3. Average Latency (ms):")
        print(f"   Mean: {analysis['latency']['mean']:.2f} ± {analysis['latency']['std']:.2f}")
        print(f"   Range: [{analysis['latency']['min']:.2f}, {analysis['latency']['max']:.2f}]")
        
        print(f"\n4. Reschedule Count:")
        print(f"   Mean: {analysis['reschedule_count']['mean']:.2f} ± {analysis['reschedule_count']['std']:.2f}")
        print(f"   Range: [{analysis['reschedule_count']['min']:.0f}, {analysis['reschedule_count']['max']:.0f}]")
        
        print(f"\n5. Adaptation Rate:")
        print(f"   Mean: {analysis['adaptation_rate']['mean']:.4f} ± {analysis['adaptation_rate']['std']:.4f}")
        
        print(f"\n6. Convergence Rate:")
        print(f"   Mean: {analysis['convergence_rate']['mean']:.4f} ± {analysis['convergence_rate']['std']:.4f}")
        
        print(f"\n7. Iterations to Converge:")
        print(f"   Mean: {analysis['iterations_to_converge']['mean']:.2f} ± {analysis['iterations_to_converge']['std']:.2f}")
        
        return analysis
        
    def export_results(self, filepath='experiment_results.json'):
        analysis = self.analyze_results()
        
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        analysis = convert_to_native(analysis)
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(self.results),
            'analysis': analysis,
            'detailed_results': self.results
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nResults exported to: {filepath}")
        return filepath
        
    def plot_results(self, save_path='experiment_plots'):
        os.makedirs(save_path, exist_ok=True)
        
        if not self.results:
            print("No results to plot!")
            return
            
        iterations_data = []
        success_data = []
        match_data = []
        latency_data = []
        
        for result in self.results:
            for iteration in result['iterations']:
                iterations_data.append(iteration['iteration'])
                success_data.append(iteration['metrics'].get('success_rate', 0))
                match_data.append(iteration['metrics'].get('match_score', 0))
                latency_data.append(iteration['metrics'].get('avg_latency', 0))
                
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(iterations_data, success_data, 'b-', alpha=0.6, label='Success Rate')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Success Rate over Iterations')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(iterations_data, match_data, 'g-', alpha=0.6, label='Match Score')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Match Score')
        axes[0, 1].set_title('Graph Match Score over Iterations')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(iterations_data, latency_data, 'r-', alpha=0.6, label='Latency')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Average Latency over Iterations')
        axes[1, 0].grid(True, alpha=0.3)
        
        analysis = self.analyze_results()
        
        metrics_names = ['Success Rate', 'Match Score', 'Latency\n(normalized)', 'Reschedule\n(count)']
        metrics_means = [
            analysis['success_rate']['mean'],
            analysis['match_score']['mean'],
            analysis['latency']['mean'] / 100,
            analysis['reschedule_count']['mean']
        ]
        metrics_stds = [
            analysis['success_rate']['std'],
            analysis['match_score']['std'],
            analysis['latency']['std'] / 100,
            analysis['reschedule_count']['std']
        ]
        
        x = np.arange(len(metrics_names))
        bars = axes[1, 1].bar(x, metrics_means, yerr=metrics_stds, capsize=5, 
                              color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_names)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Average Metrics Summary')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'experiment_summary.png'), dpi=150)
        plt.close()
        
        print(f"Plots saved to: {save_path}")


def run_paper_experiments():
    print("="*60)
    print("Cloud-Edge Computing Graph Matching Experiment")
    print("Paper: Task-Resource Topology Matching with Feedback Optimization")
    print("="*60)
    
    runner = ExperimentRunner()
    
    runner.run_multiple_experiments(num_experiments=10, max_iterations=15)
    
    analysis = runner.print_analysis()
    
    runner.export_results('./experiment_results.json')
    
    runner.plot_results('./experiment_plots')
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    
    return analysis


if __name__ == '__main__':
    run_paper_experiments()
