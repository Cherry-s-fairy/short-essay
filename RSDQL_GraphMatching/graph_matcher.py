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
        
    def _task_id(self, idx):
        """Convert a 0-based task list index to the actual task ID used in edges
        and _task_resources (which are keyed by the 1-based 'id' field)."""
        if idx < len(self.task_graph.tasks):
            return self.task_graph.tasks[idx]['id']
        return idx

    def _is_feasible(self, service_id, node_id):
        """Return True iff the service's resource demands fit within the node's capacity."""
        if service_id >= len(self.task_graph.tasks) or node_id >= len(self.resource_graph.nodes):
            return False
        service = self.task_graph.tasks[service_id]
        node = self.resource_graph.nodes[node_id]
        cpu_cap = node.get('total_cpu', node.get('cpu', 0))
        mem_cap = node.get('total_memory', node.get('memory', 0))
        return service['cpu_demand'] <= cpu_cap and service['memory_demand'] <= mem_cap

    def _calculate_node_similarity(self, service_id, node_id):
        service_feature = self.task_graph.get_task_feature(service_id)
        node_feature = self.resource_graph.get_node_feature(node_id)

        if service_id < len(self.task_graph.tasks):
            service = self.task_graph.tasks[service_id]
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

        # Hard constraint: infeasible assignment gets zero similarity
        if cpu_demand > cpu_capacity or mem_demand > mem_capacity:
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
        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()
        
        if n_services == 0 or n_nodes == 0:
            return np.zeros((1, 1))
        
        cost_matrix = np.zeros((n_services, n_nodes))

        for i in range(n_services):
            for j in range(n_nodes):
                if not self._is_feasible(i, j):
                    cost_matrix[i][j] = 1e6  # infeasible: never choose this pair
                else:
                    similarity = self._calculate_node_similarity(i, j)
                    cost_matrix[i][j] = 1.0 - similarity

        return cost_matrix
        
    def _hungarian_matching(self):
        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()

        if n_services == 0 or n_nodes == 0:
            return {}

        base_cost = self._build_cost_matrix()  # [n_services, n_nodes]

        if n_services > n_nodes:
            # Many-to-one: tile each physical node column k times so every service
            # gets a unique virtual slot while mapping back to a real node.
            k = (n_services + n_nodes - 1) // n_nodes   # ceil(n_services / n_nodes)
            cost_matrix = np.tile(base_cost, k)[:, :n_services]  # [n_services, n_services]
        else:
            cost_matrix = base_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping = {}
        for i, j in zip(row_ind, col_ind):
            if i < n_services:
                mapping[i] = j % n_nodes   # virtual column → physical node

        # Validate aggregate node capacity; fall back to capacity-aware greedy if violated
        if n_services > n_nodes:
            node_usage = {}
            overloaded = False
            for idx, node_id in mapping.items():
                demand = self.task_graph.get_task_resource_demand(self._task_id(idx))
                if node_id not in node_usage:
                    node_usage[node_id] = {'cpu': 0.0, 'memory': 0.0}
                node_usage[node_id]['cpu'] += demand['cpu']
                node_usage[node_id]['memory'] += demand['memory']
            for node_id, usage in node_usage.items():
                if node_id < len(self.resource_graph.nodes):
                    node = self.resource_graph.nodes[node_id]
                    cpu_cap = node.get('total_cpu', node.get('cpu', 0))
                    mem_cap = node.get('total_memory', node.get('memory', 0))
                    if usage['cpu'] > cpu_cap or usage['memory'] > mem_cap:
                        overloaded = True
                        break
            if overloaded:
                return self._learned_matching()

        return mapping
        
    def _greedy_matching(self):
        """Capacity-weighted greedy (one-to-one or many-to-one).
        Scores each (service, node) pair as similarity * remaining_capacity_fraction
        so that loaded nodes are penalised and services spread evenly.
        """
        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()

        # Remaining capacity per node (starts at full)
        node_remaining = {}
        node_original = {}
        for j in range(n_nodes):
            if j < len(self.resource_graph.nodes):
                node = self.resource_graph.nodes[j]
                cpu_cap = node.get('total_cpu', node.get('cpu', 1))
                mem_cap = node.get('total_memory', node.get('memory', 1))
                node_remaining[j] = {'cpu': cpu_cap, 'memory': mem_cap}
                node_original[j] = {'cpu': max(cpu_cap, 1e-9), 'memory': max(mem_cap, 1e-9)}

        mapping = {}
        used_nodes = set()

        for _ in range(n_services):
            best_score = -1.0
            best_service = -1
            best_node = -1

            for i in range(n_services):
                if i in mapping:
                    continue
                demand = self.task_graph.get_task_resource_demand(self._task_id(i))
                for j in range(n_nodes):
                    if n_services <= n_nodes and j in used_nodes:
                        continue
                    remaining = node_remaining.get(j, {'cpu': 0, 'memory': 0})
                    if demand['cpu'] > remaining['cpu'] or demand['memory'] > remaining['memory']:
                        continue
                    similarity = self._calculate_node_similarity(i, j)
                    # Scale by fraction of capacity still available (spread load evenly)
                    cap_frac = (remaining['cpu'] / node_original[j]['cpu'] +
                                remaining['memory'] / node_original[j]['memory']) / 2.0
                    score = similarity * cap_frac
                    if score > best_score:
                        best_score = score
                        best_service = i
                        best_node = j

            if best_service >= 0 and best_node >= 0:
                mapping[best_service] = best_node
                demand = self.task_graph.get_task_resource_demand(self._task_id(best_service))
                node_remaining[best_node]['cpu'] -= demand['cpu']
                node_remaining[best_node]['memory'] -= demand['memory']
                if n_services <= n_nodes:
                    used_nodes.add(best_node)

        return mapping
        
    def _learned_matching(self):
        """Priority-weighted greedy: assigns high-priority services first to their
        best feasible node, tracking remaining capacity to avoid per-node overload.
        Falls back to hungarian if not all services can be placed."""
        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()

        # Sort services by priority descending so critical ones get first pick
        service_order = sorted(
            range(n_services),
            key=lambda i: (self.task_graph.tasks[i].get('priority', 0)
                           if i < len(self.task_graph.tasks) else 0),
            reverse=True
        )

        # Track remaining capacity per node (used for many-to-one scenarios)
        node_remaining = {}
        for j in range(n_nodes):
            if j < len(self.resource_graph.nodes):
                node = self.resource_graph.nodes[j]
                node_remaining[j] = {
                    'cpu': node.get('total_cpu', node.get('cpu', 0)),
                    'memory': node.get('total_memory', node.get('memory', 0))
                }

        mapping = {}
        used_nodes = set()

        for service_id in service_order:
            demand = self.task_graph.get_task_resource_demand(self._task_id(service_id))
            best_node, best_score = -1, -1.0

            for node_id in range(n_nodes):
                # One-to-one mode: skip already-used nodes
                if n_services <= n_nodes and node_id in used_nodes:
                    continue
                # Check remaining (not original) capacity
                remaining = node_remaining.get(node_id, {'cpu': 0, 'memory': 0})
                if demand['cpu'] > remaining['cpu'] or demand['memory'] > remaining['memory']:
                    continue
                score = self._calculate_node_similarity(service_id, node_id)
                if score > best_score:
                    best_score = score
                    best_node = node_id

            if best_node >= 0:
                mapping[service_id] = best_node
                node_remaining[best_node]['cpu'] -= demand['cpu']
                node_remaining[best_node]['memory'] -= demand['memory']
                if n_services <= n_nodes:
                    used_nodes.add(best_node)

        # Fall back to hungarian if some services could not be placed
        return mapping if len(mapping) == n_services else self._hungarian_matching()
        
    def _heuristic_matching(self):
        """Communication-aware greedy (one-to-one or many-to-one).
        Assigns services in descending order of their total communication weight,
        choosing the node that minimises expected communication distance to already-placed
        neighbours — so heavily communicating services end up on nearby nodes.
        """
        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()

        # Pre-compute total communication weight per service
        comm_weight = {}
        for i in range(n_services):
            task_id = self._task_id(i)
            total = sum(e['weight'] for e in self.task_graph.edges
                        if e['src'] == task_id or e['dst'] == task_id)
            comm_weight[i] = total

        # Sort by descending communication load
        service_order = sorted(range(n_services), key=lambda i: comm_weight[i], reverse=True)

        # Remaining capacity per node
        node_remaining = {}
        for j in range(n_nodes):
            if j < len(self.resource_graph.nodes):
                node = self.resource_graph.nodes[j]
                node_remaining[j] = {
                    'cpu': node.get('total_cpu', node.get('cpu', 0)),
                    'memory': node.get('total_memory', node.get('memory', 0))
                }

        mapping = {}
        used_nodes = set()

        for service_id in service_order:
            task_id_s = self._task_id(service_id)
            demand = self.task_graph.get_task_resource_demand(task_id_s)
            best_node, best_score = -1, float('inf')

            for node_id in range(n_nodes):
                if n_services <= n_nodes and node_id in used_nodes:
                    continue
                remaining = node_remaining.get(node_id, {'cpu': 0, 'memory': 0})
                if demand['cpu'] > remaining['cpu'] or demand['memory'] > remaining['memory']:
                    continue
                # Communication distance to already-placed neighbours
                comm_dist = 0.0
                for placed_idx, placed_node in mapping.items():
                    edge_w = self.task_graph.get_edge_weight(task_id_s, self._task_id(placed_idx))
                    if edge_w > 0:
                        d = self.resource_graph.get_shortest_path_distance(node_id, placed_node)
                        if d < float('inf'):
                            comm_dist += edge_w * d
                # Lower comm_dist is better; break ties with similarity
                sim = self._calculate_node_similarity(service_id, node_id)
                score = comm_dist - sim * 0.1   # small bonus for resource fit
                if score < best_score:
                    best_score = score
                    best_node = node_id

            if best_node >= 0:
                mapping[service_id] = best_node
                demand = self.task_graph.get_task_resource_demand(task_id_s)
                node_remaining[best_node]['cpu'] -= demand['cpu']
                node_remaining[best_node]['memory'] -= demand['memory']
                if n_services <= n_nodes:
                    used_nodes.add(best_node)

        return mapping if len(mapping) == n_services else self._hungarian_matching()
        
    def calculate_match_score(self):
        if not self.mapping:
            return 0.0

        n_total = self.task_graph.get_task_count()

        # Deployment completeness: fraction of services actually placed
        completeness = len(self.mapping) / max(n_total, 1)

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

        # Weight completeness as the primary gate: partial deployment is penalised
        self.match_score = completeness * (0.7 * node_score + 0.3 * edge_score)
        return self.match_score
        
    def validate_mapping(self):
        if not self.mapping:
            return False, "No mapping available"

        n_services = self.task_graph.get_task_count()
        n_nodes = self.resource_graph.get_node_count()

        if len(self.mapping) != n_services:
            return False, f"Mapping incomplete: {len(self.mapping)}/{n_services}"

        # One-to-one constraint only applies when there are enough nodes;
        # when n_services > n_nodes, multiple services must share nodes (many-to-one).
        if n_services <= n_nodes:
            if len(set(self.mapping.values())) != len(self.mapping):
                return False, "Duplicate node assignment"

        # Aggregate resource usage per node to detect cumulative overload
        node_usage = {}
        for service_id, node_id in self.mapping.items():
            demand = self.task_graph.get_task_resource_demand(self._task_id(service_id))
            if node_id not in node_usage:
                node_usage[node_id] = {'cpu': 0, 'memory': 0}
            node_usage[node_id]['cpu'] += demand['cpu']
            node_usage[node_id]['memory'] += demand['memory']

        for node_id, usage in node_usage.items():
            if node_id < len(self.resource_graph.nodes):
                node = self.resource_graph.nodes[node_id]
                cpu_cap = node.get('total_cpu', node.get('cpu', 0))
                mem_cap = node.get('total_memory', node.get('memory', 0))
                if usage['cpu'] > cpu_cap:
                    return False, f"Node {node_id} CPU overloaded: {usage['cpu']} > {cpu_cap}"
                if usage['memory'] > mem_cap:
                    return False, f"Node {node_id} Memory overloaded: {usage['memory']} > {mem_cap}"

        return True, "Valid mapping"
        
    def get_mapping(self):
        return self.mapping
        
    def get_deployment_plan(self):
        if not self.mapping:
            return []
            
        deployment_plan = []
        
        for service_id, node_id in self.mapping.items():
            if service_id < len(self.task_graph.tasks):
                service = self.task_graph.tasks[service_id]
                deployment_plan.append({
                    'service_id': service_id,            # 0-based list index
                    'task_id': self._task_id(service_id),  # actual task ID (1-based)
                    'node_id': node_id,
                    'cpu_demand': service['cpu_demand'],
                    'memory_demand': service['memory_demand']
                })
                
        return deployment_plan
