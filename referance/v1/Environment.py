from dataSet.data import Data
import HyperParams as hp
import numpy as np
import math

data = Data()
NodeNumber = data.NodeNumber
ContainerNumber = data.ContainerNumber
ServiceNumber = data.ServiceNumber
ResourceType = data.ResourceType
service_containerNumber = data.service_containerNum  # 每个服务所需容器数列表
service_container = data.service_container  # 每个服务所启动的容器列表
service_container_relationship = data.container_service  # 微服务和容器的映射
container_state_raw = data.container_state_queue[:]  # 容器状态（cpu,mem）队列
# [[1, 0, ..., 1, cpu, mem], [], []], 前面container_num个元素代表该container是否部署在该节点上，最后两个属性是cpu和mem
node_state_raw = [0] * (NodeNumber * (ContainerNumber + 2))


class Environment:
    def __init__(self):
        self.max_reward = None
        self.min_cost = None
        self.all_cost = None
        self.state_dim = None
        self.action = None
        self.state = None
        self.node_state_queue = []
        self.container_state_queue = []
        self.action_dim = NodeNumber * ContainerNumber
        self.container_num = ContainerNumber
        self.node_num = NodeNumber
        self.service_num = ServiceNumber
        self.min_cost = 1e18
        self.max_reward = -1e18
        self.prepare()

    def prepare(self):
        self.container_state_queue = container_state_raw[:]
        self.node_state_queue = node_state_raw[:]
        self.state = self.container_state_queue + self.node_state_queue
        self.action = [-1, -1]
        self.state_dim = len(self.state)

    def reward(self, episode_t, episode_cost):
        better = False
        if episode_t == 1:
            reward = hp.lambda1  # 0
        else:
            if episode_cost < 0:
                reward = hp.lambda2
            else:
                if math.fabs(self.min_cost - episode_cost) < hp.precision:
                    reward = self.max_reward
                    better = True
                elif self.min_cost > episode_cost:
                    reward = self.max_reward + hp.lambda3
                    better = True
                else:
                    reward = hp.lambda4 * (self.min_cost - episode_cost)

        self.max_reward = max(reward, self.max_reward)
        if episode_cost > 0:
            self.min_cost = min(self.min_cost, episode_cost)
        # print("min_cost = {}".format(self.min_cost))

        return reward, better

    def reset(self):
        self.prepare()
        return self.state

    def update_state(self, action_pair):
        # update container state
        # [nodeId(+0), cpu(+1), memory(+2)(for container1), nodeId, cpu, memory(for container2), ...]
        node_id = action_pair[0]
        container_id = action_pair[1]
        self.container_state_queue[container_id * 3] = node_id
        # memory and cpu remains unchangeable

        # update node state
        # [container_1_bool(+0), container_2_bool, container_m_bool, cpu(+containerNum), memory(for node_1)(+containerNum + 1),
        #  container_1_bool, container_2_bool, container_m_bool, cpu, memory(for node_2),
        #  container_1_bool, container_2_bool, container_m_bool, cpu, memory(for node_n)]
        self.node_state_queue[node_id * (self.container_num + 2) + container_id] = 1
        # add the resource of container up to node
        # cpu
        self.node_state_queue[node_id * (self.container_num + 2) + self.container_num] += self.container_state_queue[
            container_id * 3 + 1]
        # memory
        self.node_state_queue[node_id * (self.container_num + 2) + self.container_num + 1] += self.container_state_queue[
            container_id * 3 + 2]
        # update state
        self.state = self.container_state_queue + self.node_state_queue

    def serviceComCost(self, service_i, service_j):
        cost = 0
        # each two services share an interaction weight
        interaction_weight_in_service = data.service_weight[service_i][service_j]
        container_list_in_service_i = data.service_container[service_i]
        container_list_in_service_j = data.service_container[service_j]
        for container_k in container_list_in_service_i:
            for container_l in container_list_in_service_j:
                cost += interaction_weight_in_service * self.getDisBetweenContainers(container_k, container_l)

        return cost

    def usageVar(self):
        var = 0
        node_cpu_list = []
        node_memory_list = []
        for node_id in range(NodeNumber):
            node_cpu = self.node_state_queue[node_id * (self.container_num + 2) + ContainerNumber]
            node_memory = self.node_state_queue[node_id * (self.container_num + 2) + ContainerNumber + 1]
            node_cpu_list.append(node_cpu)
            node_memory_list.append(node_memory)
            if node_cpu_list[node_id] > 1 or node_memory_list[node_id] > 1:
                var = -10
        var += 0.5 * np.var(node_cpu_list) + 0.5 * np.var(node_memory_list)
        return var

    def comCost(self):
        cost = 0
        for service_i in range(data.ServiceNumber):
            for service_j in range(data.ServiceNumber):
                cost += self.serviceComCost(service_i, service_j)

        return 0.5 * cost

    def cost(self):
        com_cost = self.comCost()
        com_cost /= 1e4
        usage_var = self.usageVar()
        print("com_cost = {}, usage_var = {}".format(com_cost, usage_var))
        res = 0 * com_cost + 1 * usage_var
        return res, com_cost, usage_var

    def step(self, action_index):
        # input: action(Targetnode，ContainerIndex)
        # output: next state, cost, done
        action_pair = self.index_to_pair(action_index)
        self.update_state(action_pair)
        # unnecessary
        # cost, com_cost, var_usage = self.comCost()

        deployed_container_num = 0
        for containerId in range(ContainerNumber):
            if self.container_state_queue[containerId * 3] != -1:
                deployed_container_num += 1

        done = deployed_container_num == ContainerNumber
        return self.state, done

    def index_to_pair(self, action_num):
        action_pair = [-1, -1]
        action_pair[0] = int(action_num / self.container_num)
        action_pair[1] = action_num % self.container_num
        return action_pair

    def getDisBetweenContainers(self, container_k, container_l):
        # find the node of containers
        # container to node is in self.container_state_queue [nodeID, cpu, memory, nodeID, cpu, memory, ...]
        node_container_k = self.container_state_queue[container_k * 3]
        node_container_l = self.container_state_queue[container_l * 3]
        # find the service of containers
        service_container_k = service_container_relationship[container_k]
        service_container_l = service_container_relationship[container_l]

        # if nodes of two containers aren't the same and services of two containers aren't the same,
        # the distance between the two containers is self.Dist[node_container_k][node_container_l]
        if node_container_k != node_container_l and service_container_k != service_container_l:
            container_distance = data.Dist[node_container_k][node_container_l]
        else:
            container_distance = 0

        return container_distance
