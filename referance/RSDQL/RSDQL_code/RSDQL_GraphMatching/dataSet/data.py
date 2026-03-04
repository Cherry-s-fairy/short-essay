#-*- coding: utf-8 -*-

import xml.dom.minidom
from xml.etree import ElementTree as ET

CPUnum = 7
Mem = 8 * 1024

class Data:
    def __init__(self, xml_path='./dataSet/data.xml'):
        self.xml_path = xml_path
        self._load_data()
        
    def _load_data(self):
        dom1 = xml.dom.minidom.parse(self.xml_path)
        root = dom1.documentElement
        dom2 = ET.parse(self.xml_path)
        
        self.NodeNumber = int(root.getElementsByTagName('nodeNumber')[0].firstChild.data)
        self.ContainerNumber = int(root.getElementsByTagName('containerNumber')[0].firstChild.data)
        self.ServiceNumber = int(root.getElementsByTagName('serviceNumber')[0].firstChild.data)
        self.ResourceType = int(root.getElementsByTagName('resourceType')[0].firstChild.data)
        
        self.service_containernum = []
        self.service_container = []
        self.service_container_relationship = []
        self.container_state_queue = []
        
        for oneper in dom2.findall('./number/containerNumber'):
            for child in oneper:
                self.service_container_relationship.append(int(child.text))
                
        for oneper in dom2.findall('./number/serviceNumber'):
            for child in oneper:
                self.service_containernum.append(int(child.text))
                self.service_container.append([int(child[0].text)])
                self.container_state_queue.append(-1)
                self.container_state_queue.append(int(child[0][0].text) / CPUnum)
                self.container_state_queue.append(int(child[0][1].text) / Mem)
        
        Dist_temp = []
        for oneper in dom2.findall('./distance'):
            for child in oneper:
                Dist_temp.append(float(child.text))
        self.Dist = [Dist_temp[i:i + self.NodeNumber] for i in range(0, len(Dist_temp), self.NodeNumber)]
        
        weight_temp = []
        for oneper in dom2.findall('./weight'):
            for child in oneper:
                weight_temp.append(float(child.text))
        self.service_weight = [weight_temp[i:i + self.ServiceNumber] for i in range(0, len(weight_temp), self.ServiceNumber)]
        
        self.node_resources = self._extract_node_resources()
        self.node_distances = self.Dist
        
    def _extract_node_resources(self):
        resources = []
        for i in range(self.NodeNumber):
            resources.append({
                'cpu': 1.0,
                'memory': 1.0,
                'bandwidth': 1.0
            })
        return resources


class Node:
    def __init__(self, node_id, cpu, memory, bandwidth=1.0):
        self.id = node_id
        self.cpu = cpu
        self.memory = memory
        self.bandwidth = bandwidth
        self.services = []
        
    def __repr__(self):
        return f"Node(id={self.id}, cpu={self.cpu:.2f}, mem={self.memory:.2f})"


class Service:
    def __init__(self, service_id, cpu_demand, memory_demand):
        self.id = service_id
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.dependencies = []
        
    def add_dependency(self, other_service, weight=1.0):
        self.dependencies.append({
            'service': other_service,
            'weight': weight
        })
        
    def __repr__(self):
        return f"Service(id={self.id}, cpu={self.cpu_demand:.2f}, mem={self.memory_demand:.2f})"


class Edge:
    def __init__(self, src, dst, weight=1.0, latency=0.0):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.latency = latency
        
    def __repr__(self):
        return f"Edge({self.src}->{self.dst}, weight={self.weight:.2f})"
