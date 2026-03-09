#-*- coding: utf-8 -*-

import xml.dom.minidom
from math import inf
from xml.etree import ElementTree as ET

def _get_element_text(elem, *names, default=0.0):
    for name in names:
        child = elem.find(name)
        if child is not None and child.text:
            return float(child.text)
    return default

class Data:
    def __init__(self, xml_path='data.xml'):
        self.xml_path = xml_path
        self._load_data()
        
    def _load_data(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        self.uav_nodes = {}
        self.service_nodes = {}
        self.uav_edges = []
        self.service_edges = []

        uavnode = root.find('uavnode')
        for node_elem in uavnode.find('nodeNumber'):
            node_id = int(node_elem.find('Id').text)
            cpu = float(node_elem.find('CPU').text)
            memory = float(node_elem.find('Memory').text)
            self.uav_nodes[node_id] = UAV(node_id, cpu, memory)

        nodeEdge = uavnode.find('nodeEdge')
        if nodeEdge is not None:
            for edge_elem in nodeEdge:
                src = int(edge_elem.find('src').text)
                dst = int(edge_elem.find('dst').text)
                bandwidth = _get_element_text(edge_elem, 'bandwidth', default=0)
                latency = _get_element_text(edge_elem, 'latency', default=inf)
                loss = _get_element_text(edge_elem, 'loss', default=1)
                self.uav_nodes[src].add_src_edge(src, dst, bandwidth, latency, loss)
                self.uav_nodes[dst].add_dst_edge(src, dst, bandwidth, latency, loss)
                self.uav_edges.append(Edge(src, dst, bandwidth, latency, loss, 0))

        servicenode = root.find('servicenode')
        for node_elem in servicenode.find('nodeNumber'):
            node_id = int(node_elem.find('Id').text)
            cpu_demand = float(node_elem.find('CPU').text)
            memory_demand = float(node_elem.find('Memory').text)
            self.service_nodes[node_id] = Service(node_id, cpu_demand, memory_demand)

        nodeEdge = servicenode.find('nodeEdge')
        if nodeEdge is not None:
            for edge_elem in nodeEdge:
                src = int(edge_elem.find('src').text)
                dst = int(edge_elem.find('dst').text)
                bandwidth = _get_element_text(edge_elem, 'bandwidth', default=0)
                latency = _get_element_text(edge_elem, 'latency', default=inf)
                data = _get_element_text(edge_elem, 'data', default=0)
                self.service_nodes[src].add_src_edge(src, dst, bandwidth, latency, 0, data)
                self.service_nodes[dst].add_dependencies(src, dst, bandwidth, latency, 0, data)
                self.service_edges.append(Edge(src, dst, bandwidth, latency, 0, data))


class UAV:
    def __init__(self, node_id, cpu, memory):
        self.id = node_id
        self.total_cpu = cpu
        self.total_memory = memory
        self.remain_cpu = cpu
        self.remain_memory = memory
        self.src_edges = []
        self.dst_edges = []
        self.services = []

    def add_src_edge(self, src, dst, bandwidth, latency, loss):
        self.src_edges.append({
            'src': src,
            'dst': dst,
            'bandwidth': bandwidth,
            'latency': latency,
            'loss': loss
        })

    def add_dst_edge(self, src, dst, bandwidth, latency, loss):
        self.dst_edges.append({
            'src': src,
            'dst': dst,
            'bandwidth': bandwidth,
            'latency': latency,
            'loss': loss
        })

    def __repr__(self):
        return f"UAV(id={self.id}, total_cpu={self.total_cpu:.2f}, total_mem={self.total_memory:.2f}, remain_cpu={self.remain_cpu:.2f}, remain_mem={self.remain_memory:.2f}, src_edges={self.src_edges}, dst_edges={self.dst_edges}, service={self.services})"


class Service:
    def __init__(self, service_id, cpu_demand, memory_demand):
        self.id = service_id
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.src_edges = []
        self.dependencies = []
        self.node = 0

    def add_src_edge(self, src, dst, bandwidth, latency, loss, data):
        self.src_edges.append({
            'src': src,
            'dst': dst,
            'bandwidth': bandwidth,
            'latency': latency,
            'loss': loss,
            'data': data
        })

    def add_dependencies(self, src, dst, bandwidth, latency, loss, data):
        self.dependencies.append({
            'src': src,
            'dst': dst,
            'bandwidth': bandwidth,
            'latency': latency,
            'loss': loss,
            'data': data
        })

    def __repr__(self):
        return f"Service(id={self.id}, cpu={self.cpu_demand:.2f}, mem={self.memory_demand:.2f}, src_edges={self.src_edges}, dependencies={self.dependencies}, node={self.node})"


class Edge:
    def __init__(self, src, dst, bandwidth, latency, loss, data):
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth
        self.latency = latency
        self.loss = loss
        self.data = data

    def __repr__(self):
        return f"Edge({self.src}->{self.dst}, bandwidth={self.bandwidth:.2f}, latency={self.latency:.2f}, loss={self.loss:.2f}, data={self.data:.2f})"


def test_data_loading():
    data = Data('dataSet/data.xml')

    print("=== UAV Nodes ===")
    for node_id, node in data.uav_nodes.items():
        print(node)

    print("\n=== Service Nodes ===")
    for service_id, service in data.service_nodes.items():
        print(service)

    print("\n=== UAV Edges ===")
    for edge in data.uav_edges:
        print(f"  {edge.src} -> {edge.dst}: bw={edge.bandwidth:.2f}, lat={edge.latency:.2f}, loss={edge.loss:.2f}")

    print("\n=== Service Edges ===")
    for edge in data.service_edges:
        print(f"  {edge.src} -> {edge.dst}: bw={edge.bandwidth:.2f}, lat={edge.latency:.2f}, loss={edge.loss:.2f}, data={edge.data}")

if __name__ == "__main__":
    test_data_loading()