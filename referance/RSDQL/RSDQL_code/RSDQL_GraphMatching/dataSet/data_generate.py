import random
import xml.etree.ElementTree as ET

def uav_random_memory():
    return random.choice([128, 256, 512, 1024])

def uav_random_cpu():
    return random.choice([2, 4, 8, 16])

def uav_random_bandwidth():
    return random.choice([80, 120, 160, 180])  # Mbps

def uav_random_latency():
    return random.choice([5, 7, 9, 10, 13, 15, 18])  # ms

def uav_random_loss():
    return round(random.choice([0, 1, 2, 3]) * 0.01, 2)

def task_random_memory():
    return random.choice([2, 4, 8, 16, 32, 64])

def task_random_cpu():
    return random.choice([0.5, 1, 1.5, 2, 2.5])

def task_random_priority():
    return random.choice([1, 2, 3, 4, 5])  # Mbps

def task_random_bandwidth():
    return random.choice([20, 40, 60, 80])  # Mbps

def task_random_latency():
    return random.choice([15, 18, 20, 23, 26, 28])  # ms

def task_random_loss():
    return round(random.choice([0, 1, 2, 3, 4, 5, 6]) * 0.01, 2)

def task_random_data(max_val=200):
    return random.randint(20, max_val)

def random_src_dst(max_id) :
    src = random.randint(1, max_id)
    dst = random.randint(1, max_id)
    while dst == src:
        dst = random.randint(1, max_id)
    return src, dst

def generate_uav_nodes(count):
    nodes = []
    for i in range(1, count + 1):
        node = ET.Element(f"n{i}")
        ET.SubElement(node, "Id").text = str(i)
        ET.SubElement(node, "CPU").text = str(uav_random_cpu())
        ET.SubElement(node, "Memory").text = str(uav_random_memory())
        nodes.append(node)
    return nodes

def generate_uav_edges(count, node_num):
    edges = []
    exist_edges = set()
    max_possible_edges = node_num * (node_num - 1)
    if count > max_possible_edges:
        raise ValueError(f"无法生成{count}条不重复边，最多仅能生成{max_possible_edges}条（节点数：{node_num}）")

    while len(edges) < count:  # 改用while循环，确保生成足够数量的不重复边
        src, dst = random_src_dst(node_num)
        pair = (src, dst)
        if pair in exist_edges:
            continue

        exist_edges.add(pair)
        edge = ET.Element(f"e{len(edges) + 1}")
        ET.SubElement(edge, "src").text = str(src)
        ET.SubElement(edge, "dst").text = str(dst)
        ET.SubElement(edge, "bandwidth").text = str(uav_random_bandwidth())
        ET.SubElement(edge, "latency").text = str(uav_random_latency())
        ET.SubElement(edge, "loss").text = str(uav_random_loss())
        edges.append(edge)
    return edges

def generate_task_nodes(count):
    nodes = []
    for i in range(1, count + 1):
        node = ET.Element(f"t{i}")
        ET.SubElement(node, "Id").text = str(i)
        ET.SubElement(node, "CPU").text = str(task_random_cpu())
        ET.SubElement(node, "Memory").text = str(task_random_memory())
        ET.SubElement(node, "Priority").text = str(task_random_priority())
        nodes.append(node)
    return nodes

def generate_task_edges(count, node_num):
    edges = []
    exist_edges = set()
    max_possible_edges = node_num * (node_num - 1)
    if count > max_possible_edges:
        raise ValueError(f"无法生成{count}条不重复边，最多仅能生成{max_possible_edges}条（节点数：{node_num}）")

    while len(edges) < count:  # 改用while循环，确保生成足够数量的不重复边
        src, dst = random_src_dst(node_num)
        pair = (src, dst)
        if pair in exist_edges:
            continue

        exist_edges.add(pair)
        edge = ET.Element(f"e{len(edges)+1}")
        ET.SubElement(edge, "src").text = str(src)
        ET.SubElement(edge, "dst").text = str(dst)
        ET.SubElement(edge, "bandwidth").text = str(task_random_bandwidth())
        ET.SubElement(edge, "latency").text = str(task_random_latency())
        ET.SubElement(edge, "loss").text = str(task_random_loss())
        ET.SubElement(edge, "data").text = str(task_random_data())
        edges.append(edge)
    return edges

def create_xml(uav_count=6, uav_edge_count=4, task_count=6, task_edge_count=4):
    data = ET.Element("data")

    uavnode = ET.SubElement(data, "uavnode")
    node_number = ET.SubElement(uavnode, "nodeNumber")
    node_number.text = str(uav_count)
    uavnode.append(node_number)
    for node in generate_uav_nodes(uav_count):
        node_number.append(node)
    node_edge = ET.SubElement(uavnode, "nodeEdge")
    node_edge.text = str(uav_edge_count)
    uavnode.append(node_edge)
    for edge in generate_uav_edges(uav_edge_count, uav_count):
        node_edge.append(edge)

    tasknode = ET.SubElement(data, "tasknode")
    node_number = ET.SubElement(tasknode, "nodeNumber")
    node_number.text = str(task_count)
    tasknode.append(node_number)
    for node in generate_task_nodes(task_count):
        node_number.append(node)
    node_edge = ET.SubElement(tasknode, "nodeEdge")
    node_edge.text = str(task_edge_count)
    tasknode.append(node_edge)
    for edge in generate_task_edges(task_edge_count, task_count):
        node_edge.append(edge)

    tree = ET.ElementTree(data)
    return tree

if __name__ == "__main__":
    random.seed()  # 初始化随机种子
    tree = create_xml(uav_count=5, uav_edge_count=2, task_count=50, task_edge_count=20)
    tree.write("data.xml", encoding="utf-8", xml_declaration=True)
    print("XML 文件已生成：data.xml")