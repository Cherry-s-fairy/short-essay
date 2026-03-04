import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv


# 读取无人机数据文件
def read_drone_data(file_path):
    df = pd.read_csv(file_path)

    # 假设CSV文件包含无人机ID、位置和速度等数据
    features = df[['x', 'y', 'z', 'speed', 'load']].values  # 每个无人机的特征向量
    return features, df


# 构建图结构，连接距离较近的无人机
def build_graph(features, threshold=10.0):
    num_nodes = len(features)
    edge_index = []

    # 根据欧氏距离构建图边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(features[i, :3] - features[j, :3])  # 只考虑位置 (x, y, z) 作为距离计算
            if distance < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


# GNN模型定义
class DroneGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DroneGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


# 主函数
def main(file_path):
    # Step 1: 读取无人机数据
    features, df = read_drone_data(file_path)

    # Step 2: 将数据转换为PyTorch张量
    x = torch.tensor(features, dtype=torch.float)

    # Step 3: 构建图
    edge_index = build_graph(features, threshold=10.0)  # 10.0是邻接的距离阈值

    # Step 4: 创建GNN模型
    model = DroneGNN(in_channels=x.size(1), out_channels=32)  # 输入特征维度与无人机数据特征数量一致
    out = model(x, edge_index)

    print(out)


# 运行示例
if __name__ == "__main__":
    file_path = "drone_data.csv"  # 假设数据文件名为drone_data.csv
    main(file_path)