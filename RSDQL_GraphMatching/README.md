# RSDQL_GraphMatching

**用强化学习智能地将50个微服务部署到5个UAV节点上，优化通信延迟和资源利用率。**

基于图匹配和深度强化学习的微服务部署优化系统

## ⚡ 快速开始

### 环境诊断
```bash
cd RSDQL_GraphMatching
python diagnose.py
```

### 修复依赖 (如需要)
```bash
pip install "numpy<2" --force-reinstall
pip install torch scipy matplotlib
```

### 运行完整测试
```bash
python test_graph_matching.py
```

### 训练模型
```bash
python train.py
```

### 对比算法
```bash
python compare_agents.py
```

## 📁 项目结构

```
RSDQL_GraphMatching/
├── dataSet/
│   ├── data.py                # XML 解析器
│   ├── data.xml               # 基础设施和任务配置
│   └── data_generate.py        # 数据生成工具
│
├── 【核心模块】
│   ├── resource_graph.py       # 资源拓扑图 (5个UAV节点)
│   ├── task_graph.py           # 任务拓扑图 (50个微服务)
│   ├── graph_matcher.py        # 图匹配 (4种算法)
│   ├── feedback_optimizer.py   # 反馈优化器
│   └── env.py                  # RL 环境 (reset/step)
│
├── 【RL 模块】
│   ├── rainbow_agent.py        # Rainbow DQN 智能体
│   ├── train.py                # 训练脚本
│   └── compare_agents.py       # 算法对比
│
├── 【GAT 编码】(可选)
│   ├── resources_gat_encoder.py # 资源图 GAT 编码
│   └── tasks_gat_encoder.py     # 任务图 GAT 编码
│
├── 【测试 & 文档】✨ NEW
│   ├── test_graph_matching.py  # 综合测试套件 (32个测试)
│   ├── test_env.py             # 环境测试
│   ├── TEST_GUIDE.md           # 详细测试文档
│   ├── NUMPY_FIX.md            # NumPy兼容性修复
│   ├── diagnose.py             # 环境诊断工具
│   └── README.md               # 本文件
```

## 🎯 核心思想

### 1. **资源拓扑图** (ResourceTopologyGraph)
- 表示 5 个 UAV 节点的基础设施
- 节点属性: CPU 容量、内存容量、出度
- 网络属性: 带宽、延迟、丢包率
- 计算: Floyd-Warshall 最短路径

### 2. **任务拓扑图** (TaskTopologyGraph)
- 表示 50 个微服务的依赖关系
- 节点属性: CPU 需求、内存需求、优先级
- 边属性: 依赖关系、数据流、通信开销
- 编码: [可选] GAT 图注意力网络

### 3. **图匹配** (GraphMatcher)
4 种匹配算法:
- **Hungarian**: 线性分配，全局最优 O(n³)
- **Greedy**: 快速贪心，次优 O(n²)
- **Learned**: 优先级感知，支持多对一
- **Heuristic**: 启发式平衡

匹配质量评分: `0.7 × node_similarity + 0.3 × edge_similarity`

### 4. **反馈优化** (FeedbackOptimizer)
- 计算部署指标 (成功率、延迟、负载均衡)
- 迭代调整任务图 (分割/合并微服务)

### 5. **RL 环境** (Env)
```python
state, valid, msg = env.reset()  # 初始化环境

# 选择匹配算法 (action: 0=greedy, 1=hungarian, 2=learned, 3=heuristic, 4=hybrid)
next_state, reward, done, info = env.step(action=0)

# 返回信息包括: match_score, cost, metrics, feedback, deployment_plan
```

### 6. **RL 智能体** (RainbowDQNAgent)
6 大改进:
- ✓ Dueling DQN - 分离价值流和优势流
- ✓ Double DQN - 减少 Q 值高估
- ✓ PER - 优先级经验回放
- ✓ NoisyNet - 参数化噪声探索
- ✓ N-Step Returns - 多步临界信号
- ✓ Distributed - 分布式学习

## 🛠️ 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| **数据** | XML 解析 | 加载基础设施和任务配置 |
| **数值** | NumPy, SciPy | 矩阵运算、Hungarian 算法 |
| **图** | 邻接矩阵、距离矩阵 | 编码拓扑结构 |
| **图编码** | PyTorch Geometric + GAT | 图神经网络 (可选) |
| **强化学习** | PyTorch + DQN | Rainbow DQN 智能体 |
| **可视化** | Matplotlib | 实验结果图表 |

## 🧪 测试覆盖

完整测试套件 **32 个测试点**:

| Suite | 内容 | 测试数 | 预期耗时 |
|-------|------|--------|---------|
| **1** | 数据加载 | 4 | < 1s |
| **2** | 资源图构建 | 6 | < 1s |
| **3** | 任务图构建 | 4 | < 1s |
| **4** | 图匹配算法 | 4 | < 1s |
| **5** | 部署验证 | 4 | < 1s |
| **6** | RL 环境集成 | 5 | < 2s |
| **7** | 性能基准 | 5 | < 2s |
| **总计** | - | **32** | **< 10s** |

运行测试:
```bash
python test_graph_matching.py
```

详细测试文档: 查看 `TEST_GUIDE.md`

## 📊 数据格式

编辑 `dataSet/data.xml` 修改基础设施或工作负载:

```xml
<uav_node id="0">
  <cpu>100</cpu>
  <memory>128</memory>
</uav_node>

<service_node id="0">
  <cpu_demand>10</cpu_demand>
  <memory_demand>16</memory_demand>
  <priority>0</priority>
</service_node>
```

## 📈 关键参数 (在 train.py / compare_agents.py 中)

```python
LEARNING_RATE = 0.001
GAMMA = 0.9              # 折扣因子
MEMORY_SIZE = 20000      # 经验回放缓冲区
BATCH_SIZE = 32
MAX_EPISODE = 1000
PRIORITY_ALPHA = 0.6     # PER 指数
PRIORITY_BETA = 0.4      # PER IS 校正
NOISY_SIGMA = 0.5        # NoisyNet 噪声尺度
N_STEP = 3
```

## 📁 输出文件

- `rainbow_model.npz` - 保存的 Rainbow DQN 权重
- `experiment_results.json` - 实验指标
- `training_results.png` - 训练曲线
- `experiment_plots/` - 详细图表
- `reward.txt` - 每回合奖励日志

## ✅ 依赖

必需:
- `numpy`, `scipy` - 核心数值计算 (linear_sum_assignment 用于 Hungarian)
- `torch`, `torch_geometric` - GAT 编码 (可选，无则使用 NumPy 平均)
- `matplotlib` - 可视化

## 📝 核心概念速查

| 术语 | 含义 |
|------|------|
| **Task Graph** | 微服务拓扑 (谁调用谁) |
| **Resource Graph** | 基础设施拓扑 (UAV 和网络) |
| **Mapping** | 分配方案 (Task → UAV) |
| **Match Score** | 方案质量评分 [0, 1] |
| **Constraint** | 资源约束 (CPU/内存限制) |
| **Feedback** | 优化反馈 (分割/合并任务) |
| **Action** | RL 动作 (选择匹配算法) |
| **Reward** | 部署成本的反面 (越高越好) |
| **N-to-1 Mapping** | 多个任务映射到同一节点 |

## 🚀 后续步骤

1. ✅ **环境诊断** → `python diagnose.py`
2. ✅ **运行测试** → `python test_graph_matching.py`
3. ✅ **训练模型** → `python train.py`
4. ✅ **对比算法** → `python compare_agents.py`

## 📚 相关文档

- **TEST_GUIDE.md** - 详细的测试文档和故障排查
- **NUMPY_FIX.md** - NumPy 兼容性问题解决
- **CLAUDE.md** - 项目指南 (开发者说明)

## 💡 常见问题

**Q: 为什么有 4 种匹配算法？**
A: 不同场景权衡不同。小任务用 Hungarian (最优)，大任务用 Greedy (高速)。

**Q: Many-to-one 映射是什么？**
A: 多个微服务可分配到同一 UAV，只要不超容量限制。

**Q: RL 智能体学习什么？**
A: 在什么情况下选择哪种匹配算法，最小化部署总成本。

**Q: 如何修改基础设施配置？**
A: 编辑 `dataSet/data.xml`，修改 UAV 数量、容量、网络拓扑。

## Python 版本要求

- Python 3.8+

---

祝你使用愉快！🎉
