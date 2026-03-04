# RSDQL_GraphMatching

基于图匹配的微服务部署优化系统

## 项目结构

```
RSDQL_GraphMatching/
├── dataSet/           # 数据模块
│   ├── data.py        # 数据加载
│   └── data.xml       # 原始数据
├── resource_graph.py    # 资源拓扑图生成器
├── task_graph.py         # 任务拓扑图生成器
├── graph_matcher.py      # 图匹配算法
├── feedback_optimizer.py # 反馈优化器
├── env.py                # 环境模块
├── agent.py              # 智能体
├── algorithm.py          # 强化学习算法
├── model.py              # 神经网络模型
├── replay_memory.py      # 经验回放
└── train.py              # 训练主流程
```

## 核心思想

1. **资源拓扑图**: 表示底层基础设施的资源和连接
2. **任务拓扑图**: 表示微服务之间的依赖关系
3. **图匹配**: 将任务图映射到资源图
4. **反馈优化**: 将部署效果反馈给任务图生成器

## 技术栈

- Python 3.7+
- PaddlePaddle (深度学习框架)
- NetworkX (图操作)
- NumPy
