# LangChain & Agent 面试题实操练习

本目录对应面试常见问题，提供可直接运行的代码示例。

## 目录结构

```
practice/
├── agent/
│   ├── 01_agent_components.py      # Agent 核心组件构成
│   ├── 02_react_agent.py           # ReAct 框架实现与解析
│   ├── 03_planning_cot_tot.py      # 规划能力：CoT / ToT / GoT
│   ├── 04_memory_system.py         # 短期记忆 & 长期记忆设计
│   ├── 05_tool_use_function_call.py # Tool Use & Function Calling
│   ├── 06_multi_agent.py           # 多智能体协同系统
│   └── 07_agent_safety.py          # Agent 安全与对齐
├── rag/
│   ├── 01_rag_pipeline.py          # 完整 RAG 流水线
│   ├── 02_chunking_strategies.py   # 文本切块策略对比
│   ├── 03_advanced_retrieval.py    # 高级检索：HyDE/MMR/融合检索
│   ├── 04_lost_in_middle.py        # Lost in the Middle 问题与缓解
│   ├── 05_rag_evaluation.py        # RAG 系统评估指标
│   └── 06_adaptive_rag.py          # 自适应 RAG / 多轮检索
└── README.md
```

## 运行方式

```bash
cd backend
# 确保 .env 已配置 OPENAI_API_KEY（填写 DashScope Key）
poetry run python ../practice/agent/01_agent_components.py
poetry run python ../practice/rag/01_rag_pipeline.py
```

## 面试题索引

| # | 面试题 | 对应文件 |
|---|--------|----------|
| A1 | 如何定义基于 LLM 的智能体？核心组件？ | `agent/01_agent_components.py` |
| A2 | ReAct 框架详解 | `agent/02_react_agent.py` |
| A3 | CoT / ToT / GoT 规划能力 | `agent/03_planning_cot_tot.py` |
| A4 | 短期记忆与长期记忆设计 | `agent/04_memory_system.py` |
| A5 | Tool Use & Function Calling | `agent/05_tool_use_function_call.py` |
| A6 | 多智能体系统 | `agent/06_multi_agent.py` |
| A7 | Agent 安全与对齐 | `agent/07_agent_safety.py` |
| R1 | RAG 工作原理与完整流水线 | `rag/01_rag_pipeline.py` |
| R2 | 文本切块策略 | `rag/02_chunking_strategies.py` |
| R3 | 高级检索技术 HyDE/MMR/融合 | `rag/03_advanced_retrieval.py` |
| R4 | Lost in the Middle 问题 | `rag/04_lost_in_middle.py` |
| R5 | RAG 系统评估 | `rag/05_rag_evaluation.py` |
| R6 | 自适应 RAG / 多轮检索 | `rag/06_adaptive_rag.py` |
