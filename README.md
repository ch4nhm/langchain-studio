# LangChain Demo Project

## 项目简介
本项目是一个基于 LangChain 的企业级智能助手平台。支持：
1. **智能问答 (Retrieval QA)**：基于 Chroma/PGVector 和 LangChain RAG 链路。
2. **结构化 SQL Agent**：支持通过自然语言进行数据库查询与结果提取。
3. **多轮记忆管理**：基于 PostgreSQL 持久化和滑动窗口 token 截断的会话管理。

## 系统要求
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Poetry (Python 包管理)

## 快速启动

1. **环境变量配置**：
```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env 填入你的 OPENAI_API_KEY
```

2. **本地环境初始化**：
```bash
make init
```

3. **数据导入 (ETL)**：
```bash
# 生成模拟数据并注入到 Chroma 向量库
make ingest
```

4. **一键启动 (Docker Compose)**：
```bash
# 启动 api, frontend, postgres, redis, chroma 五件套
docker-compose up -d
```

如果你希望在本地开发而不使用 Docker：
```bash
make start-backend  # 在一个终端运行
make start-frontend # 在另一个终端运行
```

## API 示例

### 同步问答
```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"query": "如何配置环境变量？", "session_id": "test-session"}'
```

### 流式对话 (SSE)
```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"query": "解释架构设计", "session_id": "test-session"}'
```

## 测试命令
```bash
# 运行单元测试
make test

# 运行代码规范检查
make lint
```

## 架构图
请参考 `backend/docs/architecture.md` 中的 C4 容器图与序列图。

## 开发者手册
详细指南请参考 `backend/docs/developer_manual.md`。
