# 系统架构文档

## 1. 整体架构概览

本项目是一个基于 LangChain 的全栈 AI 应用，采用前后端分离架构，后端以 FastAPI 提供 RESTful/SSE 接口，前端使用 React + Vite + TailwindCSS 构建。

```
┌─────────────────────────────────────────────────────────────────┐
│                          客户端 (Browser)                        │
│                      React + Vite + TailwindCSS                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼──────────────────────────────────┐
│                        FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  /ask (同步) │  │ /chat (流式) │  │   /sql (SQL Agent)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│  ┌──────▼─────────────────▼──────┐   ┌──────────▼───────────┐  │
│  │       RAG Chain (LCEL)        │   │      SQL Agent        │  │
│  │  问题重写 → 向量检索 → LLM生成  │   │  NL → SQL → Execute  │  │
│  └──────┬────────────────────────┘   └──────────┬───────────┘  │
│         │                                        │              │
└─────────┼────────────────────────────────────────┼─────────────┘
          │                                        │
┌─────────▼──────────┐              ┌──────────────▼─────────────┐
│   ChromaDB (向量库) │              │    PostgreSQL (关系型DB)    │
│  - 技术文档向量存储  │              │  - 聊天历史 (message_store) │
│  - 相似度检索       │              │  - 结构化业务数据           │
└────────────────────┘              └────────────────────────────┘
          │                                        │
┌─────────▼────────────────────────────────────────────────────┐
│               阿里通义千问 (DashScope API)                     │
│     LLM: qwen-turbo     Embedding: text-embedding-v1          │
└──────────────────────────────────────────────────────────────┘
          │
┌─────────▼──────────┐
│   Redis            │
│  - API 限流计数器   │
└────────────────────┘
```

---

## 2. 后端模块结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 应用入口，中间件与路由注册
│   ├── api/
│   │   ├── router.py        # API 路由定义 (/health /ask /chat /sql)
│   │   └── middleware.py    # 限流中间件 (基于 Redis)
│   ├── core/
│   │   ├── config.py        # 全局配置 (Pydantic Settings，读取 .env)
│   │   ├── llm_factory.py   # LLM 实例工厂 (集中管理模型初始化)
│   │   └── exceptions.py    # 自定义异常类与全局异常处理器
│   ├── chain/
│   │   └── retrieval_qa.py  # RAG 链 (LCEL 构建：问题重写 + 检索 + 生成)
│   ├── agent/
│   │   └── sql_agent.py     # SQL Agent (自然语言 → SQL 执行器)
│   ├── retrieval/
│   │   └── vector_store.py  # ChromaDB 向量库封装 + DashScope Embedding
│   ├── memory/
│   │   └── session_manager.py # 会话历史管理 (PostgreSQL持久化 + 滑动窗口)
│   └── service/
│       └── chat_service.py  # 业务逻辑层 (串联 chain/agent/memory)
├── scripts/
│   ├── ingest.py            # 文档向量化导入脚本
│   └── generate_dummy_data.py # 测试数据生成脚本
├── tests/
│   └── test_api.py          # API 集成测试
├── pyproject.toml           # Poetry 依赖配置
├── poetry.toml              # Poetry 虚拟环境配置
├── Dockerfile               # 容器化构建文件
└── locustfile.py            # 性能压测脚本
```

---

## 3. 核心数据流

### 3.1 RAG 同步问答流 (`POST /ask`)

```
用户请求
  │  { query: "LangChain 是什么?", session_id: "abc" }
  ▼
router.py :: ask_sync()
  │
  ▼
chat_service.py :: process_qa_request()
  │  1. 从 PostgreSQL 加载历史消息
  │  2. 应用滑动窗口截断 (保留最近 10 条)
  │
  ▼
retrieval_qa.py :: get_retrieval_qa_chain().invoke()
  │
  ├─► [Step 1] 独立问题重写 (Condense Question)
  │     Prompt(历史+原始问题) → qwen-turbo → 独立问题
  │
  ├─► [Step 2] 向量检索
  │     独立问题 → DashScope Embedding → ChromaDB.similarity_search(k=5)
  │     返回最相关的 5 个文档片段
  │
  └─► [Step 3] 答案生成
        格式化文档 + 独立问题 → qa_prompt → qwen-turbo → 最终答案
  │
  ▼
chat_service.py
  │  3. 将 HumanMessage + AIMessage 写入 PostgreSQL
  │
  ▼
响应: { answer, source_documents, session_id }
```

### 3.2 流式对话流 (`POST /chat`)

```
用户请求
  │  { query: "...", session_id: "abc" }
  ▼
router.py :: chat_stream()
  │  返回 StreamingResponse (media_type="text/event-stream")
  ▼
chat_service.py :: process_chat_stream()  [AsyncGenerator]
  │
  ├─► 与同步流程相同的 RAG Chain
  │   使用 chain.astream() 异步流式获取输出
  │
  ├─► 逐 chunk yield SSE 格式数据:
  │     data: {"chunk": "Lang", "session_id": "abc"}
  │     data: {"chunk": "Chain", "session_id": "abc"}
  │     ...
  │     data: [DONE]
  │
  └─► 流结束后将完整答案写入 PostgreSQL
```

### 3.3 SQL Agent 流 (`POST /sql`)

```
用户请求
  │  { query: "统计每个月的订单数量" }
  ▼
sql_agent.py :: get_sql_agent().invoke()
  │
  └─► ReAct 循环 (最多 3 次迭代):
        Thought: 需要查看哪些表？
        Action: sql_db_list_tables → 获取所有表名
        Thought: 需要了解 orders 表结构
        Action: sql_db_schema → 获取建表语句
        Thought: 可以生成 SQL 了
        Action: sql_db_query → 执行 SELECT ...
        Final Answer: 根据查询结果的自然语言解释
  │
  ▼
响应: { answer, sql_executed, raw_result, session_id }
```

---

## 4. 中间件架构

### 4.1 中间件执行顺序

请求进入时，中间件按照**注册的逆序**执行：

```
请求 →  RequestID中间件  →  RateLimitMiddleware  →  CORSMiddleware  →  路由处理器
响应 ←  RequestID中间件  ←  RateLimitMiddleware  ←  CORSMiddleware  ←  路由处理器
```

| 中间件 | 作用 | 实现位置 |
|--------|------|----------|
| CORS | 允许跨域请求 | FastAPI 内置 |
| RateLimit | 基于 Redis 的 IP 限流 | `app/api/middleware.py` |
| RequestID | 为每个请求生成唯一 UUID，写入响应头 `X-Request-ID` | `app/main.py` |

---

## 5. 存储架构

| 存储组件 | 用途 | 技术选型 |
|----------|------|----------|
| ChromaDB | 文档向量存储，支持相似度搜索 | `chromadb` (本地持久化) |
| PostgreSQL | 聊天历史持久化、业务数据存储 | `psycopg2` + `pgvector` |
| Redis | API 限流计数器 | `redis-py` |

### ChromaDB 集合设计

```
collection: tech_docs
  - id: 文档片段唯一 ID
  - embedding: float[] (text-embedding-v1 生成的向量)
  - document: 原始文档文本
  - metadata: { source, page, chunk_index, ... }
```

### PostgreSQL 表设计

```sql
-- LangChain 自动创建的消息历史表
CREATE TABLE message_store (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    message     JSONB NOT NULL  -- 存储序列化后的 BaseMessage 对象
);
```

---

## 6. 部署架构

### Docker Compose 服务编排

```
docker-compose.yml
  ├── backend    (FastAPI, port 8000)
  ├── frontend   (Vite, port 5173)
  ├── postgres   (PostgreSQL, port 5432)
  ├── redis      (Redis, port 6379)
  └── chroma     (ChromaDB, port 8001, 可选)
```

### CI/CD

通过 `.github/workflows/ci.yml` 自动执行：
- 代码风格检查 (black, isort, flake8)
- 类型检查 (mypy)
- 单元/集成测试 (pytest)
