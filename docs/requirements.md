# 需求文档

## 1. 项目背景与目标

本项目是一个 LangChain 全生命周期 Demo 工程，旨在覆盖从文档导入、向量检索、多轮对话记忆管理到 SQL 自然语言查询的完整 AI 应用开发链路，帮助开发者快速理解和落地 LangChain 的核心概念与最佳实践。

**核心目标：**
- 提供一个可运行的、生产级结构的 LangChain 参考实现
- 覆盖 RAG、Agent、Memory、Streaming 四大核心场景
- 前后端分离，具备完整的 API 接口文档和测试
- 支持 Docker 一键部署

---

## 2. 功能需求

### 2.1 RAG 同步问答

**功能描述：** 用户提交自然语言问题，系统从私有知识库中检索相关文档片段，结合 LLM 生成有依据的回答，并返回引用来源。

**业务规则：**
- 支持多轮对话上下文（携带 `session_id`）
- 历史消息最多保留最近 10 条（滑动窗口）
- 回答引用的源文档需同步返回给前端
- 多轮对话中的代词/指代需自动重写为独立问题再检索

**验收标准：**
- `POST /ask` 接口在 5 秒内返回完整答案
- 返回的 `source_documents` 与回答内容相关
- 同一 `session_id` 的多轮对话保持上下文连贯

**示例场景：**
```
第一轮：
  用户: "什么是 ChromaDB？"
  AI: "ChromaDB 是一个开源的向量数据库，专为 AI 应用设计..."

第二轮：
  用户: "它支持哪些编程语言的 SDK？"  ← 「它」需被重写为「ChromaDB」
  AI: "ChromaDB 官方提供 Python 和 JavaScript/TypeScript SDK..."
```

---

### 2.2 流式对话

**功能描述：** 与 RAG 问答功能相同，但以 SSE 流式方式逐字推送答案，实现打字机效果，改善长文本场景的用户体验。

**业务规则：**
- 使用 Server-Sent Events (SSE) 协议推送
- 每个 chunk 包含 `{ chunk, session_id }` 字段
- 流结束时发送 `data: [DONE]`
- 流式结束后再将完整答案持久化到数据库
- 发生错误时推送 `{ error: "错误信息" }` 后终止流

**验收标准：**
- 首个字符响应时间 < 2 秒
- 流式输出与同步接口的最终答案内容一致
- 网络中断后前端重连可继续接收（SSE 自动重连机制）

---

### 2.3 SQL 自然语言查询

**功能描述：** 用户用自然语言描述数据查询需求，系统自动生成并执行 SQL，返回结果的自然语言解释。

**业务规则：**
- Agent 最多迭代 3 次，防止无限循环
- 只允许 `SELECT` 查询，禁止 DDL/DML（需在 Agent 提示词中约束）
- 解析错误时自动重试
- 返回自然语言解释，而非裸 SQL 结果

**验收标准：**
- 简单聚合查询（COUNT、SUM、AVG）可正确执行
- 多表关联查询可正确识别表关系
- 查询失败时返回友好错误信息，不暴露数据库内部错误

**示例场景：**
```
用户: "2024 年每个季度的销售额分别是多少？"
Agent 内部执行:
  1. 查看所有表：orders, users, products
  2. 查看 orders 表结构：id, user_id, amount, created_at
  3. 执行 SQL:
     SELECT
       EXTRACT(QUARTER FROM created_at) as quarter,
       SUM(amount) as total_sales
     FROM orders
     WHERE EXTRACT(YEAR FROM created_at) = 2024
     GROUP BY quarter
     ORDER BY quarter;
  4. 返回: "2024年Q1销售额为 ¥120万，Q2为 ¥98万，Q3为 ¥145万，Q4为 ¥167万。"
```

---

### 2.4 会话记忆持久化

**功能描述：** 每次对话的问答记录持久化存储到 PostgreSQL，服务重启后历史不丢失。

**业务规则：**
- 以 `session_id` 为维度隔离不同用户/会话
- 消息以 JSONB 格式存储于 `message_store` 表
- 超过 10 条时自动应用滑动窗口截断（内存中截断，数据库保留全量）
- `session_id` 未传入时自动生成 UUID

**验收标准：**
- 服务重启后同一 `session_id` 的历史消息可正常加载
- 超长对话（>10轮）不导致 Token 超限报错

---

### 2.5 API 限流

**功能描述：** 基于 Redis 对每个客户端 IP 进行请求频率限制，防止滥用。

**业务规则：**
- 超出限流阈值时返回 HTTP 429 Too Many Requests
- 限流计数器存储于 Redis，支持分布式部署
- 健康检查接口 `/health` 不受限流影响

**验收标准：**
- 限流触发时响应体包含友好提示信息
- 限流窗口重置后请求可正常处理

---

### 2.6 请求追踪

**功能描述：** 每个 HTTP 请求自动生成唯一 ID，写入响应头，便于日志关联和问题排查。

**业务规则：**
- 请求 ID 格式为 UUID v4
- 通过响应头 `X-Request-ID` 返回给客户端
- 请求 ID 在整个请求生命周期内可通过 `request.state.request_id` 访问

---

## 3. 非功能需求

### 3.1 性能

| 接口 | 目标响应时间 | 说明 |
|------|-------------|------|
| `GET /health` | < 50ms | 无业务逻辑 |
| `POST /ask` | < 8s P95 | 含向量检索 + LLM 推理 |
| `POST /chat` 首字节 | < 2s | SSE 首个 chunk |
| `POST /sql` | < 15s P95 | Agent 多轮迭代 |

### 3.2 可靠性

- 自定义 `LangChainException` 统一捕获业务异常，返回结构化错误响应
- SQL Agent 设置 `max_iterations=3` 和 `handle_parsing_errors=True` 防止失控
- 向量库检索失败时抛出明确异常，不静默返回空结果

### 3.3 可维护性

- 所有公共函数具备完整类型注解（Mypy strict 模式）
- 代码风格遵循 Black + isort 规范
- pre-commit hooks 保证提交前代码质量
- CI Pipeline 自动运行测试和 lint

### 3.4 可扩展性

- LLM 切换：修改 `llm_factory.py` 一处即可切换底层模型（通义千问 / GPT / Claude）
- 向量库切换：修改 `vector_store.py` 可替换为 Pinecone、Weaviate 等
- 记忆后端切换：修改 `session_manager.py` 可替换为 Redis、MongoDB 等
- 新增 Agent：在 `agent/` 目录添加新文件，在 `router.py` 注册新路由

---

## 4. 技术选型说明

| 技术 | 选型 | 理由 |
|------|------|------|
| LLM | 通义千问 (qwen-turbo) | 国内访问稳定，成本低，支持中文 |
| Embedding | text-embedding-v1 | 与 LLM 同一生态，中文效果好 |
| 向量库 | ChromaDB | 开源、本地部署、零依赖、易上手 |
| 关系型DB | PostgreSQL + pgvector | 成熟稳定，支持向量扩展，适合生产 |
| 缓存/限流 | Redis | 高性能、原子操作适合计数器 |
| Web框架 | FastAPI | 原生异步、类型安全、自动文档 |
| 依赖管理 | Poetry | 锁文件精确、虚拟环境隔离好 |
| 前端 | React + Vite + TailwindCSS | 现代化工具链，热更新快 |

---

## 5. 约束与假设

- **模型访问**：需要有效的阿里云 DashScope API Key，并已开通通义千问和 Embedding 服务
- **网络**：后端服务需能访问 `dashscope.aliyuncs.com`
- **数据隐私**：用户的查询内容会发送至阿里云 API，不适用于包含敏感数据的场景
- **并发**：当前为单实例部署，高并发场景需配合负载均衡和多实例部署
- **SQL 安全**：SQL Agent 当前未做严格的只读限制，生产环境应使用只读数据库用户连接
