# 开发者手册

## 1. 环境准备

### 1.1 前置依赖

| 工具 | 版本要求 | 说明 |
|------|----------|------|
| Python | ^3.11 | 后端运行时 |
| Poetry | ^1.8 | Python 依赖与虚拟环境管理 |
| Node.js | ^18 | 前端运行时 |
| Docker & Docker Compose | 最新稳定版 | 基础设施服务（PostgreSQL / Redis） |
| Git | 任意 | 版本控制 |

### 1.2 克隆与初始化

```bash
git clone <repo-url>
cd langchain_demo

# 安装 pre-commit hooks（代码提交前自动检查格式）
pre-commit install
```

---

## 2. 后端开发

### 2.1 安装依赖

```bash
cd backend

# 安装所有依赖（包含 dev 依赖）
poetry install

# 仅安装生产依赖
poetry install --only main

# 查看虚拟环境路径
poetry env info
```

> **知识点：Poetry 虚拟环境**
> `backend/poetry.toml` 中配置了 `in-project = true`，Poetry 会在 `backend/.venv/` 下创建虚拟环境。
> VSCode / Cursor 会自动检测并使用该 Python 解释器，无需手动切换。

### 2.2 配置环境变量

```bash
cd backend
cp .env.example .env
# 编辑 .env，填写真实配置
```

`.env` 配置项说明：

```dotenv
# 阿里云 DashScope API Key（通义千问 LLM + Embedding）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# ChromaDB 持久化路径（相对于 backend 目录）
CHROMA_PERSIST_DIRECTORY=./chroma_db

# PostgreSQL 连接信息
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=langchain

# Redis 连接 URL
REDIS_URL=redis://localhost:6379/0
```

> **知识点：Pydantic Settings 自动加载**
> `app/core/config.py` 中的 `Settings` 类继承自 `pydantic_settings.BaseSettings`。
> Pydantic 会按以下优先级加载配置：
> 1. 操作系统环境变量（最高优先级）
> 2. `.env` 文件
> 3. 类属性默认值（最低优先级）
> 这意味着在 Docker 中通过 `-e` 传入的环境变量会覆盖 `.env` 文件中的值。

### 2.3 启动基础设施

```bash
# 在项目根目录启动 PostgreSQL 和 Redis
docker-compose up -d postgres redis

# 确认服务已就绪
docker-compose ps
```

### 2.4 导入文档数据

在启动 API 服务之前，需要先将文档向量化写入 ChromaDB：

```bash
cd backend

# 将 docs/ 目录下的文档向量化并存入 ChromaDB
poetry run python scripts/ingest.py

# 生成 PostgreSQL 测试数据（可选）
poetry run python scripts/generate_dummy_data.py
```

> **知识点：文档向量化 (Embedding) 流程**
> 1. `ingest.py` 使用 `DirectoryLoader` 加载文档文件
> 2. 通过 `RecursiveCharacterTextSplitter` 按字符分割为小块（chunk）
> 3. 每个 chunk 经过 `DashScopeEmbeddings`（text-embedding-v1）转换为高维向量
> 4. 向量连同原始文本和元数据一起持久化存储到 ChromaDB 的 `tech_docs` 集合中

### 2.5 启动开发服务器

```bash
cd backend

# 方式一：通过 Poetry 启动（推荐）
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 方式二：直接运行 main.py
poetry run python app/main.py
```

启动成功后访问：
- API 服务：`http://localhost:8000`
- 交互式文档 (Swagger UI)：`http://localhost:8000/docs`
- ReDoc 文档：`http://localhost:8000/redoc`

---

## 3. API 接口详解

### 3.1 健康检查 `GET /health`

```bash
curl http://localhost:8000/health
```

响应示例：
```json
{ "status": "ok" }
```

### 3.2 同步问答 `POST /ask`

适用于需要一次性获取完整回答的场景。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LangChain 的 LCEL 有什么优势？",
    "session_id": "user-session-001"
  }'
```

响应示例：
```json
{
  "answer": "LCEL（LangChain Expression Language）的主要优势包括：1. 声明式语法，链路构建直观简洁；2. 原生支持流式输出（astream）；3. 自动支持异步和并行执行；4. 内置调试和追踪支持。",
  "source_documents": [
    {
      "content": "LCEL 是 LangChain 在 v0.1 引入的新式链构建方式...",
      "metadata": { "source": "docs/lcel_guide.md", "page": 1 }
    }
  ],
  "session_id": "user-session-001"
}
```

> **知识点：RAG（检索增强生成）**
> 纯 LLM 的知识截止于训练日期，无法了解私有/最新文档。RAG 的解决思路：
> 1. **离线阶段**：将私有文档切片、向量化、存入向量库
> 2. **在线阶段**：用户提问时先去向量库检索最相关的文档片段
> 3. 将检索到的片段作为上下文拼入 Prompt，再交给 LLM 生成回答
> 这样 LLM 的回答有了「参考资料」，准确性大幅提升，且可引用来源。

### 3.3 流式对话 `POST /chat`

适用于前端需要实时显示打字效果的场景（SSE）。

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "帮我解释一下向量数据库的原理",
    "session_id": "user-session-001"
  }'
```

响应为 SSE 流（逐行输出）：
```
data: {"chunk": "向量", "session_id": "user-session-001"}

data: {"chunk": "数据库", "session_id": "user-session-001"}

data: {"chunk": "通过将文本转换为高维向量...", "session_id": "user-session-001"}

data: [DONE]
```

前端 JavaScript 接入示例：
```javascript
const response = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: '你好', session_id: 'abc' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  const lines = text.split('\n').filter(l => l.startsWith('data: '));
  for (const line of lines) {
    const data = line.replace('data: ', '');
    if (data === '[DONE]') return;
    const { chunk } = JSON.parse(data);
    console.log(chunk); // 实时追加到界面
  }
}
```

> **知识点：Server-Sent Events (SSE)**
> SSE 是一种基于 HTTP 的单向实时推送协议，相比 WebSocket 更简单：
> - 服务端：设置 `Content-Type: text/event-stream`，持续 `yield` 数据
> - 客户端：使用 `EventSource` API 或 `fetch` + ReadableStream 接收
> - 每条消息格式为 `data: <内容>\n\n`
> - 天然支持断线重连（浏览器 EventSource 会自动重连）

### 3.4 SQL 自然语言查询 `POST /sql`

允许用自然语言查询数据库，无需编写 SQL。

```bash
curl -X POST http://localhost:8000/sql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "统计各个月份的订单数量，按时间升序排列",
    "session_id": "user-session-002"
  }'
```

响应示例：
```json
{
  "answer": "根据查询结果，1月有 120 笔订单，2月有 98 笔订单，3月有 145 笔订单。整体呈上升趋势。",
  "sql_executed": "N/A",
  "raw_result": "N/A",
  "session_id": "user-session-002"
}
```

> **知识点：ReAct Agent 工作原理**
> SQL Agent 基于 ReAct（Reasoning + Acting）范式工作，本质是一个循环：
> ```
> Thought（思考）→ Action（行动）→ Observation（观察）→ Thought → ...
> ```
> 具体步骤：
> 1. **Thought**：LLM 思考「要回答这个问题，我需要知道哪些表」
> 2. **Action**：调用 `sql_db_list_tables` 工具获取所有表名
> 3. **Observation**：得到表名列表 `[orders, users, products]`
> 4. **Thought**：「需要查看 orders 表的结构」
> 5. **Action**：调用 `sql_db_schema` 工具
> 6. **Observation**：得到建表 DDL
> 7. **Thought**：「现在可以写 SQL 了」
> 8. **Action**：调用 `sql_db_query` 执行 SQL
> 9. **Final Answer**：将结果翻译为自然语言

---

## 4. 核心知识点详解

### 4.1 LCEL（LangChain Expression Language）

LCEL 使用管道符 `|` 将多个 `Runnable` 组件串联：

```python
# 基础示例：Prompt | LLM | 输出解析器
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话解释 {concept}")
chain = prompt | llm | StrOutputParser()

# 同步调用
result = chain.invoke({"concept": "向量数据库"})

# 流式调用
for chunk in chain.stream({"concept": "向量数据库"}):
    print(chunk, end="", flush=True)

# 异步调用
result = await chain.ainvoke({"concept": "向量数据库"})
```

本项目中的 RAG 链结构：
```python
full_chain = (
    RunnablePassthrough.assign(standalone_question=condense_question_chain)
    | RunnablePassthrough.assign(question=lambda x: x["standalone_question"])
    | RunnablePassthrough.assign(context=retriever)
    | RunnablePassthrough.assign(answer=qa_chain)
)
```

### 4.2 多轮对话记忆管理

本项目采用 **PostgreSQL 持久化 + 滑动窗口截断** 方案：

```python
# session_manager.py 核心逻辑
def apply_sliding_window(messages, k=10):
    """只保留最近 k 条消息，防止 Token 超限"""
    if messages and messages[0].type == "system":
        # 保留 SystemMessage + 最近 k-1 条
        return [messages[0]] + messages[-(k-1):]
    return messages[-k:]
```

对话历史在 PostgreSQL 的 `message_store` 表中以 JSONB 格式存储：
```json
{
  "type": "human",
  "data": { "content": "LangChain 是什么？", "type": "human" }
}
```

> **为什么需要独立问题重写（Condense Question）？**
> 
> 多轮对话中存在指代消解问题：
> - 用户第一轮：「介绍一下 LangChain」
> - 用户第二轮：「**它**支持哪些向量数据库？」
> 
> 如果直接用「它支持哪些向量数据库？」去检索，向量库不知道「它」是谁。
> 重写后变成：「**LangChain** 支持哪些向量数据库？」，检索准确率大幅提升。

### 4.3 向量相似度检索原理

```
文档片段: "ChromaDB 是一个开源的向量数据库..."
         ↓ text-embedding-v1
向量:     [0.23, -0.51, 0.87, 0.12, ...]  (1536 维)

用户查询: "有哪些向量数据库?"
         ↓ text-embedding-v1
查询向量: [0.21, -0.49, 0.91, 0.09, ...]  (1536 维)

余弦相似度 = dot(文档向量, 查询向量) / (|文档向量| * |查询向量|) = 0.94 ✓ 高度相关
```

ChromaDB 检索配置（`vector_store.py`）：
```python
retriever = get_vector_store().as_retriever(
    search_kwargs={"k": 5}  # 返回最相关的 5 个文档片段
)
```

### 4.4 LLM 工厂模式

`llm_factory.py` 集中管理模型实例化，好处是切换模型只需改一处：

```python
# 当前：通义千问
def get_llm(temperature=0.0) -> ChatTongyi:
    return ChatTongyi(model="qwen-turbo", temperature=temperature, ...)

# 若要切换为 GPT-4，只需改这里：
# def get_llm(temperature=0.0) -> ChatOpenAI:
#     return ChatOpenAI(model="gpt-4o", temperature=temperature, ...)
```

`temperature` 参数影响输出的随机性：

| temperature | 适用场景 | 输出特点 |
|-------------|----------|----------|
| 0.0 | SQL 生成、信息提取、RAG 回答 | 确定、保守、可复现 |
| 0.3~0.7 | 摘要、翻译 | 平衡准确与流畅 |
| 0.8~1.0 | 创意写作、头脑风暴 | 多样、创意、随机 |

---

## 5. 测试

### 5.1 运行测试

```bash
cd backend

# 运行所有测试
poetry run pytest

# 带覆盖率报告
poetry run pytest --cov=app --cov-report=html

# 只运行指定测试文件
poetry run pytest tests/test_api.py -v

# 运行带特定标记的测试
poetry run pytest -m "not slow" -v
```

### 5.2 压力测试

```bash
cd backend

# 启动 Locust Web UI（http://localhost:8089）
poetry run locust -f locustfile.py --host=http://localhost:8000

# 无 UI 模式（自动化压测）
poetry run locust -f locustfile.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 60s --headless
```

---

## 6. 代码规范

### 6.1 格式化与检查

```bash
cd backend

# 代码格式化（Black）
poetry run black app/ tests/

# import 排序（isort）
poetry run isort app/ tests/

# 风格检查（Flake8）
poetry run flake8 app/ tests/

# 类型检查（Mypy）
poetry run mypy app/
```

### 6.2 Pre-commit 自动检查

安装后，每次 `git commit` 时自动执行以上检查：

```bash
# 首次安装
pre-commit install

# 手动对全部文件执行一次
pre-commit run --all-files
```

### 6.3 代码风格约定

- 行长限制：**88 字符**（Black 默认）
- Python 版本目标：**3.11**
- import 风格：`isort` profile = `black`
- 类型注解：所有公共函数必须有类型注解（mypy strict 模式）
- 异常处理：业务异常统一使用 `LangChainException`，勿直接抛出裸异常

---

## 7. Docker 部署

### 7.1 完整服务启动

```bash
# 在项目根目录
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看后端日志
docker-compose logs -f backend
```

### 7.2 单独构建后端镜像

```bash
cd backend
docker build -t langchain-demo-backend:latest .

# 本地运行镜像
docker run -p 8000:8000 --env-file .env langchain-demo-backend:latest
```

### 7.3 环境变量注入（生产）

生产环境不应将 `.env` 文件打包进镜像，应通过以下方式注入：

```yaml
# docker-compose.yml 示例
services:
  backend:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}   # 从宿主机环境变量读取
      - POSTGRES_SERVER=postgres
      - REDIS_URL=redis://redis:6379/0
```

---

## 8. 常见问题排查

### Q1: `poetry install` 报 Python 版本不匹配

```bash
# 查看当前 Python 版本
python --version

# 指定 Poetry 使用 Python 3.11
poetry env use python3.11
poetry install
```

### Q2: ChromaDB 报 `Collection tech_docs does not exist`

原因：未执行文档导入脚本。

```bash
cd backend
poetry run python scripts/ingest.py
```

### Q3: PostgreSQL 连接失败

```bash
# 确认 Docker 服务正在运行
docker-compose ps postgres

# 测试连接
docker-compose exec postgres psql -U postgres -d langchain -c "\dt"
```

### Q4: API 返回 429 Too Many Requests

触发了 Redis 限流中间件。等待限流窗口重置，或在开发环境中临时禁用 `RateLimitMiddleware`（`app/main.py`）。

### Q5: SSE 流式接口无输出

检查以下几点：
1. 请求头是否包含 `Accept: text/event-stream`
2. 代理服务器（Nginx 等）是否禁用了响应缓冲（需设置 `X-Accel-Buffering: no`）
3. 后端是否有异常（查看 `X-Request-ID` 响应头后在日志中搜索对应 ID）

         