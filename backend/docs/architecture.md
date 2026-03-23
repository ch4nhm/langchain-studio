# 技术选型与架构

## 1. 技术选型
- **编程语言**：Python 3.11 + Poetry (依赖锁定)
- **框架**：LangChain >= 0.1.0, FastAPI
- **向量数据库**：Chroma (本地) / PGVector (生产)
- **关系数据库**：PostgreSQL 14+
- **模型**：OpenAI GPT-3.5-turbo (文本生成), text-embedding-ada-002 (嵌入)
- **前端**：React + Vite + TailwindCSS
- **中间件**：Redis (速率限制)
- **测试与部署**：Pytest, Locust, Docker, GitHub Actions, Prometheus, Grafana

## 2. 架构分层设计
```text
[Web Frontend] (React SPA, Vite)
       |
       v
[API Layer] (FastAPI: /ask, /chat, /health)
       |
       v
[Service Layer] (业务编排，异常处理，请求路由)
       |
       +--> [Agent Layer] (Structured SQL Agent, 工具调用)
       |
       +--> [Chain Layer] (Retrieval QA Chain, 流程组合)
       |
       v
[Memory & Retrieval] (PostgresChatMessageHistory, Chroma/PGVector)
```

## 3. C4 模型 (Context & Container)

### 3.1 容器图
```mermaid
C4Container
    Person(user, "用户", "通过浏览器访问 Web 界面")
    Container_Boundary(sys, "LangChain 智能问答系统") {
        Container(web, "Web Frontend", "React, Vite", "提供聊天 UI、Markdown 渲染、溯源展示")
        Container(api, "API Server", "FastAPI, Python", "提供 REST 和 SSE 流式接口，异常处理与限流")
        Container(agent, "Core Agent & Chains", "LangChain", "执行 Retrieval QA 和 SQL 交互，组装 prompt，记忆管理")
        ContainerDb(vectordb, "Vector Database", "Chroma / PGVector", "存储文档块向量，提供 KNN 检索")
        ContainerDb(rdbms, "Relational Database", "PostgreSQL", "存储聊天记录，作为 SQL Agent 的目标数据库")
        ContainerDb(redis, "Cache & Rate Limit", "Redis", "基于令牌桶的速率限制")
    }
    System_Ext(llm, "LLM Provider", "OpenAI GPT-3.5-turbo")

    Rel(user, web, "访问", "HTTPS")
    Rel(web, api, "调用接口", "JSON/REST/SSE")
    Rel(api, redis, "检查限流", "Redis Protocol")
    Rel(api, agent, "发起会话任务")
    Rel(agent, rdbms, "存取记忆 / 执行 SQL")
    Rel(agent, vectordb, "文档检索", "gRPC/HTTP")
    Rel(agent, llm, "发送 Prompt", "HTTPS")
```

### 3.2 序列图 (Retrieval QA)
```mermaid
sequenceDiagram
    participant User as User (Web)
    participant API as FastAPI Server
    participant Chain as Retrieval QA Chain
    participant Mem as PostgresChatMessageHistory
    participant VDB as Chroma VectorDB
    participant LLM as OpenAI GPT-3.5-turbo

    User->>API: POST /chat {query, session_id}
    API->>Mem: 获取历史会话记录 (滑动窗口截断)
    Mem-->>API: 历史记录 (List[Message])
    API->>Chain: invoke(query, history)
    Chain->>LLM: 重写问题 (考虑上下文)
    LLM-->>Chain: 独立的问题 (Standalone Query)
    Chain->>VDB: similarity_search(Standalone Query, top_k=5)
    VDB-->>Chain: 召回文档块列表 (List[Document])
    Chain->>LLM: 基于文档块生成答案 (Prompt)
    LLM-->>Chain: 回答与引用内容
    Chain->>Mem: 保存用户提问与 AI 回答
    Chain-->>API: 返回结果 (带 source_documents)
    API-->>User: SSE 流式或同步响应 (含溯源)
```
