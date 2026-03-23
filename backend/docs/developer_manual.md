# 开发者手册

本系统基于 LangChain 开发，以下是核心拓展指南：

## 1. 如何新增一条 Chain
1. 在 `backend/app/chain/` 目录下创建新文件，例如 `my_chain.py`。
2. 继承 `Runnable` 协议，或使用 `RunnableSequence` 组装你的提示词、模型和输出解析器。
3. 定义输入与输出的 Pydantic 模型作为契约。
4. 在 `backend/app/api/router.py` 中注册相应的接口。

## 2. 如何切换 LLM
系统在 `backend/app/core/llm_factory.py` 中管理模型实例：
```python
from langchain_openai import ChatOpenAI
# 若需切换为 AzureOpenAI:
from langchain_openai import AzureChatOpenAI

def get_llm():
    # 读取环境变量进行实例化
    return ChatOpenAI(temperature=0)
```

## 3. 如何替换向量库
在 `backend/app/retrieval/vector_store.py` 中：
```python
from langchain_community.vectorstores import Chroma
# 或者替换为 PGVector
from langchain_community.vectorstores.pgvector import PGVector

def get_vector_store():
    # 返回对应的 VectorStore 实例
    pass
```

## 4. 运行与调试 ETL 流水线
使用 `make ingest` 命令：
1. 确保 `.env` 中的 `OPENAI_API_KEY` 有效。
2. 将源文件（`.md`）放入 `data/` 目录。
3. 执行清洗与向量化过程。
