"""
面试题 R1: RAG 完整流水线
离线阶段: 文档加载 -> 切块 -> Embedding -> 存入向量库
在线阶段: 用户提问 -> Embedding -> 向量检索 -> 拼装Prompt -> LLM生成
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")

RAW_DOCUMENTS = [
    """LangChain 是用于构建大语言模型应用的开源框架，由 Harrison Chase 于2022年创立。
它提供 Chain、Agent、Memory、RAG 等核心抽象，支持 Python 和 JavaScript。
LCEL（LangChain Expression Language）是最新的链式构建语法，使用管道符 | 连接组件。""",

    """RAG（Retrieval-Augmented Generation）由 Facebook AI Research 在2020年提出。
核心思想：生成答案前先从外部知识库检索相关信息，作为上下文提供给 LLM。
解决了 LLM 知识截止、幻觉、无法访问私有数据等核心问题。
RAG vs 微调：RAG 不改变模型权重，数据可实时更新，成本低；微调改变权重，成本高。""",

    """向量数据库专门存储和检索高维向量。主流选项：ChromaDB（开源本地）、
Pinecone（云服务）、Weaviate（开源）、Milvus（高性能）、pgvector（PostgreSQL扩展）。
检索算法：HNSW（分层可导航小世界图）、IVF（倒排文件索引）、FLAT（暴力搜索）。
余弦相似度是最常用的度量方式，值域[-1,1]，越接近1越相似。""",

    """Embedding 将文本转换为高维向量，语义相似的文本在向量空间中距离相近。
常用模型：text-embedding-ada-002（OpenAI，1536维）、text-embedding-v1（通义千问）、
bge-large-zh（BAAI，1024维，中文优化）。
评估指标：MTEB 基准、检索召回率 Recall@K、语义相似度相关性。""",

    """Agent 是能感知环境、自主规划并采取行动的 AI 系统。
工作循环：接收目标 -> 规划步骤 -> 选择工具 -> 执行 -> 观察结果 -> 循环。
ReAct（Reasoning + Acting）是最经典框架，将思维链和工具调用结合。""",
]


def build_knowledge_base() -> Chroma:
    """离线阶段：构建向量知识库"""
    print("\n【离线阶段：构建知识库】")

    docs = [
        Document(page_content=t, metadata={"source": f"doc_{i}"})
        for i, t in enumerate(RAW_DOCUMENTS)
    ]
    print(f"  Step1: 加载原始文档 {len(docs)} 篇")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Step2: 切块完成，共 {len(chunks)} 个 chunk")
    print(f"  示例: '{chunks[0].page_content[:60]}...'")

    embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=API_KEY)
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, collection_name="rag_demo")
    print(f"  Step3: 向量化并存入 ChromaDB 完成")
    return vectorstore


def build_rag_chain(vectorstore: Chroma):
    """在线阶段：构建 RAG 查询链"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是专业的知识库问答助手。仅根据以下上下文回答问题，无相关信息时说'无法回答'，不要编造。\n\n"
         "上下文：\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n---\n".join(
            f"[来源:{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
        )

    # LCEL 构建 RAG 链
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def rag_query_with_sources(question: str, chain, retriever):
    """带引用来源的 RAG 查询"""
    print(f"\n【在线阶段：RAG 查询】")
    print(f"问题: {question}")

    # 单独展示检索结果
    docs = retriever.invoke(question)
    print(f"\n检索到 {len(docs)} 个相关片段:")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:80]}...")

    answer = chain.invoke(question)
    print(f"\nLLM 生成答案: {answer}")
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("RAG 完整流水线演示")
    print("=" * 60)

    vectorstore = build_knowledge_base()
    chain, retriever = build_rag_chain(vectorstore)

    questions = [
        "LangChain 的核心设计理念是什么？",
        "RAG 和微调有什么区别？",
        "选择向量数据库时应该考虑哪些因素？",
        "今天股市行情如何？",  # 测试无关问题
    ]
    for q in questions:
        rag_query_with_sources(q, chain, retriever)
        print()

    print("""
    RAG vs 微调对比：
    ┌───────────────┬──────────────────────┬──────────────────────┐
    │ 维度           │ RAG                  │ 微调 (Fine-tuning)   │
    ├───────────────┼──────────────────────┼──────────────────────┤
    │ 知识更新       │ 实时更新向量库        │ 需重新训练           │
    │ 成本           │ 低（只更新数据）      │ 高（GPU训练）        │
    │ 幻觉控制       │ 好（有检索依据）      │ 较差                 │
    │ 私有数据       │ 支持                 │ 支持                 │
    │ 推理延迟       │ 稍高（含检索步骤）   │ 正常                 │
    │ 适用场景       │ 知识库问答，文档检索  │ 特定风格/任务适配    │
    └───────────────┴──────────────────────┴──────────────────────┘
    """)
