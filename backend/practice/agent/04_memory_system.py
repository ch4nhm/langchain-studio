"""
面试题 A4: Memory 是 Agent 的一个关键模块。
如何为 Agent 设计短期记忆和长期记忆系统？可以借助哪些外部工具或技术？

记忆类型：
  短期记忆 (Short-term / Working Memory)
    - 对话历史 (Conversation Buffer)
    - 摘要记忆 (Conversation Summary) - Token压缩
    - 滑动窗口 (Sliding Window)       - 保留最近K条
  长期记忆 (Long-term Memory)
    - 向量数据库存储 + 语义检索
    - 实体记忆 (Entity Memory)        - 记住对话中的实体信息
    - 知识图谱                         - 结构化长期记忆
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.3,
    dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
)

# =============================================================================
# 短期记忆 1: ConversationBufferMemory（完整对话缓冲）
# 优点：完整保留所有历史  缺点：对话过长时 Token 超限
# =============================================================================
def demo_buffer_memory():
    print("\n【短期记忆 1: Buffer Memory - 完整保留所有对话】")
    memory = ConversationBufferMemory(return_messages=True)

    conversations = [
        "我叫张三，是一名 Python 开发者。",
        "我正在学习 LangChain，主要关注 Agent 方向。",
        "你还记得我叫什么名字吗？我是做什么的？"
    ]
    for user_msg in conversations:
        # 加载历史
        history = memory.load_memory_variables({})["history"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个记忆力很好的助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"history": history, "input": user_msg})
        # 保存到记忆
        memory.save_context({"input": user_msg}, {"output": response})
        print(f"  用户: {user_msg}")
        print(f"  AI:   {response[:100]}")

    print(f"  [记忆中共 {len(memory.chat_memory.messages)} 条消息]")

# =============================================================================
# 短期记忆 2: ConversationBufferWindowMemory（滑动窗口）
# 优点：Token 可控  缺点：丢失窗口外的历史
# =============================================================================
def demo_window_memory():
    print("\n【短期记忆 2: Window Memory - 只保留最近 K 轮】")
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)  # 只保留最近2轮

    for i in range(5):
        user_msg = f"这是第 {i+1} 条消息，内容是关于话题{i+1}的讨论。"
        history = memory.load_memory_variables({})["history"]
        print(f"  轮次 {i+1}: 历史消息数={len(history)}, 发送: {user_msg[:30]}...")
        memory.save_context({"input": user_msg}, {"output": f"收到第{i+1}条消息"})

    print(f"  最终记忆中消息数: {len(memory.chat_memory.messages)} (k=2, 保留4条 = 2轮*2)")

# =============================================================================
# 短期记忆 3: ConversationSummaryMemory（摘要记忆）
# 优点：Token 高效，保留语义  缺点：摘要有信息损失
# =============================================================================
def demo_summary_memory():
    print("\n【短期记忆 3: Summary Memory - 自动摘要压缩历史】")
    memory = ConversationSummaryMemory(llm=llm, return_messages=False)

    dialogs = [
        ("我是一名机器学习工程师，工作了5年。", "您好，很高兴认识您！"),
        ("我最近在研究大语言模型的微调技术，主要用 LoRA 方法。", "LoRA 是一种很有效的参数高效微调方法。"),
        ("我们公司正在搭建一个内部知识库问答系统。", "这是 RAG 的典型应用场景。"),
    ]
    for human, ai in dialogs:
        memory.save_context({"input": human}, {"output": ai})

    summary = memory.load_memory_variables({})["history"]
    print(f"  原始对话: {sum(len(h+a) for h,a in dialogs)} 字符")
    print(f"  摘要结果: {summary}")
    print(f"  压缩后:   {len(summary)} 字符")

# =============================================================================
# 长期记忆: VectorStore Memory（向量数据库存储）
# 特点：语义检索，可跨会话，持久化
# 适用：记住用户偏好、历史事实、领域知识
# =============================================================================
def demo_vector_memory():
    print("\n【长期记忆: Vector Store Memory - 语义检索历史】")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
    )
    # 使用内存模式的 Chroma（不持久化，仅演示）
    vectorstore = Chroma(embedding_function=embeddings, collection_name="long_term_memory")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    # 模拟存入长期记忆（跨会话的重要信息）
    memory.save_context(
        {"input": "我最喜欢的编程语言是 Python，特别喜欢异步编程。"},
        {"output": "好的，我记住了您偏爱 Python 和异步编程。"}
    )
    memory.save_context(
        {"input": "我的团队规模是10人，主要做 NLP 方向的产品。"},
        {"output": "了解，您领导一个10人的NLP团队。"}
    )
    memory.save_context(
        {"input": "我们公司在北京，主要客户是金融和医疗行业。"},
        {"output": "明白了。"}
    )

    # 语义检索相关记忆
    query = "我应该用什么语言开发我们的 AI 系统？"
    relevant = memory.load_memory_variables({"prompt": query})
    print(f"  查询: '{query}'")
    print(f"  检索到的相关记忆:\n{relevant['history']}")

# =============================================================================
# 记忆系统架构总结图
# =============================================================================
def print_memory_architecture():
    print("""
    Agent 记忆系统完整架构：
    ┌─────────────────────────────────────────────────────────────┐
    │                     Agent Memory System                      │
    ├──────────────────────┬──────────────────────────────────────┤
    │   短期记忆            │   长期记忆                            │
    │   (Working Memory)   │   (Long-term Memory)                  │
    ├──────────────────────┼──────────────────────────────────────┤
    │ • Buffer Memory      │ • Vector DB (ChromaDB/Pinecone)       │
    │   完整保留所有对话    │   语义检索，跨会话持久化              │
    │                      │                                       │
    │ • Window Memory      │ • Entity Memory                       │
    │   滑动窗口，保留K轮  │   提取并记忆实体信息(人名/地名等)    │
    │                      │                                       │
    │ • Summary Memory     │ • Knowledge Graph                     │
    │   LLM摘要压缩历史    │   结构化知识存储与推理                │
    │                      │                                       │
    │ • Summary+Buffer     │ • 外部数据库 (PostgreSQL/Redis)       │
    │   混合策略           │   结构化持久化存储                    │
    ├──────────────────────┴──────────────────────────────────────┤
    │  选型建议：                                                  │
    │  短对话(<10轮)    → Buffer Memory                            │
    │  长对话(>20轮)    → Summary 或 Window Memory                 │
    │  跨会话用户画像   → Vector Store Memory                      │
    │  生产环境         → PostgresChatMessageHistory + 滑动窗口    │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("Agent 记忆系统演示")
    print("=" * 60)
    demo_buffer_memory()
    demo_window_memory()
    demo_summary_memory()
    demo_vector_memory()
    print_memory_architecture()
