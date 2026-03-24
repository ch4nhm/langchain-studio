"""
面试题 R4: Lost in the Middle 问题与缓解方法
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# 构造多个文档，关键答案分散在不同位置
DOCS = [
    Document(page_content="文档A: LangChain 创立于2022年，支持多种LLM。【关键】最新版本是0.3.x。社区活跃。", metadata={"source": "A", "score": 0.92}),
    Document(page_content="文档B: RAG由Facebook AI于2020年提出。【关键】主要解决LLM知识截止和幻觉问题。", metadata={"source": "B", "score": 0.88}),
    Document(page_content="文档C（不相关）: Python是编程语言，1991年创建，以简洁语法著称，广泛用于AI领域。", metadata={"source": "C", "score": 0.25}),
    Document(page_content="文档D（不相关）: Docker是容器化平台，简化部署流程。Kubernetes是编排工具。", metadata={"source": "D", "score": 0.20}),
    Document(page_content="文档E: Agent是自主AI系统，ReAct是经典框架。【关键】核心组件：LLM、Tools、Memory、Planning。", metadata={"source": "E", "score": 0.85}),
]


def ask_with_docs(docs, question, label):
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"请根据以下上下文回答问题。上下文：\n{context}"),
        ("human", "{q}")
    ])
    ans = (prompt | llm | StrOutputParser()).invoke({"q": question})
    print(f"  {label}:")
    print(f"    {ans[:150]}")


def demo_lost_in_middle():
    """展示关键信息在中间vs首尾的差异"""
    print("\n【Lost in the Middle 问题演示】")
    q = "LangChain最新版本？RAG解决什么问题？Agent核心组件？"

    # 关键文档(A,B,E)夹在不相关文档(C,D)中间
    middle = [DOCS[0], DOCS[2], DOCS[1], DOCS[3], DOCS[4]]  # 关键分散在中间
    # 关键文档放在首尾
    ends   = [DOCS[0], DOCS[1], DOCS[2], DOCS[3], DOCS[4]]  # 关键在首尾

    ask_with_docs(middle, q, "关键文档夹在中间（效果差）")
    ask_with_docs(ends,   q, "关键文档放首尾（效果好）")


def demo_rerank():
    """重排序：最相关文档放首尾，缓解 Lost in the Middle"""
    print("\n【缓解方案1: 重排序（最相关放首尾）】")
    ranked = sorted(DOCS, key=lambda d: d.metadata["score"], reverse=True)
    # 奇偶交错放置：最高分→位置0，次高分→末尾，第三高→位置1...
    reordered, left, right = [], True, []
    for d in ranked:
        if left:
            reordered.append(d)
        else:
            right.append(d)
        left = not left
    reordered = reordered + right[::-1]
    print("  重排序后顺序:")
    for i, d in enumerate(reordered, 1):
        print(f"  位置{i} [文档{d.metadata['source']}] score={d.metadata['score']:.2f}: {d.page_content[:50]}...")


def demo_compression():
    """上下文压缩：用LLM提取每个文档中最相关的句子"""
    print("\n【缓解方案2: 上下文压缩（LLMChainExtractor）】")
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    emb = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=API_KEY)
    vs = Chroma.from_documents(DOCS, embedding=emb, collection_name="litm")
    base = vs.as_retriever(search_kwargs={"k": 4})
    compressor = LLMChainExtractor.from_llm(llm)
    comp_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
    q = "LangChain最新版本是多少？"
    docs = comp_retriever.invoke(q)
    print(f"  问题: '{q}'")
    print(f"  压缩后保留 {len(docs)} 个片段:")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:100]}")


def demo_map_reduce():
    """Map-Reduce：先对每文档单独问答，再汇总"""
    print("\n【缓解方案3: Map-Reduce（逐文档问答后汇总）】")
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    q = "这段文字中有关于LangChain版本、RAG用途或Agent组件的信息吗？有则提取，无则回答无。"
    map_chain = ChatPromptTemplate.from_messages([
        ("system", "从以下文档中提取与问题相关的信息，无相关信息则说无。"),
        ("human", "文档: {doc}\n问题: {q}")
    ]) | llm | StrOutputParser()

    partial_answers = []
    for doc in DOCS[:3]:  # 仅演示前3个
        ans = map_chain.invoke({"doc": doc.page_content, "q": q})
        partial_answers.append(ans)
        print(f"  Map [{doc.metadata['source']}]: {ans[:60]}...")

    reduce_chain = ChatPromptTemplate.from_messages([
        ("system", "综合以下各文档的提取结果，给出最终完整答案。"),
        ("human", "各文档提取结果:\n{parts}")
    ]) | llm | StrOutputParser()
    final = reduce_chain.invoke({"parts": "\n".join(partial_answers)})
    print(f"  Reduce 汇总: {final[:150]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Lost in the Middle 问题与缓解方案")
    print("=" * 60)
    demo_lost_in_middle()
    demo_rerank()
    demo_compression()
    demo_map_reduce()
    print("""
    缓解方案总结：
    1. 减少 k       仅返回2-3个最相关文档，减少干扰
    2. 重排序       Rerank后将最高分放首位和末位
    3. 上下文压缩   LLM提取每文档中最相关句子
    4. Map-Reduce  逐文档问答后汇总，不受位置影响
    """)
