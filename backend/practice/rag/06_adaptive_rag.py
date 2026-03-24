"""
面试题 R6: 自适应 RAG / 多轮检索
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")

DOCS = [
    Document(page_content="LangChain 0.3.x于2024年发布，改进了LCEL语法和流式支持。"),
    Document(page_content="RAG通过检索外部知识增强LLM准确性，解决知识截止和幻觉问题。"),
    Document(page_content="Agent使用ReAct框架，通过思考-行动-观察循环完成复杂任务。"),
    Document(page_content="向量数据库使用HNSW算法实现高效近似最近邻搜索。"),
    Document(page_content="Embedding将文本转为高维向量，常用维度768、1024、1536。"),
]


def get_llm():
    return ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)


def build_retriever():
    emb = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=API_KEY)
    vs = Chroma.from_documents(DOCS, embedding=emb, collection_name="adaptive_rag")
    return vs.as_retriever(search_kwargs={"k": 2})


def self_rag(question: str, retriever):
    """
    Self-RAG: LLM 自判断是否需要检索，以及检索结果是否充分。
    关键 token: [Retrieve] / [No Retrieve] / [Relevant] / [Irrelevant]
    """
    print(f"\n【Self-RAG: '{question}'】")
    llm = get_llm()

    need = (ChatPromptTemplate.from_messages([
        ("system", "问题是否需要检索外部知识？通用常识/计算回答NO，专业事实/私有数据回答YES。只回答YES或NO。"),
        ("human", "{q}")
    ]) | llm | StrOutputParser()).invoke({"q": question}).strip().upper()
    print(f"  需要检索: {need}")

    if need == "NO":
        ans = (ChatPromptTemplate.from_messages([("human", "{q}")]) | llm | StrOutputParser()).invoke({"q": question})
        print(f"  直接回答: {ans[:100]}")
        return ans

    docs = retriever.invoke(question)
    ctx = "\n".join(d.page_content for d in docs)
    print(f"  检索到 {len(docs)} 个文档")

    sufficient = (ChatPromptTemplate.from_messages([
        ("system", "上下文是否足够回答问题？只回答YES或NO。"),
        ("human", "问题: {q}\n上下文: {ctx}")
    ]) | llm | StrOutputParser()).invoke({"q": question, "ctx": ctx}).strip().upper()
    print(f"  检索充分: {sufficient}")

    ans = (ChatPromptTemplate.from_messages([
        ("system", "根据上下文回答问题，不足时说明。上下文: {ctx}"),
        ("human", "{q}")
    ]) | llm | StrOutputParser()).invoke({"q": question, "ctx": ctx})
    print(f"  最终答案: {ans[:120]}")
    return ans


def iterative_rag(question: str, retriever, max_rounds: int = 2):
    """
    Iterative RAG: 多轮检索，每轮识别知识缺口并补充检索。
    适合需要多个知识点组合的复杂问题。
    """
    print(f"\n【Iterative RAG: '{question}'】")
    llm = get_llm()
    context, answer, gap = "", "", ""

    for r in range(1, max_rounds + 1):
        print(f"  第{r}轮检索...")
        query = question if r == 1 else f"{question} 补充: {gap}"
        ctx_new = "\n".join(d.page_content for d in retriever.invoke(query))
        context += f"\n{ctx_new}"

        answer = (ChatPromptTemplate.from_messages([
            ("system", f"根据上下文尽可能完整回答。上下文:\n{context}"),
            ("human", "{q}")
        ]) | llm | StrOutputParser()).invoke({"q": question})

        gap = (ChatPromptTemplate.from_messages([
            ("system", "答案中还缺少什么信息？已完整则回答'无'。20字内。"),
            ("human", "问题:{q}\n当前答案:{ans}")
        ]) | llm | StrOutputParser()).invoke({"q": question, "ans": answer})
        print(f"  知识缺口: {gap}")
        if "无" in gap or "完整" in gap:
            break

    print(f"  最终答案: {answer[:150]}")
    return answer


def adaptive_rag(question: str, retriever):
    """
    Adaptive RAG: 根据问题复杂度动态选择策略。
    简单 -> 直接回答; 单跳 -> 单次检索; 多跳 -> 多轮检索
    """
    print(f"\n【Adaptive RAG: '{question}'】")
    llm = get_llm()

    strategy = (ChatPromptTemplate.from_messages([
        ("system",
         "判断问题类型，只回答以下之一:\n"
         "DIRECT - 通用常识，无需检索\n"
         "SINGLE - 单一知识点，一次检索足够\n"
         "MULTI  - 复杂多跳，需要多轮检索"),
        ("human", "{q}")
    ]) | llm | StrOutputParser()).invoke({"q": question}).strip().upper()

    print(f"  选择策略: {strategy}")
    if "DIRECT" in strategy:
        ans = (ChatPromptTemplate.from_messages([("human", "{q}")]) | llm | StrOutputParser()).invoke({"q": question})
        print(f"  直接回答: {ans[:100]}")
    elif "MULTI" in strategy:
        ans = iterative_rag(question, retriever, max_rounds=2)
    else:
        docs = retriever.invoke(question)
        ctx = "\n".join(d.page_content for d in docs)
        ans = (ChatPromptTemplate.from_messages([
            ("system", "根据上下文回答问题。上下文: {ctx}"),
            ("human", "{q}")
        ]) | llm | StrOutputParser()).invoke({"q": question, "ctx": ctx})
        print(f"  单次检索答案: {ans[:120]}")
    return ans


if __name__ == "__main__":
    print("=" * 60)
    print("自适应 RAG / 多轮检索演示")
    print("=" * 60)
    retriever = build_retriever()

    self_rag("1+1等于多少？", retriever)           # 应走 NO 直接回答
    self_rag("LangChain最新版本是多少？", retriever) # 应走检索

    iterative_rag("LangChain和RAG如何结合使用？", retriever)

    adaptive_rag("Python如何定义函数？", retriever)  # DIRECT
    adaptive_rag("向量数据库用什么算法？", retriever)  # SINGLE
    adaptive_rag("LangChain的RAG如何利用向量数据库和Embedding？", retriever)  # MULTI

    print("""
    高级RAG范式对比：
    ┌──────────────────┬─────────────────────────────────────┐
    │ 范式              │ 核心思想                             │
    ├──────────────────┼─────────────────────────────────────┤
    │ Naive RAG        │ 一次检索，一次生成                   │
    │ Self-RAG         │ LLM自判断是否需要检索及质量评估      │
    │ Iterative RAG    │ 多轮检索，识别并填补知识缺口         │
    │ Corrective RAG   │ 评估检索质量，差则触发Web搜索补救    │
    │ Adaptive RAG     │ 根据问题复杂度动态路由到不同策略     │
    │ Graph RAG        │ 用知识图谱替代向量库，捕捉关系推理   │
    └──────────────────┴─────────────────────────────────────┘
    """)
