"""
面试题 R3: 高级检索技术
- HyDE: 生成假设文档再检索
- MMR: 相关性+多样性平衡
- Multi-Query: 自动生成多个查询扩大召回
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import MultiQueryRetriever

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")

DOCS = [
    Document(page_content="ChromaDB 是开源向量数据库，支持本地持久化，适合快速原型和中小规模场景。"),
    Document(page_content="Pinecone 是云原生向量数据库，提供托管服务，适合生产环境大规模检索。"),
    Document(page_content="Milvus 是高性能开源向量数据库，支持十亿级向量，适合大规模生产部署。"),
    Document(page_content="pgvector 是 PostgreSQL 的向量扩展，适合已有 PG 基础设施的团队。"),
    Document(page_content="Weaviate 是开源向量数据库，内置 GraphQL 接口，支持混合搜索。"),
    Document(page_content="HNSW 是主流向量索引算法，在速度和召回率之间取得良好平衡。"),
    Document(page_content="余弦相似度衡量两向量夹角，值域[-1,1]，常用于文本相似度，不受向量长度影响。"),
    Document(page_content="RAG 评估指标：Recall@K（召回率）、MRR（平均倒数排名）、NDCG（归一化折损累积增益）。"),
]


def build_vs() -> Chroma:
    emb = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=API_KEY)
    return Chroma.from_documents(DOCS, embedding=emb, collection_name="adv_rag")


def demo_basic(vs, query):
    print(f"\n【基础相似度检索: '{query}'】")
    docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(query)
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:70]}")


def demo_mmr(vs, query):
    """
    MMR (Maximal Marginal Relevance):
    同时考虑相关性和多样性，避免返回高度重复的文档。
    lambda_mult: 0=最大多样性, 1=最大相关性
    """
    print(f"\n【MMR 检索（多样性+相关性）: '{query}'】")
    docs = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6, "lambda_mult": 0.5}
    ).invoke(query)
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:70]}")
    print("  注: 从 fetch_k=6 候选中选出既相关又多样的 k=3 个结果")


def demo_hyde(vs, query):
    """
    HyDE (Hypothetical Document Embeddings):
    先让 LLM 生成假设回答，再用假设回答的向量检索真实文档。
    解决 query(短) 与 document(长) 的语义鸿沟问题。
    """
    print(f"\n【HyDE 检索（假设文档嵌入）: '{query}'】")
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    hyde_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "根据问题生成一段可能的答案文档（80字内），捕捉相关概念和术语。"),
            ("human", "{q}")
        ]) | llm | StrOutputParser()
    )
    hypo_doc = hyde_chain.invoke({"q": query})
    print(f"  假设文档: {hypo_doc[:80]}...")
    docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(hypo_doc)
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content[:70]}")


def demo_multi_query(vs, query):
    """
    Multi-Query Retrieval:
    自动将一个查询扩展为多个语义相近的查询，扩大召回范围。
    """
    print(f"\n【Multi-Query 检索（多查询扩展）: '{query}'】")
    llm = ChatTongyi(model="qwen-turbo", temperature=0.5, dashscope_api_key=API_KEY)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vs.as_retriever(search_kwargs={"k": 2}),
        llm=llm
    )
    docs = retriever.invoke(query)
    unique = list({d.page_content: d for d in docs}.values())
    print(f"  召回 {len(unique)} 个唯一文档（已去重）")
    for i, d in enumerate(unique, 1):
        print(f"  [{i}] {d.page_content[:70]}")


if __name__ == "__main__":
    print("=" * 60)
    print("高级 RAG 检索技术演示")
    print("=" * 60)
    vs = build_vs()
    query = "有哪些向量数据库可以在生产环境使用？"
    demo_basic(vs, query)
    demo_mmr(vs, query)
    demo_hyde(vs, query)
    demo_multi_query(vs, query)
    print("""
    高级检索技术对比：
    ┌─────────────┬──────────────────────────────┬────────────────────┐
    │ 技术         │ 核心思想                      │ 适用场景            │
    ├─────────────┼──────────────────────────────┼────────────────────┤
    │ 基础相似度   │ 余弦/点积相似度排序           │ 通用场景            │
    │ MMR         │ 相关性+多样性双目标           │ 避免冗余结果        │
    │ HyDE        │ 假设文档弥补语义鸿沟          │ 短query检索长文档   │
    │ Multi-Query │ 多角度查询扩大召回            │ 提升召回率          │
    │ 混合检索     │ 向量+BM25 融合               │ 专业术语/精确匹配   │
    └─────────────┴──────────────────────────────┴────────────────────┘
    """)
