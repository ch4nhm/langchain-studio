from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
import chromadb

def get_embeddings():
    """获取标准的 Embedding 模型。"""
    return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=settings.OPENAI_API_KEY)

def get_vector_store() -> Chroma:
    """初始化并返回 Chroma 向量存储。"""
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
    embeddings = get_embeddings()
    return Chroma(
        client=client,
        collection_name="tech_docs",
        embedding_function=embeddings
    )
