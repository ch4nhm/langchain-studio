from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
import chromadb

def get_embeddings():
    """
    获取标准的文本嵌入 (Embedding) 模型。
    用于将文本片段和用户的查询转换为高维向量，以便进行相似度计算。
    默认使用 OpenAI 的 text-embedding-ada-002 模型。
    """
    return OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key=settings.OPENAI_API_KEY
    )

def get_vector_store() -> Chroma:
    """
    初始化并返回 Chroma 向量存储实例。
    连接到本地持久化的 Chroma 数据库，加载文档的向量数据。
    
    Returns:
        Chroma: 配置好集合名称和嵌入函数的 Chroma 实例。
    """
    # 初始化 Chroma 的持久化客户端，指定存储路径
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
    
    # 获取嵌入函数
    embeddings = get_embeddings()
    
    # 返回 Chroma 向量库封装，指定使用的集合(collection)
    return Chroma(
        client=client,
        collection_name="tech_docs", # 存储技术文档的集合名称，需与数据导入时一致
        embedding_function=embeddings
    )
