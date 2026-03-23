from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    """
    项目全局配置类，基于 Pydantic 的 BaseSettings。
    它会自动从环境变量或 .env 文件中加载对应的配置项。
    """
    # 项目基础信息
    PROJECT_NAME: str = "LangChain Demo"
    
    # OpenAI API 密钥，必需提供以调用 LLM 和 Embedding
    OPENAI_API_KEY: str = ""
    
    # Chroma 向量数据库持久化存储目录
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # PostgreSQL 关系型数据库配置（用于存储聊天历史等）
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "langchain"
    
    # Redis 缓存配置（用于 API 限流等）
    REDIS_URL: str = "redis://localhost:6379/0"
    
    @property
    def DATABASE_URL(self) -> str:
        """动态计算 PostgreSQL 数据库连接 URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # 指定从项目根目录的 .env 文件中加载环境变量
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# 实例化单例配置对象，供整个应用导入使用
settings = Settings()
