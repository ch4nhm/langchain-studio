from typing import Any, Dict
from pydantic import BaseModel, Field
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from app.core.config import settings
from app.core.llm_factory import get_llm
from app.core.exceptions import LangChainException

class SQLAgentInput(BaseModel):
    """SQL Agent 的输入模型。"""
    query: str = Field(..., description="用户对数据库的自然语言查询。")
    session_id: str = Field(..., description="唯一的会话标识符。")

class SQLAgentOutput(BaseModel):
    """SQL Agent 的输出模型。"""
    answer: str = Field(..., description="对 SQL 执行结果的自然语言解释。")
    sql_executed: str = Field(..., description="执行的 SQL 查询语句。")
    raw_result: Any = Field(..., description="数据库的原始执行结果。")

def get_sql_database() -> SQLDatabase:
    """初始化并返回 SQL 数据库连接。"""
    try:
        return SQLDatabase.from_uri(settings.DATABASE_URL)
    except Exception as e:
        raise LangChainException(
            code="DB_CONNECTION_ERROR",
            message=f"连接数据库失败: {str(e)}",
            status_code=500
        )

def get_sql_agent() -> Any:
    """构建并返回结构化 SQL Agent。
    
    Returns:
        AgentExecutor: 配置好的 SQL Agent 执行器。
    """
    llm = get_llm(temperature=0)
    db = get_sql_database()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=3, # 最多重试 3 次
        handle_parsing_errors=True # 解析错误自动重试
    )
    
    return agent_executor
