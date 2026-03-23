from typing import Any, Dict
from pydantic import BaseModel, Field
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from app.core.config import settings
from app.core.llm_factory import get_llm
from app.core.exceptions import LangChainException

class SQLAgentInput(BaseModel):
    """SQL Agent 的输入模型，定义了接口请求的结构。"""
    query: str = Field(..., description="用户对数据库的自然语言查询。")
    session_id: str = Field(..., description="唯一的会话标识符，用于保持对话上下文。")

class SQLAgentOutput(BaseModel):
    """SQL Agent 的输出模型，定义了接口返回的结构。"""
    answer: str = Field(..., description="对 SQL 执行结果的自然语言解释。")
    sql_executed: str = Field(..., description="实际执行的 SQL 查询语句。")
    raw_result: Any = Field(..., description="从数据库返回的原始执行结果。")

def get_sql_database() -> SQLDatabase:
    """
    初始化并返回 SQL 数据库连接实例。
    
    Returns:
        SQLDatabase: LangChain 封装的 SQL 数据库对象。
        
    Raises:
        LangChainException: 如果数据库连接失败，抛出自定义异常。
    """
    try:
        return SQLDatabase.from_uri(settings.DATABASE_URL)
    except Exception as e:
        raise LangChainException(
            code="DB_CONNECTION_ERROR",
            message=f"连接数据库失败: {str(e)}",
            status_code=500
        )

def get_sql_agent() -> Any:
    """
    构建并返回结构化的 SQL Agent 执行器。
    该 Agent 能够理解自然语言，将其转换为 SQL 语句，并在数据库上执行以获取答案。
    
    Returns:
        AgentExecutor: 配置好的 SQL Agent 执行器。
    """
    # 1. 获取语言模型实例（通常使用 temperature=0 保证 SQL 生成的确定性）
    llm = get_llm(temperature=0)
    
    # 2. 获取数据库连接
    db = get_sql_database()
    
    # 3. 创建 SQL 工具包，为 Agent 提供操作数据库的能力（如查看表结构、执行查询等）
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 4. 使用 LangChain 内置的工厂方法创建 SQL Agent
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True, # 开启详细日志，便于调试
        agent_type="openai-tools", # 使用 OpenAI 的 Function Calling 能力来驱动 Agent
        max_iterations=3, # 设置最大迭代次数，防止 Agent 陷入无限循环（最多重试 3 次）
        handle_parsing_errors=True # 开启解析错误自动重试机制
    )
    
    return agent_executor
