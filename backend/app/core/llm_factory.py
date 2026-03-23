from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """初始化并返回 LLM 模型。
    
    Args:
        temperature (float): 模型的 temperature 参数。
        
    Returns:
        ChatOpenAI: 配置好的 ChatOpenAI 实例。
    """
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        openai_api_key=settings.OPENAI_API_KEY
    )
