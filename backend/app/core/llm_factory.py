from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    初始化并返回大语言模型 (LLM) 实例的工厂函数。
    集中管理 LLM 的实例化，方便后续如果需要切换模型（如换成 AzureOpenAI、千问等）时，
    只需修改此处一处代码即可。
    
    Args:
        temperature (float): 模型的随机性/创造性参数。
            - 0.0: 最确定、最保守的回答（适合信息提取、SQL 生成等）
            - 较高值: 更具创造性的回答（适合闲聊、创作等）
            默认值为 0.0。
        
    Returns:
        ChatOpenAI: 配置好的基于 OpenAI API 的 Chat 模型实例。
    """
    return ChatOpenAI(
        model="gpt-3.5-turbo", # 指定使用的模型版本
        temperature=temperature, # 传入温度参数
        openai_api_key=settings.OPENAI_API_KEY # 从全局配置中读取 API Key
    )
