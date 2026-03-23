from langchain_community.chat_models.tongyi import ChatTongyi
from app.core.config import settings

def get_llm(temperature: float = 0.0) -> ChatTongyi:
    """
    初始化并返回大语言模型 (LLM) 实例的工厂函数。
    集中管理 LLM 的实例化，方便后续如果需要切换模型时，只需修改此处一处代码即可。
    
    Args:
        temperature (float): 模型的随机性/创造性参数。
            - 0.0: 最确定、最保守的回答（适合信息提取、SQL 生成等）
            - 较高值: 更具创造性的回答（适合闲聊、创作等）
            默认值为 0.0。
        
    Returns:
        ChatTongyi: 配置好的基于阿里通义千问 API 的 Chat 模型实例。
    """
    # 如果环境变量中的 key 还是叫 OPENAI_API_KEY，为了兼容你可以继续用，
    # 或者建议在 .env 中新增 DASHSCOPE_API_KEY
    # 这里我们直接将原有的 key 传给千问（假设用户把千问的 key 填在了这个配置里）
    api_key = settings.OPENAI_API_KEY
    
    return ChatTongyi(
        model="qwen-turbo", # 指定使用的通义千问模型版本，如 qwen-turbo, qwen-plus, qwen-max
        temperature=temperature, # 传入温度参数
        dashscope_api_key=api_key # 传入千问所需的 API Key
    )
