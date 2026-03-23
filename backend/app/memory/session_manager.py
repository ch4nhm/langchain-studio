from typing import Any, List
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import filter_messages
from app.core.config import settings

def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    """
    初始化并返回基于 PostgreSQL 数据库的聊天消息历史记录对象。
    它会自动在指定的表中存储和读取给定 session_id 的对话历史，实现记忆持久化。
    
    Args:
        session_id (str): 唯一的会话标识符，用于区分不同的用户或对话流。
        
    Returns:
        PostgresChatMessageHistory: 与数据库连接的聊天历史记录对象。
    """
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=settings.DATABASE_URL,
        table_name="message_store" # 指定在数据库中存储消息的表名
    )

def apply_sliding_window(messages: List[BaseMessage], k: int = 10) -> List[BaseMessage]:
    """
    对消息列表应用滑动窗口截断机制。
    只保留最近的 k 条消息，防止随着对话的进行，上下文过长导致超出 LLM 的 Token 限制。
    
    Args:
        messages (List[BaseMessage]): 原始的历史消息列表。
        k (int): 要保留的最近消息的数量。
        
    Returns:
        List[BaseMessage]: 截断后的消息列表。
    """
    # 如果总消息数未超过窗口大小，直接返回
    if len(messages) <= k:
        return messages
    
    # 如果列表的第一条是系统消息(SystemMessage)，通常包含重要的全局指令，必须保留
    if messages and messages[0].type == "system":
        return [messages[0]] + messages[-(k-1):]
    
    # 否则直接截取最后 k 条消息
    return messages[-k:]

def apply_token_trimming(messages: List[BaseMessage], max_tokens: int = 2000) -> List[BaseMessage]:
    """
    (高级功能备用) 修剪消息以适应特定的 token 限制，比滑动窗口更精确。
    
    Args:
        messages (List[BaseMessage]): 原始消息列表。
        max_tokens (int): 允许的最大 token 数量。
        
    Returns:
        List[BaseMessage]: 经过 token 修剪的消息列表。
    """
    # 注意：在实际生产场景中，你应该使用 tiktoken 等库提供真实的 token 计数器函数。
    # 这里提供一个简单的估算实现作为演示。
    
    # 简单的占位符：假设平均每个词消耗 1.5 个 token
    def _token_count(msg: BaseMessage) -> int:
        return len(str(msg.content).split()) * 1.5
        
    total_tokens = sum(_token_count(m) for m in messages)
    
    # 当总 token 数超限，且列表中还有超过1条消息时循环丢弃最旧的消息
    while total_tokens > max_tokens and len(messages) > 1:
        # 如果第一条是系统消息，则跳过它，丢弃索引为 1 的消息；否则丢弃索引为 0 的消息
        idx = 1 if messages[0].type == "system" else 0
        dropped = messages.pop(idx)
        total_tokens -= _token_count(dropped)
        
    return messages

def get_session_history(session_id: str) -> Any:
    """
    获取给定会话的消息历史记录的对外统一接口。
    默认使用 Postgres 管理的持久化历史。
    
    Args:
        session_id (str): 唯一的会话标识符。
        
    Returns:
        PostgresChatMessageHistory: 聊天历史记录对象。
    """
    return get_postgres_history(session_id)
