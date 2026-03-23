from typing import Any, List
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import filter_messages
from app.core.config import settings

def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    """初始化并返回基于 Postgres 的聊天消息历史记录。
    
    Args:
        session_id (str): 唯一的会话标识符。
        
    Returns:
        PostgresChatMessageHistory: 持久化的聊天历史记录对象。
    """
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=settings.DATABASE_URL,
        table_name="message_store"
    )

def apply_sliding_window(messages: List[BaseMessage], k: int = 10) -> List[BaseMessage]:
    """对消息应用简单的滑动窗口截断。
    
    Args:
        messages (List[BaseMessage]): 原始消息列表。
        k (int): 要保留的最近消息的数量。
        
    Returns:
        List[BaseMessage]: 截断后的消息列表。
    """
    if len(messages) <= k:
        return messages
    # 如果存在系统消息，则保留它
    if messages and messages[0].type == "system":
        return [messages[0]] + messages[-(k-1):]
    return messages[-k:]

def apply_token_trimming(messages: List[BaseMessage], max_tokens: int = 2000) -> List[BaseMessage]:
    """修剪消息以适应特定的 token 限制。
    
    Args:
        messages (List[BaseMessage]): 原始消息列表。
        max_tokens (int): 最大 token 限制。
        
    Returns:
        List[BaseMessage]: 经过 token 修剪的消息列表。
    """
    # 使用 Langchain 内置的消息过滤器作为 token 计数的替代
    # 注意：在生产场景中，你会传递一个 token 计数器函数。
    # filter_messages 可以丢弃最旧的消息
    from langchain_core.messages import filter_messages
    
    # 简单的占位符：假设平均每个词 20 个 token（这里实现中是 * 1.5，我们仅保留中文注释意思即可）
    def _token_count(msg: BaseMessage) -> int:
        return len(str(msg.content).split()) * 1.5
        
    total_tokens = sum(_token_count(m) for m in messages)
    while total_tokens > max_tokens and len(messages) > 1:
        # 丢弃最旧的非系统消息
        idx = 1 if messages[0].type == "system" else 0
        dropped = messages.pop(idx)
        total_tokens -= _token_count(dropped)
        
    return messages

def get_session_history(session_id: str) -> Any:
    """获取给定会话的消息历史记录，由 Postgres 管理。
    
    Args:
        session_id (str): 唯一的会话标识符。
        
    Returns:
        PostgresChatMessageHistory: 聊天历史记录对象。
    """
    return get_postgres_history(session_id)
