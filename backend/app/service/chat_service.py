import uuid
import json
from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage
from app.chain.retrieval_qa import get_retrieval_qa_chain
from app.agent.sql_agent import get_sql_agent
from app.memory.session_manager import get_session_history, apply_sliding_window

def process_qa_request(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    处理同步的问答请求 (RAG)。
    
    Args:
        query (str): 用户的提问。
        session_id (str, optional): 会话ID。如果不提供，将自动生成一个。
        
    Returns:
        Dict: 包含回答、引用源文档和会话ID的字典。
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # 获取并处理历史记录
    history = get_session_history(session_id)
    chat_history = history.messages
    # 应用滑动窗口截断（保留最后 10 条消息），防止历史过长导致 Token 超限
    chat_history = apply_sliding_window(chat_history, k=10)
    
    # 获取预配置的检索问答链
    chain = get_retrieval_qa_chain()
    
    # 阻塞执行整个链条
    response = chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    # 将本次对话的问答存入数据库进行持久化
    history.add_user_message(query)
    history.add_ai_message(response["answer"])
    
    # 提取被 LLM 采纳为上下文的源文档信息，供前端展示引用
    sources = []
    for doc in response.get("context", []):
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
        
    return {
        "answer": response["answer"],
        "source_documents": sources,
        "session_id": session_id
    }

async def process_chat_stream(query: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    处理流式对话请求 (RAG)，用于 SSE (Server-Sent Events) 推送。
    
    Args:
        query (str): 用户的提问。
        session_id (str, optional): 会话ID。
        
    Yields:
        str: 格式化为 SSE 规范的字符串数据块。
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # 获取并处理历史记录
    history = get_session_history(session_id)
    chat_history = apply_sliding_window(history.messages, k=10)
    
    # 获取预配置的检索问答链
    chain = get_retrieval_qa_chain()
    
    full_answer = ""
    
    try:
        # 使用 astream 异步流式执行链条
        async for chunk in chain.astream({
            "question": query,
            "chat_history": chat_history
        }):
            # 由于我们的链输出包含多个 key，这里只关注 "answer" 字段的生成片段
            if "answer" in chunk:
                piece = chunk["answer"]
                full_answer += piece
                # 按照 SSE 的格式要求 yield 数据
                yield f"data: {json.dumps({'chunk': piece, 'session_id': session_id})}\n\n"
            
    except Exception as e:
        # 如果发生异常，推送错误信息给前端
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
        
    # 当流式生成完全结束后，将完整的回答存入数据库
    history.add_user_message(query)
    history.add_ai_message(full_answer)
    
    # 发送结束标识
    yield "data: [DONE]\n\n"

def process_sql_request(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    处理 SQL 代理请求。
    将自然语言转换为 SQL 并在数据库中执行。
    
    Args:
        query (str): 用户的自然语言提问（如“统计有多少个用户”）。
        session_id (str, optional): 会话ID。
        
    Returns:
        Dict: 包含回答和执行细节的字典。
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # 获取配置好的 SQL Agent
    agent = get_sql_agent()
    
    # 执行 Agent。Agent 内部会自动进行多次思考-行动(Thought-Action)循环
    response = agent.invoke({"input": query})
    
    return {
        "answer": response.get("output", ""),
        "sql_executed": "N/A", # 注意：标准 LangChain SQLAgent 不直接暴露最后执行的 SQL，需要通过自定义 CallbackHandler 才能精确提取
        "raw_result": "N/A",
        "session_id": session_id
    }
