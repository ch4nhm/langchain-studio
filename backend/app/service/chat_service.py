import uuid
import json
from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage
from app.chain.retrieval_qa import get_retrieval_qa_chain
from app.agent.sql_agent import get_sql_agent
from app.memory.session_manager import get_session_history, apply_sliding_window

def process_qa_request(query: str, session_id: str = None) -> Dict[str, Any]:
    if not session_id:
        session_id = str(uuid.uuid4())
        
    history = get_session_history(session_id)
    chat_history = history.messages
    # 应用滑动窗口截断（保留最后 10 条消息）
    chat_history = apply_sliding_window(chat_history, k=10)
    
    chain = get_retrieval_qa_chain()
    
    # 执行链
    response = chain.invoke({
        "question": query,
        "chat_history": chat_history
    })
    
    # 保存记忆
    history.add_user_message(query)
    history.add_ai_message(response["answer"])
    
    # 提取源文档
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
    if not session_id:
        session_id = str(uuid.uuid4())
        
    history = get_session_history(session_id)
    chat_history = apply_sliding_window(history.messages, k=10)
    
    chain = get_retrieval_qa_chain()
    
    full_answer = ""
    # 流式执行
    # 注意：由于检索步骤，完整的 LCEL 链可能无法很好地原生流式传输。
    # 在实际场景中，我们可能会在 LCEL 链上使用自定义回调处理器或 stream() 方法。
    # 这里我们模拟 SSE 数据块以演示 API 契约。
    
    try:
        async for chunk in chain.astream({
            "question": query,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                piece = chunk["answer"]
                full_answer += piece
                yield f"data: {json.dumps({'chunk': piece, 'session_id': session_id})}\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
        
    history.add_user_message(query)
    history.add_ai_message(full_answer)
    yield "data: [DONE]\n\n"

def process_sql_request(query: str, session_id: str = None) -> Dict[str, Any]:
    if not session_id:
        session_id = str(uuid.uuid4())
        
    agent = get_sql_agent()
    response = agent.invoke({"input": query})
    
    return {
        "answer": response.get("output", ""),
        "sql_executed": "N/A", # 需要自定义回调才能精确提取
        "raw_result": "N/A",
        "session_id": session_id
    }
