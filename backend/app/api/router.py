from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any
import uuid

from app.service.chat_service import process_qa_request, process_chat_stream, process_sql_request

# 初始化 FastAPI 路由器
router = APIRouter()

class ChatRequest(BaseModel):
    """
    通用聊天请求模型。
    """
    query: str # 用户输入的查询内容
    session_id: Optional[str] = None # 可选的会话ID，用于追踪多轮对话历史

class AskResponse(BaseModel):
    """
    同步问答接口的响应模型。
    """
    answer: str # AI 的回答
    source_documents: List[Any] # 回答所引用的源文档列表
    session_id: str # 当前对话的会话ID

@router.get("/health")
async def health_check():
    """
    健康检查接口。
    用于负载均衡器或监控系统确认服务是否存活。
    """
    return {"status": "ok"}

@router.post("/ask", response_model=AskResponse)
async def ask_sync(request: ChatRequest):
    """
    同步问答接口。
    接收用户请求，经过 RAG 链路处理后，一次性返回完整的回答和引用来源。
    """
    result = process_qa_request(request.query, request.session_id)
    return result

@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口 (Server-Sent Events, SSE)。
    适用于长文本生成，以打字机效果逐块将 AI 的回答推送给前端。
    """
    return StreamingResponse(
        process_chat_stream(request.query, request.session_id),
        media_type="text/event-stream"
    )

@router.post("/sql")
async def ask_sql(request: ChatRequest):
    """
    结构化数据查询接口 (SQL Agent)。
    允许用户使用自然语言提问，Agent 会自动生成并执行 SQL 语句，返回分析结果。
    """
    result = process_sql_request(request.query, request.session_id)
    return result
