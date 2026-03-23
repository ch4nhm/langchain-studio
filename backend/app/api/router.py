from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any
import uuid

from app.service.chat_service import process_qa_request, process_chat_stream, process_sql_request

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    source_documents: List[Any]
    session_id: str

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/ask", response_model=AskResponse)
async def ask_sync(request: ChatRequest):
    result = process_qa_request(request.query, request.session_id)
    return result

@router.post("/chat")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        process_chat_stream(request.query, request.session_id),
        media_type="text/event-stream"
    )

@router.post("/sql")
async def ask_sql(request: ChatRequest):
    result = process_sql_request(request.query, request.session_id)
    return result
