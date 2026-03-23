from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse

class LangChainException(Exception):
    """所有自定义 LangChain 错误的基类。"""
    
    def __init__(self, code: str, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        """
        Args:
            code (str): 错误代码。
            message (str): 人类可读的错误信息。
            status_code (int): HTTP 状态码。
            details (dict): 额外的错误上下文。
        """
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

async def langchain_exception_handler(request: Request, exc: LangChainException) -> JSONResponse:
    """在 FastAPI 中全局处理 LangChainException。"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    content = {
        "code": exc.code,
        "message": exc.message,
        "request_id": request_id,
        "details": exc.details
    }
    return JSONResponse(status_code=exc.status_code, content=content)
