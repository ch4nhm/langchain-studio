from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse

class LangChainException(Exception):
    """
    所有自定义业务和 LangChain 相关错误的基类。
    继承此异常可以方便地在 FastAPI 中进行全局错误捕获和格式化返回。
    """
    
    def __init__(self, code: str, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        """
        初始化自定义异常。
        
        Args:
            code (str): 错误代码（例如 "DB_CONNECTION_ERROR"），供前端进行针对性处理。
            message (str): 人类可读的错误信息，可以直接展示给用户。
            status_code (int): 对应的 HTTP 状态码，默认为 500 (内部服务器错误)。
            details (dict): 额外的错误上下文或调试信息（可选）。
        """
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

async def langchain_exception_handler(request: Request, exc: LangChainException) -> JSONResponse:
    """
    FastAPI 中的全局异常处理器。
    当应用抛出 LangChainException 时，拦截该异常并转换为标准格式的 JSON 响应。
    
    Args:
        request: 触发异常的原始 HTTP 请求。
        exc: 抛出的异常实例。
    """
    # 从请求状态中提取由中间件生成的唯一 request_id，以便于在日志中追踪错误
    request_id = getattr(request.state, "request_id", "unknown")
    
    # 构建标准的错误响应体
    content = {
        "code": exc.code,
        "message": exc.message,
        "request_id": request_id,
        "details": exc.details
    }
    
    # 返回指定状态码和内容的 JSON 响应
    return JSONResponse(status_code=exc.status_code, content=content)
