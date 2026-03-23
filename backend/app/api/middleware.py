import time
import redis.asyncio as redis
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    基于 Redis 的全局限流中间件。
    用于防止接口被恶意刷量，保护后端服务。
    """
    def __init__(self, app):
        super().__init__(app)
        # 初始化异步 Redis 客户端
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        # 限制条件：在 window 秒内，最多允许 limit 次请求
        self.limit = 30
        self.window = 60

    async def dispatch(self, request: Request, call_next):
        """
        处理每一个进入的 HTTP 请求，执行限流逻辑。
        """
        # 放行健康检查接口，不进行限流
        if request.url.path == "/health":
            return await call_next(request)
            
        # 获取客户端 IP 作为限流的标识键
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"
        
        # 获取当前 IP 在时间窗口内的请求次数
        current = await self.redis_client.get(key)
        
        # 如果超过限制，返回 429 Too Many Requests 错误
        if current is not None and int(current) >= self.limit:
            ttl = await self.redis_client.ttl(key) # 获取剩余等待时间
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests"},
                headers={"Retry-After": str(ttl)} # 告知客户端多久后可重试
            )
            
        # 记录本次请求：使用 pipeline 批量执行以提升性能
        pipe = self.redis_client.pipeline()
        await pipe.incr(key, 1) # 请求次数 +1
        await pipe.expire(key, self.window) # 设置/刷新过期时间
        await pipe.execute()
        
        # 继续执行后续的请求处理
        response = await call_next(request)
        return response
