import time
import redis.asyncio as redis
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.limit = 30
        self.window = 60

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"
        
        current = await self.redis_client.get(key)
        if current is not None and int(current) >= self.limit:
            ttl = await self.redis_client.ttl(key)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests"},
                headers={"Retry-After": str(ttl)}
            )
            
        pipe = self.redis_client.pipeline()
        pipe.incr(key, 1)
        pipe.expire(key, self.window)
        await pipe.execute()
        
        response = await call_next(request)
        return response
