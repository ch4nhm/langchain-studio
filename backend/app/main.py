import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router
from app.api.middleware import RateLimitMiddleware
from app.core.exceptions import LangChainException, langchain_exception_handler
from app.core.config import settings

# 初始化 FastAPI 应用程序实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="LangChain Demo Project API",
    version="0.1.0"
)

# 添加 CORS 中间件，允许跨域请求（前端调用所需）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 在生产环境中应替换为具体的域名
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法 (GET, POST 等)
    allow_headers=["*"], # 允许所有请求头
)

# 添加自定义的基于 Redis 的限流中间件
app.add_middleware(RateLimitMiddleware)

# 添加请求头追踪中间件，为每个请求生成并附加唯一的 Request-ID
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4()) # 生成 UUID 作为请求 ID
    request.state.request_id = request_id # 将请求 ID 存入 request 状态中，供后续处理使用
    response = await call_next(request) # 将请求传递给下一个中间件或路由处理器
    response.headers["X-Request-ID"] = request_id # 将请求 ID 添加到响应头中，方便客户端和日志追踪
    return response

# 注册全局异常处理器，捕获并格式化自定义的 LangChainException
app.add_exception_handler(LangChainException, langchain_exception_handler)

# 包含应用的主路由
app.include_router(router)

# 本地开发启动入口
if __name__ == "__main__":
    import uvicorn
    # 使用 Uvicorn 运行 ASGI 应用，开启热重载以便开发
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
