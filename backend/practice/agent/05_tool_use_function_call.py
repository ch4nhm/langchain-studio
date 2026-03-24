"""
面试题 A5: Tool Use 是扩展 Agent 能力的有效途径。
LLM 是如何学会调用外部 API 或工具的？（从 Function Calling 角度解释）

Function Calling 原理：
  1. 开发者将工具描述为 JSON Schema（函数名、参数、说明）
  2. LLM 在训练时学会了识别何时需要调用工具
  3. LLM 输出结构化 JSON（工具名 + 参数），而非自然语言
  4. 代码层解析 JSON → 调用真实函数 → 将结果返回给 LLM
  5. LLM 基于工具结果生成最终自然语言回答

工具定义方式（LangChain）：
  - @tool 装饰器
  - Tool.from_function()
  - StructuredTool（支持复杂参数）
  - 继承 BaseTool（最灵活）
"""

import os
import json
import requests
from typing import Optional, Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

# =============================================================================
# 工具定义方式 1: @tool 装饰器（最简单）
# =============================================================================

@tool
def get_stock_price(ticker: str) -> str:
    """获取股票的当前价格。ticker 是股票代码，例如 'AAPL'、'BABA'。"""
    # 模拟股票价格
    prices = {"AAPL": 189.5, "BABA": 78.3, "TSLA": 245.1, "MSFT": 415.2}
    price = prices.get(ticker.upper())
    if price:
        return json.dumps({"ticker": ticker, "price": price, "currency": "USD"})
    return json.dumps({"error": f"未找到 {ticker} 的价格"})

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件。to 是收件人邮箱，subject 是主题，body 是邮件正文。"""
    # 实际场景调用 SMTP 或邮件 API
    print(f"  [模拟发送邮件] 收件人: {to}, 主题: {subject}")
    return json.dumps({"status": "success", "message": f"邮件已发送至 {to}"})

# =============================================================================
# 工具定义方式 2: StructuredTool（支持复杂嵌套参数）
# =============================================================================

class DatabaseQueryInput(BaseModel):  # 用 Pydantic 描述参数结构
    table: str = Field(description="要查询的数据库表名")
    conditions: Optional[str] = Field(default=None, description="WHERE 条件，例如 'age > 18'")
    limit: int = Field(default=10, description="返回记录数上限", ge=1, le=100)

def query_database(table: str, conditions: Optional[str] = None, limit: int = 10) -> str:
    """执行数据库查询（模拟）"""
    mock_data = {
        "users": [{"id": i, "name": f"用户{i}", "age": 20 + i} for i in range(1, limit + 1)],
        "orders": [{"id": i, "amount": i * 100, "status": "completed"} for i in range(1, limit + 1)],
    }
    data = mock_data.get(table, [])
    return json.dumps({"table": table, "count": len(data), "data": data[:3], "note": "仅展示前3条"})

database_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="查询数据库中的数据，支持指定表名、过滤条件和数量限制",
    args_schema=DatabaseQueryInput,
)

# =============================================================================
# 工具定义方式 3: 继承 BaseTool（最灵活，适合异步/复杂逻辑）
# =============================================================================

class WebScraperInput(BaseModel):
    url: str = Field(description="要抓取内容的网页 URL")
    selector: Optional[str] = Field(default=None, description="CSS 选择器，用于提取特定元素")

class WebScraperTool(BaseTool):
    name: str = "web_scraper"
    description: str = "抓取指定网页的文本内容，可以通过 CSS 选择器精确提取。"
    args_schema: Type[BaseModel] = WebScraperInput

    def _run(self, url: str, selector: Optional[str] = None) -> str:
        """同步执行"""
        # 实际场景用 requests + BeautifulSoup
        print(f"  [模拟抓取] URL: {url}, 选择器: {selector}")
        return json.dumps({
            "url": url,
            "content": "这是模拟抓取到的网页内容：LangChain 0.3 发布，新增多项功能...",
            "word_count": 128
        })

    async def _arun(self, url: str, selector: Optional[str] = None) -> str:
        """异步执行（生产环境推荐）"""
        return self._run(url, selector)

# =============================================================================
# 展示 Function Calling 的底层 JSON Schema
# =============================================================================
def show_tool_schema():
    """展示工具被转换为 JSON Schema 的样子（这就是发送给 LLM 的工具描述）"""
    print("\n【Function Calling 底层：工具被转换为 JSON Schema】")
    for t in [get_stock_price, send_email, database_tool]:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        print(f"\n工具名: {t.name}")
        print(f"描述:   {t.description}")
        print(f"参数 Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")

# =============================================================================
# 组装 Agent 并运行
# =============================================================================
def run_tool_agent():
    llm = ChatTongyi(
        model="qwen-turbo",
        temperature=0,
    )
    all_tools = [get_stock_price, send_email, database_tool, WebScraperTool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个拥有多种工具的智能助手。根据用户需求选择合适的工具完成任务。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=all_tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True)

    print("\n【Agent 运行：多工具组合调用】")
    result = executor.invoke({
        "input": "帮我查一下 AAPL 和 TSLA 的股价，然后查询数据库 users 表的前5条记录"
    })
    print(f"\n最终答案: {result['output']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Tool Use & Function Calling 演示")
    print("=" * 60)
    show_tool_schema()
    run_tool_agent()

    print("""
    Function Calling 工作原理总结：
    ┌──────────────────────────────────────────────────────────┐
    │  1. 注册阶段                                             │
    │     开发者将工具描述为 JSON Schema                       │
    │     发送给 LLM 作为系统上下文                            │
    │                                                          │
    │  2. 决策阶段                                             │
    │     LLM 分析用户意图，决定是否调用工具                   │
    │     输出结构化 JSON: {"tool": "xxx", "args": {...}}      │
    │                                                          │
    │  3. 执行阶段                                             │
    │     框架解析 JSON → 调用真实函数 → 获取结果              │
    │                                                          │
    │  4. 整合阶段                                             │
    │     将工具结果注入上下文 → LLM 生成自然语言回答          │
    └──────────────────────────────────────────────────────────┘
    """)
