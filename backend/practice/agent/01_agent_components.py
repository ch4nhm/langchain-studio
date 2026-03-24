"""
面试题 A1: 如何定义一个基于 LLM 的智能体（Agent）？它通常由哪些核心组件构成？

核心组件：
  1. LLM（大脑）       - 负责理解、推理和决策
  2. Tools（工具）     - 扩展 Agent 能力（搜索、计算、API调用）
  3. Memory（记忆）    - 短期（对话历史）+ 长期（向量库）
  4. Planning（规划）  - 将复杂目标分解为可执行步骤
  5. Action（行动）    - 执行工具调用并观察结果
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

# =============================================================================
# 组件 1: Tools（工具） - Agent 的行动能力
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2 + 3 * 4'。输入必须是合法的 Python 数学表达式。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息（模拟）。"""
    # 实际场景中这里会调用真实天气 API
    weather_data = {
        "北京": "晴天，25°C，北风3级",
        "上海": "多云，28°C，东风2级",
        "广州": "小雨，30°C，南风1级",
    }
    return weather_data.get(city, f"{city} 的天气数据暂时不可用")

@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索与查询相关的信息。"""
    # 实际场景中这里会调用向量数据库
    knowledge = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架，提供 Chain、Agent、Memory 等抽象。",
        "agent": "Agent 是一种能够感知环境、自主规划并采取行动以达成目标的 AI 系统。",
        "rag": "RAG（检索增强生成）通过在生成前检索相关文档来提升 LLM 的准确性。",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "未找到相关信息"

tools = [calculator, get_weather, search_knowledge_base]

# =============================================================================
# 组件 2: LLM（大脑）
# =============================================================================
llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0,
    dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
)

# =============================================================================
# 组件 3: Prompt（指令/角色定义）
# =============================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一个智能助手，拥有以下工具：
    - calculator: 数学计算
    - get_weather: 查询天气
    - search_knowledge_base: 查询知识库

    请根据用户的问题，合理选择工具并给出最终答案。
    思考过程要清晰，先分析需要什么信息，再决定调用哪个工具。
    """),
    MessagesPlaceholder(variable_name="chat_history"),  # 组件 4: Memory
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 组件 5: Planning/Action
])

# =============================================================================
# 组件 4: Memory（短期记忆 - 对话历史）
# =============================================================================
chat_history = []  # 实际项目中会用 PostgresChatMessageHistory 持久化

# =============================================================================
# 组装：创建 Agent
# =============================================================================
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # 打印 Agent 的思考过程
    max_iterations=5,  # 最大迭代次数防止死循环
    handle_parsing_errors=True,
)

# =============================================================================
# 运行示例
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("示例 1: 工具调用 - 数学计算")
    print("=" * 60)
    result = agent_executor.invoke({
        "input": "帮我计算 (123 + 456) * 2 - 100 的结果",
        "chat_history": chat_history
    })
    print(f"\n最终答案: {result['output']}")

    # 更新历史（模拟短期记忆）
    chat_history.extend([
        HumanMessage(content="帮我计算 (123 + 456) * 2 - 100 的结果"),
        AIMessage(content=result['output'])
    ])

    print("\n" + "=" * 60)
    print("示例 2: 多工具组合 - 天气 + 知识库")
    print("=" * 60)
    result2 = agent_executor.invoke({
        "input": "北京今天天气怎么样？顺便告诉我什么是 RAG。",
        "chat_history": chat_history
    })
    print(f"\n最终答案: {result2['output']}")

    print("\n" + "=" * 60)
    print("知识点总结")
    print("=" * 60)
    print("""
    Agent 的 5 大核心组件：
    ┌─────────────────────────────────────────────────────┐
    │  1. LLM (大脑)     - 理解意图、推理、生成决策       │
    │  2. Tools (工具)   - 赋予 Agent 与外界交互的能力   │
    │  3. Memory (记忆)  - 保持上下文和历史信息          │
    │  4. Planning (规划)- 分解目标为可执行步骤          │
    │  5. Action (行动)  - 执行工具调用，获取 Observation│
    └─────────────────────────────────────────────────────┘

    与普通 LLM 调用的区别：
    - 普通 LLM: 输入 → 输出 (单轮，被动)
    - Agent:    目标 → [规划→行动→观察]循环 → 达成目标 (多轮，主动)
    """)
