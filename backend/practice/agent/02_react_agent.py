"""
面试题 A2: 请详细解释 ReAct 框架。
它是如何将思维链（CoT）和行动（Action）结合起来以完成复杂任务的？

ReAct = Reasoning + Acting
论文：ReAct: Synergizing Reasoning and Acting in Language Models (2022)

核心循环：
  Thought  → 推理当前状态，决定下一步做什么
  Action   → 执行一个具体工具调用
  Observation → 获取工具返回的结果
  Thought  → 基于观察继续推理...
  ...
  Final Answer → 得出最终答案
"""

import os
import re
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

# =============================================================================
# 工具定义
# =============================================================================

@tool
def search(query: str) -> str:
    """搜索互联网获取信息。"""
    # 模拟搜索结果
    db = {
        "langchain 版本": "LangChain 最新版本为 0.3.x，于2024年发布。",
        "python 版本": "Python 最新稳定版为 3.12，发布于2023年10月。",
        "openai gpt-4": "GPT-4 是 OpenAI 于2023年3月发布的多模态大语言模型。",
        "通义千问": "通义千问（Qwen）是阿里巴巴开发的大语言模型系列，包括 qwen-turbo/plus/max 等版本。",
    }
    for key, val in db.items():
        if key in query.lower():
            return val
    return f"搜索 '{query}' 未找到精确结果，请尝试其他关键词。"

@tool
def calculator(expression: str) -> str:
    """执行数学计算。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误: {e}"

@tool
def lookup_table(key: str) -> str:
    """查询结构化数据表（模拟数据库查询）。"""
    data = {
        "embedding_models": "常用 Embedding 模型: text-embedding-ada-002(OpenAI), text-embedding-v1(通义千问), bge-large-zh(BAAI)",
        "vector_databases": "主流向量数据库: ChromaDB, Pinecone, Weaviate, Milvus, pgvector",
        "llm_frameworks": "主流 LLM 框架: LangChain, LlamaIndex, Haystack, Semantic Kernel",
    }
    return data.get(key.lower(), f"未找到键 '{key}' 的数据")

TOOLS = {"search": search, "calculator": calculator, "lookup_table": lookup_table}

# =============================================================================
# 手动实现 ReAct 循环（帮助理解底层原理）
# =============================================================================

REACT_PROMPT = """你是一个能够使用工具的智能助手。请使用以下格式进行推理和行动：

可用工具：
- search(query): 搜索互联网获取信息
- calculator(expression): 执行数学计算
- lookup_table(key): 查询数据表，可用的 key: embedding_models, vector_databases, llm_frameworks

格式要求（严格遵守）：
Thought: [你的推理过程]
Action: tool_name("input")
Observation: [工具返回结果 - 由系统填写]
... (可以重复 Thought/Action/Observation)
Final Answer: [最终答案]

问题: {question}

开始推理：
Thought:"""

def parse_action(text: str):
    """解析 LLM 输出中的 Action 调用"""
    match = re.search(r'Action:\s*(\w+)\("([^"]*)"\)', text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_react_manually(question: str, max_steps: int = 6):
    """
    手动实现 ReAct 循环，逐步展示每个 Thought/Action/Observation。
    这是理解 Agent 工作原理最直观的方式。
    """
    llm = ChatTongyi(
        model="qwen-turbo",
        temperature=0,
        dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
    )

    print(f"问题: {question}")
    print("-" * 50)

    # 初始 prompt
    current_prompt = REACT_PROMPT.format(question=question)
    full_trace = ""

    for step in range(max_steps):
        # LLM 推理
        response = llm.invoke(current_prompt + full_trace)
        output = response.content

        print(f"\n--- Step {step + 1} ---")
        print(output)

        # 检查是否已有最终答案
        if "Final Answer:" in output:
            final = output.split("Final Answer:")[-1].strip()
            print(f"\n{'='*50}")
            print(f"最终答案: {final}")
            return final

        # 解析并执行 Action
        tool_name, tool_input = parse_action(output)
        if tool_name and tool_name in TOOLS:
            observation = TOOLS[tool_name].invoke(tool_input)
            print(f"Observation: {observation}")
            # 将这轮的 Thought+Action+Observation 追加到上下文
            full_trace += output + f"\nObservation: {observation}\nThought:"
        else:
            print("未找到有效 Action，终止循环。")
            break

    return "达到最大迭代次数"


# =============================================================================
# 使用 LangChain 内置 ReAct Agent（生产方式）
# =============================================================================

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
请用中文回答。

Question: {input}
Thought:{agent_scratchpad}"""

def run_react_langchain(question: str):
    """使用 LangChain 封装的 ReAct Agent"""
    llm = ChatTongyi(
        model="qwen-turbo",
        temperature=0,
        dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
    )
    tools_list = [search, calculator, lookup_table]
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=tools_list, prompt=prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools_list,
        verbose=True, max_iterations=5, handle_parsing_errors=True
    )
    result = executor.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    print("=" * 60)
    print("ReAct 框架原理演示")
    print("=" * 60)

    question = "主流的向量数据库有哪些？同时计算 1024 * 1024 的值。"

    print("\n【方式 1：手动实现 ReAct 循环 - 理解底层原理】")
    run_react_manually(question)

    print("\n\n【方式 2：LangChain 封装的 ReAct Agent - 生产使用】")
    result = run_react_langchain(question)
    print(f"最终答案: {result}")

    print("""
    \nReAct 框架知识点总结：
    ┌──────────────────────────────────────────────────────────┐
    │  ReAct = Reasoning(思维链 CoT) + Acting(工具调用)        │
    │                                                          │
    │  优势：                                                  │
    │  1. 可解释性强 - 每步 Thought 记录推理过程               │
    │  2. 错误可纠正 - Observation 反馈后可修正方向            │
    │  3. 工具灵活   - 支持任意外部工具扩展                    │
    │                                                          │
    │  与纯 CoT 的区别：                                       │
    │  - CoT: 只推理，不行动，无法获取外部信息                 │
    │  - ReAct: 推理 + 行动，可与外部世界交互                  │
    └──────────────────────────────────────────────────────────┘
    """)
