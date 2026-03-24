"""
面试题 A3: 在 Agent 的设计中，"规划能力"至关重要。
请谈谈目前有哪些主流方法可以赋予 LLM 规划能力？

主流规划方法：
  1. CoT  (Chain-of-Thought)     - 思维链：一步步推理
  2. Zero-Shot CoT               - 直接加 "Let's think step by step"
  3. ToT  (Tree-of-Thoughts)     - 思维树：多路径探索 + 剪枝
  4. GoT  (Graph-of-Thoughts)    - 思维图：支持思路合并与循环
  5. Plan-and-Execute            - 先规划再执行（LangChain 实现）
  6. ReWOO                       - 一次性规划，并行执行工具
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))

llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0,
    dashscope_api_key=os.environ.get("OPENAI_API_KEY", "")
)

# =============================================================================
# 1. Standard Prompting（基准对比）
# =============================================================================
def standard_prompting(question: str) -> str:
    """普通提问，无思维链引导"""
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# =============================================================================
# 2. CoT - Chain-of-Thought（思维链）
# 核心思想：通过 few-shot 示例引导 LLM 逐步推理
# =============================================================================
def chain_of_thought(question: str) -> str:
    """
    CoT 通过提供推理示例（few-shot），引导模型展示中间推理步骤。
    论文：Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)
    """
    cot_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个擅长逐步推理的助手。请仿照以下示例，展示完整的推理步骤。

示例问题：小明有 5 个苹果，给了小红 2 个，又买了 3 个，他现在有多少个？
推理过程：
步骤1：小明初始有 5 个苹果
步骤2：给了小红 2 个，剩余 5 - 2 = 3 个
步骤3：又买了 3 个，最终 3 + 3 = 6 个
答案：6 个苹果"""),
        ("human", "问题：{question}\n请展示完整推理步骤：")
    ])
    chain = cot_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# =============================================================================
# 3. Zero-Shot CoT
# 核心思想：只需加一句 "让我们一步步思考" 即可激活推理能力
# =============================================================================
def zero_shot_cot(question: str) -> str:
    """
    Zero-Shot CoT 无需示例，只需在问题后加触发语句。
    论文：Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)
    经典触发语："Let's think step by step" / "让我们一步步思考"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}\n\n让我们一步步思考：")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# =============================================================================
# 4. ToT - Tree of Thoughts（思维树）
# 核心思想：生成多个思路分支 → 评估每个分支价值 → 选择最优路径继续
# 论文：Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023)
# =============================================================================
def tree_of_thoughts(question: str, num_branches: int = 3) -> str:
    """
    ToT 实现步骤：
    1. 生成 N 个不同的初始思路（广度优先探索）
    2. 评估每个思路的可行性和前景
    3. 选择最优思路继续深入
    4. 重复直到得出答案
    """
    # Step 1: 生成多个思路分支
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个创造性的问题解决专家。
对于给定问题，请生成 {num_branches} 个完全不同的解题思路。
每个思路用不超过3句话描述，用 [思路1], [思路2], [思路3] 标记。"""),
        ("human", "问题：{question}")
    ])
    branches_text = (generate_prompt | llm | StrOutputParser()).invoke({"question": question})
    print(f"\n生成的思路分支：\n{branches_text}")

    # Step 2: 评估并选择最优思路
    evaluate_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个批判性思维专家。
请评估以下思路，从可行性、效率、完整性三个维度打分(1-10)，并选出最优思路。
最后基于最优思路给出完整的解答。"""),
        ("human", "问题：{question}\n\n待评估思路：\n{branches}\n\n请评估并给出最终答案：")
    ])
    final = (evaluate_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "branches": branches_text
    })
    return final

# =============================================================================
# 5. Plan-and-Execute（规划后执行）
# 核心思想：先用 Planner 生成完整计划，再用 Executor 逐步执行
# 优点：计划可审查，执行可并行
# =============================================================================
def plan_and_execute(task: str) -> dict:
    """
    Plan-and-Execute 模式（LangChain 有对应实现 langchain_experimental.plan_and_execute）
    适合长任务：任务分解清晰，每步可单独验证
    """
    # Planner: 将大任务分解为有序子任务
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个任务规划专家。将用户的任务分解为5个以内的有序、可执行的子任务。
每个子任务用一行描述，格式：步骤N: [具体操作]"""),
        ("human", "任务：{task}")
    ])
    plan_text = (planner_prompt | llm | StrOutputParser()).invoke({"task": task})
    print(f"\n生成的执行计划：\n{plan_text}")

    # Executor: 逐步执行每个子任务
    steps = [line.strip() for line in plan_text.strip().split("\n") if line.strip()]
    results = []
    context = ""
    for step in steps:
        exec_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个任务执行专家。基于已完成的工作，执行当前步骤并给出结果。"),
            ("human", "总任务：{task}\n\n已完成工作：\n{context}\n\n当前步骤：{step}\n\n请执行此步骤：")
        ])
        step_result = (exec_prompt | llm | StrOutputParser()).invoke({
            "task": task, "context": context, "step": step
        })
        results.append({"step": step, "result": step_result})
        context += f"{step}\n结果: {step_result}\n\n"
        print(f"\n✓ {step}\n  → {step_result[:100]}...")

    return {"plan": plan_text, "steps": results}


if __name__ == "__main__":
    question = "一家初创公司有 3 名工程师，每人月薪 25000 元，另有固定运营成本 15000 元/月。" \
               "如果公司想在 6 个月内盈利，最低月收入需要达到多少？还需要额外考虑哪些成本？"

    print("=" * 60)
    print("1. Standard Prompting（无推理引导）")
    print("=" * 60)
    print(standard_prompting(question))

    print("\n" + "=" * 60)
    print("2. Chain-of-Thought（思维链 - few-shot）")
    print("=" * 60)
    print(chain_of_thought(question))

    print("\n" + "=" * 60)
    print("3. Zero-Shot CoT（零样本思维链）")
    print("=" * 60)
    print(zero_shot_cot(question))

    print("\n" + "=" * 60)
    print("4. Tree of Thoughts（思维树 - 多路径探索）")
    print("=" * 60)
    print(tree_of_thoughts(question, num_branches=3))

    print("\n" + "=" * 60)
    print("5. Plan-and-Execute（规划后执行）")
    print("=" * 60)
    plan_and_execute("分析这家初创公司的财务状况，给出6个月盈利的可行方案")

    print("""

    规划方法对比总结：
    ┌────────────────┬────────────────────────────────┬──────────────────┐
    │ 方法           │ 核心思想                        │ 适用场景          │
    ├────────────────┼────────────────────────────────┼──────────────────┤
    │ Standard       │ 直接回答                        │ 简单问题          │
    │ CoT (few-shot) │ 示例引导逐步推理                │ 数学/逻辑推理     │
    │ Zero-Shot CoT  │ "一步步思考" 触发推理            │ 通用推理          │
    │ ToT            │ 多路径探索+评估剪枝              │ 创意/开放问题     │
    │ GoT            │ 支持思路合并和循环引用          │ 复杂网络结构问题  │
    │ Plan-Execute   │ 先整体规划再逐步执行            │ 长任务/多步骤任务 │
    └────────────────┴────────────────────────────────┴──────────────────┘
    """)
