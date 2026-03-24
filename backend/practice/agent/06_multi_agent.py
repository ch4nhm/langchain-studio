"""
面试题 A6: 什么是多智能体系统？
让多个 LLM Agent 协同工作相比于单个 Agent 有什么优势？又会引入哪些新的复杂性？

多智能体模式：
  1. Sequential（顺序链）   - A -> B -> C，流水线任务
  2. Hierarchical（层级制） - Supervisor 分派任务给 Worker
  3. Debate（辩论式）       - 多 Agent 观点交锋，裁判汇总
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))


def get_llm(temperature=0.0):
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "sk-dummy"
    return ChatTongyi(
        model="qwen-turbo",
        temperature=temperature,
        dashscope_api_key=api_key
    )


# =============================================================================
# 模式 1: Sequential 顺序链 - 研究员 -> 写作者 -> 编辑
# =============================================================================

class ResearchAgent:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是严谨的研究员，负责收集准确信息和数据。"
                       "输出结构化研究笔记：核心概念、关键数据、重要背景。"),
            ("human", "请研究以下主题：{topic}")
        ])
        self.chain = prompt | get_llm(0) | StrOutputParser()

    def research(self, topic: str) -> str:
        print("  [研究员 Agent] 正在收集信息...")
        return self.chain.invoke({"topic": topic})


class WriterAgent:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是优秀的科技写作者，擅长将复杂技术用通俗语言表达。"
                       "请基于研究材料撰写一篇结构清晰的文章。"),
            ("human", "主题：{topic}\n\n研究材料：\n{research}\n\n请撰写500字文章：")
        ])
        self.chain = prompt | get_llm(0.7) | StrOutputParser()

    def write(self, topic: str, research: str) -> str:
        print("  [写作者 Agent] 正在撰写文章...")
        return self.chain.invoke({"topic": topic, "research": research})


class EditorAgent:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是资深编辑，负责审查文章准确性、逻辑性和可读性。"
                       "请指出优点和改进点，给出修改后的最终版本。"),
            ("human", "请审查并改进以下文章：\n\n{article}")
        ])
        self.chain = prompt | get_llm(0) | StrOutputParser()

    def edit(self, article: str) -> str:
        print("  [编辑 Agent] 正在审查文章...")
        return self.chain.invoke({"article": article})


def run_sequential_pipeline(topic: str):
    print(f"\n【顺序链多智能体：内容创作流水线】")
    print(f"主题: {topic}\n" + "-" * 50)

    notes = ResearchAgent().research(topic)
    print(f"  研究笔记: {len(notes)} 字")

    draft = WriterAgent().write(topic, notes)
    print(f"  初稿: {len(draft)} 字")

    final = EditorAgent().edit(draft)
    print(f"  终稿: {len(final)} 字")
    print(f"\n终稿预览:\n{final[:300]}...")
    return final


# =============================================================================
# 模式 2: Hierarchical 层级制 - Supervisor 分派任务
# =============================================================================

def run_supervisor(task: str):
    workers = {
        "研究员": "负责信息收集和数据分析",
        "代码工程师": "负责编写和审查代码",
        "文档写作者": "负责撰写文档和报告",
    }
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是任务调度主管。你的团队：\n"
                   + "\n".join(f"- {k}: {v}" for k, v in workers.items())
                   + "\n\n分析任务，决定调用哪些专家，说明各自的子任务。"
                     "格式：\n专家: [名称]\n子任务: [描述]\n---"),
        ("human", "用户任务：{task}")
    ])
    plan = (prompt | get_llm(0) | StrOutputParser()).invoke({"task": task})
    print(f"\n【层级制多智能体：主管分派任务】")
    print(f"任务: {task}\n主管分派方案:\n{plan}")
    return plan


# =============================================================================
# 模式 3: Debate 辩论式 - 观点交锋 + 裁判汇总
# =============================================================================

def run_debate(question: str, rounds: int = 1):
    print(f"\n【辩论式多智能体：观点交锋提升质量】")
    print(f"问题: {question}")

    llm = get_llm(0.5)
    pro_chain = ChatPromptTemplate.from_messages([
        ("system", "你持支持立场，提供有力论据并反驳对方。"),
        ("human", "问题：{q}\n对方观点：{opp}\n请发表支持论点（200字内）：")
    ]) | llm | StrOutputParser()

    con_chain = ChatPromptTemplate.from_messages([
        ("system", "你持反对立场，提出质疑并找出漏洞。"),
        ("human", "问题：{q}\n对方观点：{opp}\n请发表反对意见（200字内）：")
    ]) | llm | StrOutputParser()

    judge_chain = ChatPromptTemplate.from_messages([
        ("system", "你是公正裁判，综合双方论点给出平衡结论。"),
        ("human", "问题：{q}\n支持方：{pro}\n反对方：{con}\n请给出综合结论：")
    ]) | llm | StrOutputParser()

    pro_arg, con_arg = "", ""
    for r in range(rounds):
        print(f"  第 {r+1} 轮辩论...")
        pro_arg = pro_chain.invoke({"q": question, "opp": con_arg})
        con_arg = con_chain.invoke({"q": question, "opp": pro_arg})

    conclusion = judge_chain.invoke({"q": question, "pro": pro_arg, "con": con_arg})
    print(f"\n  裁判结论: {conclusion[:300]}...")
    return conclusion


if __name__ == "__main__":
    print("=" * 60)
    print("多智能体系统演示")
    print("=" * 60)

    run_sequential_pipeline("大语言模型在医疗领域的应用与挑战")
    run_supervisor("为 RAG 系统写技术报告，包含架构分析和示例代码")
    run_debate("LLM Agent 是否会在5年内取代大多数软件工程师的工作？")

    print("""
    多智能体 vs 单智能体对比：
    ┌──────────────┬────────────────┬──────────────────────┐
    │ 维度          │ 单 Agent       │ 多 Agent             │
    ├──────────────┼────────────────┼──────────────────────┤
    │ 专业性        │ 通才           │ 专家分工             │
    │ 处理复杂度    │ 受上下文限制   │ 分解后并行处理       │
    │ 质量校验      │ 无自我纠错     │ 相互检查/辩论        │
    │ 成本          │ 低             │ 高（多次LLM调用）    │
    │ 调试难度      │ 简单           │ 复杂（通信追踪）     │
    │ 适用场景      │ 简单单一任务   │ 复杂多步骤任务       │
    └──────────────┴────────────────┴──────────────────────┘
    """)
