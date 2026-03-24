"""
面试题 R5: RAG 系统评估指标
检索阶段: Recall@K, Precision@K, MRR, NDCG
生成阶段: Faithfulness, Answer Relevancy, LLM-as-Judge
"""
import os
import math
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../backend/.env"))
API_KEY = os.environ.get("OPENAI_API_KEY", "")


# =============================================================================
# 检索阶段评估指标
# =============================================================================

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Recall@K: 前K结果中找到的相关文档比例"""
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Precision@K: 前K结果中相关文档的精确率"""
    return len(set(retrieved[:k]) & relevant) / k if k > 0 else 0.0

def mrr(retrieved: list, relevant: set) -> float:
    """MRR: 第一个相关文档排名的倒数"""
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """NDCG@K: 归一化折损累积增益（考虑排名位置权重）"""
    def dcg(ids):
        return sum(1/math.log2(i+2) for i, d in enumerate(ids[:k]) if d in relevant)
    ideal = list(relevant)[:k] + [None]*k
    ideal_dcg = dcg(ideal)
    return dcg(retrieved) / ideal_dcg if ideal_dcg > 0 else 0.0


def eval_retrieval():
    print("\n【检索阶段评估指标对比】")
    relevant = {"doc_1", "doc_3", "doc_7"}  # 真实相关文档
    # 检索器A：相关文档排名靠前
    ret_a = ["doc_1", "doc_2", "doc_3", "doc_5", "doc_7"]
    # 检索器B：相关文档排名靠后
    ret_b = ["doc_2", "doc_5", "doc_6", "doc_1", "doc_3"]

    print(f"  相关文档: {relevant}")
    for name, ret in [("检索器A(相关文档靠前)", ret_a), ("检索器B(相关文档靠后)", ret_b)]:
        print(f"\n  {name}: {ret}")
        print(f"    Recall@5   = {recall_at_k(ret, relevant, 5):.3f}  (找到了多少相关文档)")
        print(f"    Precision@5= {precision_at_k(ret, relevant, 5):.3f}  (结果中相关文档占比)")
        print(f"    MRR        = {mrr(ret, relevant):.3f}  (第一个相关文档排多靠前)")
        print(f"    NDCG@5     = {ndcg_at_k(ret, relevant, 5):.3f}  (综合排名质量)")


# =============================================================================
# 生成阶段评估：LLM-as-Judge
# =============================================================================

def judge_faithfulness(question: str, context: str, answer: str) -> str:
    """
    忠实度 (Faithfulness): 答案中每个声明是否都有上下文支撑。
    高忠实度 = 低幻觉率
    """
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是RAG系统评估专家。请判断答案中的声明是否都有上下文支撑，给出0-10分和原因。"
         "10=完全基于上下文，0=完全凭空捏造。"),
        ("human",
         "问题: {q}\n上下文: {ctx}\n答案: {ans}\n\n请给出评分和原因:")
    ])
    return (prompt | llm | StrOutputParser()).invoke({"q": question, "ctx": context, "ans": answer})


def judge_relevancy(question: str, answer: str) -> str:
    """
    答案相关性 (Answer Relevancy): 答案是否真正回答了问题。
    """
    llm = ChatTongyi(model="qwen-turbo", temperature=0, dashscope_api_key=API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "判断答案是否直接回答了问题，给出0-10分和原因。10=完整回答，0=完全无关。"),
        ("human", "问题: {q}\n答案: {ans}\n\n请给出评分和原因:")
    ])
    return (prompt | llm | StrOutputParser()).invoke({"q": question, "ans": answer})


def eval_generation():
    print("\n【生成阶段评估：LLM-as-Judge】")
    context = "LangChain 最新版本为 0.3.x，于2024年发布。它是一个用于构建LLM应用的开源框架。"
    question = "LangChain 最新版本是多少？"

    test_cases = [
        ("好答案",   "LangChain 最新版本是 0.3.x，于2024年发布。"),
        ("幻觉答案",  "LangChain 最新版本是 1.5.0，于2025年发布，新增了量子计算支持。"),
        ("无关答案",  "天气很好，适合出门散步。"),
    ]
    for label, answer in test_cases:
        print(f"\n  [{label}] 答案: {answer}")
        faith = judge_faithfulness(question, context, answer)
        relev = judge_relevancy(question, answer)
        print(f"  忠实度评分: {faith[:80]}")
        print(f"  相关性评分: {relev[:80]}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG 系统评估演示")
    print("=" * 60)
    eval_retrieval()
    eval_generation()
    print("""
    RAG 评估体系总结：
    ┌──────────────────┬──────────────────────────────────────┐
    │ 阶段              │ 指标                                  │
    ├──────────────────┼──────────────────────────────────────┤
    │ 检索阶段          │ Recall@K, Precision@K, MRR, NDCG     │
    │ 生成阶段          │ Faithfulness, Answer Relevancy        │
    │ 端到端（有答案）  │ BLEU, ROUGE, Exact Match             │
    │ 端到端（无答案）  │ LLM-as-Judge（GPT/Claude评分）       │
    │ 框架工具          │ RAGAS, TruLens, DeepEval             │
    └──────────────────┴──────────────────────────────────────┘
    """)
