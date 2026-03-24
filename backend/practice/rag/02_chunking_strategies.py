"""
面试题 R2: 文本切块策略对比
"""
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

SAMPLE_TEXT = """
LangChain 是一个用于构建大语言模型（LLM）应用程序的开源框架。
它由 Harrison Chase 于 2022 年 10 月创立，迅速成为 AI 应用开发领域最受欢迎的框架之一。

LangChain 的核心价值在于其模块化的组件设计，主要包括以下几个模块：
第一，模型 I/O：统一接口封装，支持 OpenAI、Anthropic、通义千问等主流 LLM。
第二，检索（Retrieval）：RAG 相关组件，包括文档加载器、文本切分器、向量存储和检索器。
第三，链（Chain）：将多个组件按逻辑串联，支持顺序链、路由链等多种模式。
第四，Agent：能自主规划和使用工具完成复杂任务的智能体框架。
第五，记忆（Memory）：管理对话历史，支持多种存储后端。

LCEL 是 LangChain 0.1 后推出的新式链构建语法，使用管道符 | 将 Runnable 组件串联。
每个组件都有统一的 invoke/stream/batch 接口，支持流式输出、异步调用和并行执行。
"""


def demo_fixed_size():
    print("\n【策略1: CharacterTextSplitter - 固定字符数】")
    splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator="\n")
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"  chunk_size=150, overlap=20 -> {len(chunks)} 块")
    for i, c in enumerate(chunks[:2], 1):
        print(f"  chunk{i}({len(c)}字): {c[:60]}...")


def demo_recursive():
    print("\n【策略2: RecursiveCharacterTextSplitter - 递归分隔符（最常用）】")
    for size, overlap, label in [
        (100, 20,  "小块-精确检索"),
        (300, 50,  "中块-平衡"),
        (600, 100, "大块-完整上下文"),
    ]:
        sp = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = sp.split_text(SAMPLE_TEXT)
        print(f"  {label}: size={size}, overlap={overlap} -> {len(chunks)} 块")
    print("\n  选择原则：")
    print("  - 太小: 语义不完整，检索噪音多")
    print("  - 太大: 包含多主题，检索不精准")
    print("  - overlap: 建议 chunk_size 的 10-20%，防止边界信息丢失")


def demo_sentence():
    print("\n【策略3: 按句子边界切块 - 语义完整性优先】")
    sentences = [s.strip() for s in SAMPLE_TEXT.replace("\n", "").split("。") if s.strip()]
    window = 2
    chunks = ["。".join(sentences[i:i+window]) + "。" for i in range(len(sentences)-window+1)]
    print(f"  句子数={len(sentences)}, 窗口={window} -> {len(chunks)} 块")
    print(f"  示例: '{chunks[0][:80]}...'")


def demo_parent_document():
    print("\n【策略4: Parent Document - 小块检索+大块上下文】")
    child_sp = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    parent_sp = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    child_chunks = child_sp.split_text(SAMPLE_TEXT)
    parent_chunks = parent_sp.split_text(SAMPLE_TEXT)
    print(f"  子chunk(检索用): {len(child_chunks)} 块，平均 {sum(len(c) for c in child_chunks)//len(child_chunks)} 字")
    print(f"  父chunk(返回LLM): {len(parent_chunks)} 块，平均 {sum(len(c) for c in parent_chunks)//len(parent_chunks)} 字")
    print("  优势: 检索精准度高(小块) + LLM获得完整上下文(大块)")
    print("  LangChain 实现: ParentDocumentRetriever")


def print_strategy_guide():
    print("""
    切块策略选型指南：
    ┌──────────────────┬────────────┬────────────────────────────────┐
    │ 策略              │ 推荐场景   │ 优缺点                          │
    ├──────────────────┼────────────┼────────────────────────────────┤
    │ FixedSize        │ 快速原型   │ 简单快速；可能切断语义          │
    │ Recursive(推荐)  │ 通用场景   │ 尊重自然边界；参数需调优        │
    │ Sentence         │ 问答/摘要  │ 语义完整；块大小不均匀          │
    │ Semantic         │ 高精度场景 │ 最精准；计算成本高              │
    │ ParentDocument   │ 长文档问答 │ 精准+完整；实现稍复杂           │
    └──────────────────┴────────────┴────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("文本切块策略对比演示")
    print("=" * 60)
    demo_fixed_size()
    demo_recursive()
    demo_sentence()
    demo_parent_document()
    print_strategy_guide()
