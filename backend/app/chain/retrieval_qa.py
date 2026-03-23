from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.core.llm_factory import get_llm
from app.retrieval.vector_store import get_vector_store
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

class QAInput(BaseModel):
    """Retrieval QA 链的输入模型。"""
    query: str = Field(..., description="用户的查询字符串。")
    session_id: str = Field(..., description="唯一的会话标识符。")

class SourceDocument(BaseModel):
    """表示源文档的 Pydantic 模型。"""
    content: str
    metadata: Dict[str, Any]

class QAOutput(BaseModel):
    """Retrieval QA 链的输出模型。"""
    answer: str = Field(..., description="生成的回答。")
    source_documents: List[SourceDocument] = Field(..., description="用于提供上下文的源文档。")

def format_docs(docs: List[Any]) -> str:
    """将检索到的多个文档片段格式化为单个字符串，以便嵌入到 Prompt 的上下文中。"""
    return "\n\n".join(f"<doc id={i}>\n{doc.page_content}\n</doc>" for i, doc in enumerate(docs))

def get_retrieval_qa_chain() -> Any:
    """
    构建并返回基于 LCEL (LangChain Expression Language) 的 Retrieval QA 链。
    该链实现了完整的 RAG (Retrieval-Augmented Generation) 流程，包含：
    1. 根据聊天历史重写独立问题。
    2. 使用独立问题去向量库检索相关文档。
    3. 将检索到的文档和问题一起输入给 LLM 生成最终答案。
    
    Returns:
        Runnable: 配置好的 LCEL 链，可直接调用 invoke 或 astream 方法。
    """
    # 获取 LLM 和检索器，temperature=0 保证回答的客观性和准确性
    llm = get_llm(temperature=0)
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 5}) # 每次检索最相关的 5 个片段

    # ==========================================
    # 步骤 1: 独立问题重写 (Condense Question)
    # 解决多轮对话中的指代消解问题（如用户问“它的特点是什么”，LLM 需要知道“它”指代前文的什么内容）
    # ==========================================
    condense_question_system_template = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能会引用聊天历史记录中的上下文，"
        "请重写为一个可以脱离聊天历史记录独立理解的问题。"
        "不要回答问题，如果需要则重写，否则原样返回。"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        MessagesPlaceholder(variable_name="chat_history"), # 动态插入历史消息
        ("human", "{question}")
    ])
    
    # 构建独立问题生成器 Runnable: Prompt -> LLM -> 字符串输出
    standalone_question = condense_question_prompt | llm | StrOutputParser()
    
    # ==========================================
    # 步骤 2: 答案生成 Prompt (QA Prompt)
    # ==========================================
    qa_system_template = (
        "你是一个乐于助人的助手。请使用以下检索到的上下文片段来回答问题。"
        "如果你不知道答案，就说你不知道。最多使用三句话，并保持回答简洁。\n"
        "上下文：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_template),
        ("human", "{question}")
    ])
    
    # ==========================================
    # 步骤 3: 组装核心的检索与回答链 (Retrieval Chain)
    # ==========================================
    def get_context(inputs: dict) -> List[Any]:
        """基于独立问题执行向量库检索"""
        docs = retriever.invoke(inputs["question"])
        return docs
        
    def _format_docs_for_prompt(docs: List[Any]) -> str:
        """格式化文档为文本"""
        return format_docs(docs)

    # 构建检索链：先获取上下文 -> 格式化上下文 -> 生成答案
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: get_context(x))
        )
        | RunnablePassthrough.assign(
            formatted_context=(lambda x: _format_docs_for_prompt(x["context"]))
        )
        | RunnablePassthrough.assign(
            answer=(lambda x: (qa_prompt | llm | StrOutputParser()).invoke({
                "context": x["formatted_context"],
                "question": x["question"]
            }))
        )
    )
    
    # ==========================================
    # 步骤 4: 组装完整的端到端链路 (Full Chain)
    # 原始输入 -> 重写问题 -> (检索文档 + 生成答案)
    # ==========================================
    full_chain = (
        RunnablePassthrough.assign(
            standalone_question=standalone_question
        )
        | RunnablePassthrough.assign(
            question=lambda x: x["standalone_question"] # 将独立问题覆盖原始问题传入后续步骤
        )
        | retrieval_chain
    )
    
    return full_chain
