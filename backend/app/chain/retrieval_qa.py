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
    """将文档格式化为单个字符串，用于 Prompt 上下文。"""
    return "\n\n".join(f"<doc id={i}>\n{doc.page_content}\n</doc>" for i, doc in enumerate(docs))

def get_retrieval_qa_chain() -> Any:
    """构建并返回 Retrieval QA 链。
    
    Returns:
        Runnable: 配置好的 LCEL 链。
    """
    llm = get_llm(temperature=0)
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 5})

    # 1. 独立问题重写 Prompt
    condense_question_system_template = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能会引用聊天历史记录中的上下文，"
        "请重写为一个可以脱离聊天历史记录独立理解的问题。"
        "不要回答问题，如果需要则重写，否则原样返回。"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 问题生成器 Runnable
    standalone_question = condense_question_prompt | llm | StrOutputParser()
    
    # 2. 答案生成 Prompt
    qa_system_template = (
        "你是一个乐于助人的助手。请使用以下检索到的上下文片段来回答问题。"
        "如果你不知道答案，就说你不知道。最多使用三句话，并保持回答简洁。\n"
        "上下文：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_template),
        ("human", "{question}")
    ])
    
    # 上下文检索与组装
    retrieved_docs = retriever
    
    def get_context(inputs: dict) -> List[Any]:
        # 基于独立问题检索文档
        docs = retriever.invoke(inputs["question"])
        return docs
        
    def _format_docs_for_prompt(docs: List[Any]) -> str:
        return format_docs(docs)

    # 使用 LCEL 构建链
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
    
    # 组装完整链：重写 -> 检索 -> 回答
    full_chain = (
        RunnablePassthrough.assign(
            standalone_question=standalone_question
        )
        | RunnablePassthrough.assign(
            question=lambda x: x["standalone_question"]
        )
        | retrieval_chain
    )
    
    return full_chain
