"""
UI组件模块，包含所有Streamlit UI渲染逻辑
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Any
import logging
from utils.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from langchain.schema import Document
from config.settings import AVAILABLE_EMBEDDING_MODELS

logger = logging.getLogger(__name__)

class UIComponents:
    """UI组件类，封装了所有Streamlit UI渲染逻辑"""
    
    # 1.渲染模型选择组件
    @staticmethod
    def render_model_selection(available_models: List[str], current_model: str, embedding_models: List[str], current_embedding_model: str) -> Tuple[str, str]:
        """
        available_models - 可用模型列表
        current_model - 当前选中的模型
        embedding_models - 可用嵌入模型列表
        current_embedding_model - 当前选中的嵌入模型

        @return (用户选择的模型, 用户选择的嵌入模型)
        """
        st.sidebar.header("⚙️ 设置")
        
        new_model = st.sidebar.selectbox(
            "选择模型",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="选择要使用的语言模型"
        )
        
        new_embedding_model = st.sidebar.selectbox(
            "嵌入模型",
            options=embedding_models,
            index=embedding_models.index(current_embedding_model) if current_embedding_model in embedding_models else 0,
            help="选择用于文档嵌入的模型"
        )
        
        return new_model, new_embedding_model
    

    # 2. 渲染RAG设置组件
    @staticmethod
    def render_rag_settings(rag_enabled: bool, similarity_threshold: float, default_threshold: float) -> Tuple[bool, float]:
        """
        rag_enabled - 是否启用RAG
        similarity_threshold - 相似度阈值
        default_threshold - 默认相似度阈值

        @return (是否启用RAG, 相似度阈值)
        """
        st.sidebar.subheader("RAG设置")
        
        new_rag_enabled = st.sidebar.checkbox(
            "启用RAG",
            value=rag_enabled,
            help="启用检索增强生成功能，使用上传的文档增强回答"
        )
        
        new_similarity_threshold = st.sidebar.slider(
            "相似度阈值",
            min_value=0.0,
            max_value=1.0,
            value=similarity_threshold,
            step=0.05,
            help="调整检索相似度阈值，值越高要求匹配度越精确"
        )
        
        # 将重置相似度阈值按钮样式更改为容器宽度
        if st.sidebar.button("重置相似度阈值", use_container_width=True):
            new_similarity_threshold = default_threshold
            
        return new_rag_enabled, new_similarity_threshold
    

    # 3. 渲染聊天统计信息
    @staticmethod
    def render_chat_stats(chat_history):
        """
        chat_history - 聊天历史管理器
        """
        st.sidebar.header("💬 对话历史")
        stats = chat_history.get_stats()
        st.sidebar.info(f"总对话数: {stats['total_messages']} 用户消息: {stats['user_messages']}")
        
        if st.sidebar.button("📥 导出对话历史", use_container_width=True):
            csv = chat_history.export_to_csv()
            if csv:
                st.sidebar.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if st.sidebar.button("✨ 清空对话", use_container_width=True):
            chat_history.clear_history()
            st.rerun()


    # 4. 渲染文档上传组件
    @staticmethod
    def render_document_upload(
        document_processor: DocumentProcessor,
        vector_store: VectorStoreService,
        processed_documents: List[str]
    ) -> Tuple[List[Document], VectorStoreService]:
        """
        document_processor - 文档处理器
        vector_store - 向量存储服务
        processed_documents - 已处理的文档列表

        @return (all_docs, vector_store)
        """
        ##with结束，expander自动关闭
        with st.expander("📁 上传RAG文档", expanded=not bool(processed_documents)):
            uploaded_files = st.file_uploader(
                "上传PDF、TXT文件", 
                type=["pdf", "txt"],
                accept_multiple_files=True
            )
            
            if not vector_store.vector_store:   ##他初始值为None，代表还没有启用文档
                st.warning("⚠️ 请在侧边栏配置向量存储以启用文档处理。")
            
            all_docs = []
            if uploaded_files:
                if st.button("处理文档"):
                    with st.spinner("正在处理文档..."):
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name not in processed_documents:
                                try:    ##上面说明上传文件名不在已处理文档就执行下面
                                    # 统一处理所有文件类型
                                    result = document_processor.process_file(uploaded_file)
                                    
                                    if isinstance(result, list):
                                        # 结果是Document列表(PDF文档)
                                        all_docs.extend(result)
                                    else:
                                        # 结果是文本内容(TXT、DOCX等)
                                        doc = Document(
                                            page_content=result, 
                                            metadata={"source": uploaded_file.name}
                                        )
                                        all_docs.append(doc)
                                    
                                    processed_documents.append(uploaded_file.name)
                                    st.success(f"✅ 已处理: {uploaded_file.name}")
                                except Exception as e:
                                    st.error(f"❌ 处理失败: {uploaded_file.name} - {str(e)}")
                            else:
                                st.warning(f"⚠️ 已存在: {uploaded_file.name}")
                
                if all_docs:
                    with st.spinner("正在构建向量索引..."):
                        vector_store.vector_store = vector_store.create_vector_store(all_docs)
            
            # 显示已处理文档列表
            if processed_documents:
                st.subheader("已处理文档")
                for doc in processed_documents:
                    st.markdown(f"- {doc}")
                
                if st.button("清除所有文档"):
                    with st.spinner("正在清除向量索引..."):
                        vector_store.clear_index()
                        processed_documents.clear()
                    st.success("✅ 所有文档已清除")
                    st.rerun()
            
            return all_docs, vector_store


    # 5. 渲染聊天历史
    @staticmethod
    def render_chat_history(chat_history):
        """
        chat_history - 聊天历史管理器
        """
        for message in chat_history.history:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == "assistant_think":
                with st.expander("💡 查看推理过程 <think> ... </think>"):
                    st.markdown(content)
            elif role == "retrieved_doc":
                with st.expander(f"🔎 查看本次召回的文档块", expanded=False):
                    if isinstance(content, list):
                        for idx, doc in enumerate(content, 1):
                            st.markdown(f"**文档块{idx}:**\n{doc}")
                    else:
                        st.markdown(content)
            else:
                with st.chat_message(role):
                    st.write(content) 