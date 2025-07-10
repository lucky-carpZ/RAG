##程序主入口

import streamlit as st
from datetime import datetime
import logging
import re
from config.settings import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    AVAILABLE_EMBEDDING_MODELS
)
# RAGAgent: 用于处理用户输入和生成响应的智能体，封装模型交互逻辑。
from models.agent import RAGAgent
# ChatHistoryManager: 管理对话历史
from utils.chat_history import ChatHistoryManager
# DocumentProcessor: 处理用户上传的文档
from utils.document_processor import DocumentProcessor
# VectorStoreService: 向量数据库服务，用于文档索引与检索
from services.vector_store import VectorStoreService
# UIComponents: 用户界面组件，用于渲染UI
from utils.ui_components import UIComponents
from utils.decorators import error_handler, log_execution

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class App:
    """
    RAG应用主类
    """
    def __init__(self):
        """
        @description 初始化应用
        """
        self._init_session_state()  # 初始化会话状态
        self.chat_history = ChatHistoryManager()  # 创建聊天历史管理器
        self.document_processor = DocumentProcessor()  # 创建文档处理器
        self.vector_store = VectorStoreService()  # 创建向量存储服务
        logger.info("应用初始化成功")
    
    # 1. 初始化会话状态
    @error_handler(show_error=False)
    def _init_session_state(self):
        if 'model_version' not in st.session_state:
            st.session_state.model_version = DEFAULT_MODEL  # 设置默认模型
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []  # 初始化已处理文档列表
        if 'similarity_threshold' not in st.session_state:
            st.session_state.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD  # 设置默认相似度阈值
        if 'rag_enabled' not in st.session_state:
            st.session_state.rag_enabled = True  # 默认启用RAG功能
        if 'embedding_model' not in st.session_state:
            st.session_state.embedding_model = EMBEDDING_MODEL  # 设置默认嵌入模型
    
    # 2. 渲染侧边栏
    @error_handler()
    @log_execution
    def render_sidebar(self):
        # 更新模型选择和嵌入模型选择
        st.session_state.model_version, new_embedding_model = UIComponents.render_model_selection(
            AVAILABLE_MODELS,
            st.session_state.model_version,
            AVAILABLE_EMBEDDING_MODELS,
            st.session_state.embedding_model
        )
        
        # 检查嵌入模型是否更改
        previous_embedding_model = st.session_state.embedding_model
        st.session_state.embedding_model = new_embedding_model
        
        # 渲染更新RAG设置
        st.session_state.rag_enabled, st.session_state.similarity_threshold = UIComponents.render_rag_settings(
            st.session_state.rag_enabled,
            st.session_state.similarity_threshold,
            DEFAULT_SIMILARITY_THRESHOLD
        )
        
        # 渲染更新向量存储服务的嵌入模型
        if previous_embedding_model != st.session_state.embedding_model:
            if self.vector_store.update_embedding_model(st.session_state.embedding_model):
                # 如果向量存储已存在，则提示用户可能需要重新处理文档，上面比的是已处理文档的嵌入模型和现在已更新的嵌入模型是否一样
                if len(st.session_state.processed_documents) > 0:
                    st.sidebar.info(f"⚠️ 嵌入模型已更改为 {st.session_state.embedding_model}，您可能需要重新处理文档以使用新的嵌入模型。")
        
        # 渲染聊天统计
        UIComponents.render_chat_stats(self.chat_history)
    

    # 3. 渲染文档上传区域
    @error_handler()
    @log_execution
    def render_document_upload(self):
        all_docs, self.vector_store = UIComponents.render_document_upload(
            self.document_processor,
            self.vector_store,
            st.session_state.processed_documents    ##已处理文档
        )
    
    # 4. 处理用户输入
    @error_handler()
    @log_execution
    def process_user_input(self, prompt: str):
        """
        prompt - 用户输入的提示文本

        1️⃣ RAG模式：检索相关文档→获取上下文→调用模型
        2️⃣ 普通模式：直接调用模型
        """
        self.chat_history.add_message("user", prompt)  # 将用户消息添加到聊天历史
        if st.session_state.rag_enabled:
            self._process_rag_query(prompt)  # 如果启用RAG，处理RAG查询
        else:
            self._process_simple_query(prompt)  # 否则处理简单查询
    
    
    # 5. 处理RAG查询
    @error_handler()
    @log_execution
    def _process_rag_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        with st.spinner("🤔正在评估查询..."): 
            # 搜索相关文档
            docs = self.vector_store.search_documents(  
                prompt,
                st.session_state.similarity_threshold
            )
            logger.info(f"检索到的文档数: {len(docs)}")  
            # 获取文档上下文
            context = self.vector_store.get_context(docs)  
            # 创建RAG代理
            agent = RAGAgent(st.session_state.model_version)  
            # 运行代理获取响应
            response = agent.run(  
                prompt, 
                context=context
            )
            # 处理响应
            self._process_response(response, docs)  
    

    # 6. 处理简单查询
    @error_handler()
    @log_execution
    def _process_simple_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        with st.spinner("🤖 思考中..."): 
            # 创建RAG代理
            agent = RAGAgent(st.session_state.model_version)  
            # 运行代理获取响应
            response = agent.run(prompt)  
            # 处理响应
            self._process_response(response)  
    
    
    # 7. 处理Agent的响应
    def _process_response(self, response: str, docs=None):
        """
        response - 模型的原始响应
        docs - 检索到的文档（可选）
        """
        # 7.1 处理响应中的思考过程
        think_pattern = r'<think>([\s\S]*?)</think>'  # 定义思考过程的正则表达式模式
        think_match = re.search(think_pattern, response)  # 搜索思考过程
        if think_match:
            think_content = think_match.group(1).strip()  # 提取思考内容
            response_wo_think = re.sub(think_pattern, '', response).strip()  # 移除思考部分
        else:
            think_content = None
            response_wo_think = response
        
        # 7.2 保存响应到历史
        self.chat_history.add_message("assistant", response_wo_think)  # 添加助手回复
        if think_content:
            self.chat_history.add_message("assistant_think", think_content)  # 添加思考过程
        if docs:
            doc_contents = [doc.page_content for doc in docs]  # 提取文档内容
            self.chat_history.add_message("retrieved_doc", doc_contents)  # 添加检索到的文档


    # 入口处：运行应用
    @error_handler()
    @log_execution
    def run(self):
        st.title("🐋 Qwen 3 本地 RAG Reasoning Agent")  # 设置应用标题
        st.info("**Qwen3:** Qwen系列最新一代大语言模型，提供全面的密集型和混合专家(MoE)模型套件。")  # 显示模型信息
        
        self.render_sidebar()  # 渲染侧边栏
        self.render_document_upload()  # 渲染文档上传区域
        
        chat_col = st.columns([1])[0]  # 创建聊天列
        with chat_col:
            prompt = st.chat_input(  # 创建聊天输入框
                "询问您的文档..." if st.session_state.rag_enabled else "问我任何问题..."
            )
            
            if prompt:
                self.process_user_input(prompt)  # 处理用户输入
                
            # 渲染聊天历史
            UIComponents.render_chat_history(self.chat_history)
        
        mode_description = ""
        if st.session_state.rag_enabled:
            mode_description += "📚 RAG模式：可以询问上传文档的内容。"  
        else:
            mode_description += "💬 对话模式：直接与模型交流。"  
        
        mode_description += " 🌤️ 天气查询：可以询问任何城市的天气情况。"  
            
        st.info(mode_description)  # 显示模式描述

if __name__ == "__main__":
    app = App()  # 创建应用实例
    app.run()  # 运行应用
