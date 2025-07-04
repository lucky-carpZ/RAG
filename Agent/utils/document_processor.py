"""
文档处理模块
"""
import os
import hashlib
import json
from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path
import io
import tempfile
from utils.decorators import error_handler, log_execution
from datetime import datetime
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    文档处理器类，用于处理PDF文档
    """

    # 1. 初始化文档处理器
    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4):
        """
        cache_dir - 缓存目录,默认情况下在.cache下面
        max_workers - 最大工作线程数
        """
        self.cache_dir = Path(cache_dir)    ##传给Path对象，方便操作
        self.cache_dir.mkdir(exist_ok=True)     ##这两步就是创建文件常用操作
        self.max_workers = max_workers
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
            is_separator_regex=False    #分隔符是否为正则表达式
        )
    
    # 2. 获取缓存文件路径
    def _get_cache_path(self, file_content: bytes, file_name: str) -> Path:
        """
        file_content - 文件内容
        file_name - 文件名

        @return 缓存文件路径
        """                 ##下面这行生成一个缓存键，就是计算了这条内容的MD5哈希值
        cache_key = hashlib.md5(file_content + file_name.encode()).hexdigest()
        # logger.warning(f"\n缓存文件创建成功！！！\n")
        return self.cache_dir / f"{cache_key}.json"
    
    # 3. 从缓存加载处理结果
    def _load_from_cache(self, cache_path: str) -> Optional[List[Document]]:
        """
        @param {str} cache_path - 缓存文件路径
        @return {Optional[List[Document]]} 处理结果，如果缓存不存在则返回None
        """
        try:
            path = Path(cache_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [Document(**doc) for doc in data]
        except Exception as e:
            logger.warning(f"从缓存加载失败: {str(e)}")
        return None
    
    # 4. 保存处理结果到缓存
    def _save_to_cache(self, cache_path: Path, documents: List[Document]):
        """
        @param {Path} cache_path - 缓存文件路径
        @param {List[Document]} documents - 处理结果
        """
        try:
            # 将Document对象转换为可序列化的字典
            docs_data = [doc.dict() for doc in documents]
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
                logger.warning(f"\n缓存文件创建成功！！！\n")
        except Exception as e:
            logger.warning(f"保存到缓存失败: {str(e)}")
    

    # 5. 处理PDF文件
    @error_handler()
    @log_execution
    def _process_pdf(self, file_content: bytes, file_name: str) -> List[Document]:
        """
        file_content - PDF文件内容
        file_name - PDF文件名

        @return 处理后的文档列表
        """
        # 检查缓存
        cache_path = self._get_cache_path(file_content, file_name)
        cached_docs = self._load_from_cache(str(cache_path))
        if cached_docs is not None:
            logger.info(f"从缓存加载文件: {file_name}")
            return cached_docs
        
        # 处理PDF
        logger.info(f"处理文件: {file_name}")
        
        try:
            # 创建临时文件，使用上下文管理器自动清理
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # 使用临时文件加载PDF
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
                
                # 使用文本分割器分割文档
                split_docs = self.text_splitter.split_documents(documents)

                # 保存到缓存
                if split_docs:
                    self._save_to_cache(cache_path, split_docs)
                
                return split_docs
                
        except Exception as e:
            logger.error(f"处理PDF文件失败: {str(e)}")
            raise
    

    # 6. 清除所有缓存
    def clear_cache(self):
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            logger.info("缓存已清除")
        except Exception as e:
            logger.error(f"清除缓存失败: {str(e)}")
            raise


    # 7. 处理上传的文件，支持多种文件类型
    @error_handler()
    @log_execution
    def process_file(self, uploaded_file_or_content, file_name: str = None) -> Union[str, List[Document]]:
        """
        uploaded_file_or_content - Streamlit上传的文件对象或文件内容
        file_name - 文件名，当第一个参数为文件内容时需要提供

        @return  处理结果，可能是文本内容或Document对象列表
        """
        try:
            # 判断输入类型    ##上传是否有getvalue和那么属性
            if hasattr(uploaded_file_or_content, 'getvalue') and hasattr(uploaded_file_or_content, 'name'):
                # Streamlit上传的文件对象
                file_content = uploaded_file_or_content.getvalue()
                file_name = uploaded_file_or_content.name
            elif isinstance(uploaded_file_or_content, bytes) and file_name:
                # 直接传入的文件内容和文件名 ##instance是检查上传文件是否是字节类型，返回True，这情况是文件已经存在可认为是cache
                file_content = uploaded_file_or_content
            else:
                raise ValueError("参数错误：需要提供有效的文件对象或文件内容和文件名")
            
            # 根据文件类型进行特定处理
            if file_name.lower().endswith('.pdf'):
                docs = self._process_pdf(file_content, file_name)
                # 如果是从Streamlit上传的文件，返回文本内容，否则返回文档对象
                if hasattr(uploaded_file_or_content, 'getvalue'): ##检查对象是否有那个属性
                    return "\n\n".join(doc.page_content for doc in docs)
                return docs
            elif file_name.lower().endswith('.txt'):
                return file_content.decode('utf-8')  ##解析成utf-8字符串返回
            else:
                return f"不支持的文件类型: {file_name}"
            
        except Exception as e:
            logger.error(f"处理文件失败: {str(e)}")
            raise Exception(f"处理文件失败: {str(e)}")
            