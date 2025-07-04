"""
配置文件，包含所有常量和配置项
"""

# 1. 文件路径
VECTOR_STORE_PATH = "faiss_index"   ##向量存储的路径
HISTORY_FILE = "chat_history.json"  ##聊天记录存储的路径

# 2. 模型配置
DEFAULT_MODEL = "qwe3:8b"
AVAILABLE_MODELS = ["qwen3:1.7b", "deepseek-r1:1.5b", "qwen3:8b"]

EMBEDDING_MODEL = "bge-m3:latest"
AVAILABLE_EMBEDDING_MODELS = ["bge-m3:latest", "nomic-embed-text:latest", "mxbai-embed-large:latest", "bge-large-en-v1.5:latest", "bge-large-zh-v1.5:latest"]
EMBEDDING_BASE_URL = "http://localhost:11434"


# 3. RAG配置
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_CHUNK_SIZE = 300        ##文档分块大小
DEFAULT_CHUNK_OVERLAP = 30      ##文档分块重叠大小
MAX_RETRIEVED_DOCS = 3          ##最大检索文档数

# 4. 高德地图API配置
AMAP_API_KEY = "" 

# 5. LangChain配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 6. 对话历史配置
MAX_HISTORY_TURNS = 5 