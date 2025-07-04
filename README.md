# Qwen3 本地 RAG Reasoning Agent

> 基于 Streamlit 的本地化检索增强生成（RAG）智能体，支持文档问答、天气查询、多模型切换等功能。

---

## 项目简介

本项目是一个本地化的 RAG（Retrieval-Augmented Generation）智能体应用，集成了 Qwen3 等主流大模型，支持上传 PDF/TXT 文档进行智能问答，并内置天气查询等实用工具。界面友好，易于扩展，适合个人和团队知识管理、智能问答等场景。

---

## 主要功能
- 📚 **文档智能问答**：上传 PDF/TXT 文档，支持基于内容的智能检索与问答
- 🤖 **多模型切换**：支持 Qwen3、DeepSeek 等多种大模型
- 🌤️ **天气查询**：内置高德地图天气API，支持全国城市天气实时查询
- 🧩 **嵌入模型可选**：支持多种文档嵌入模型，灵活切换
- 🗂️ **对话历史管理**：支持导出、清空历史，便于知识沉淀
- 🖥️ **本地运行，数据安全**

---

## 安装与运行

### 1. 克隆项目
```bash
# 推荐在RAG目录下操作
cd RAG
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖
```bash
cd Agent
pip install -r requirements.txt
```

### 4. 使用 Ollama 安装所需模型
```bash
# 安装Qwen3模型
ollama pull qwen3:8b
 # 安装嵌⼊模型
ollama pull bge-m3:latest
```

### 5. 启动应用
```bash
streamlit run app.py --server.port 6060
```

### 6. 访问界面
浏览器打开 http://localhost:6060

---

## 目录结构
```
RAG/
├── Agent/
│   ├── app.py                # 主程序入口
│   ├── requirements.txt      # 依赖列表
│   ├── config/               # 配置文件
│   ├── models/               # 智能体与模型封装
│   ├── services/             # 向量库、天气等服务
│   ├── utils/                # 工具与UI组件
│   ├── faiss_index/          # 向量索引（自动生成）
│   ├── .cache/               # 文档处理缓存
│   └── chat_history.json     # 对话历史（自动生成）
└── venv/                     # 虚拟环境（建议本地创建，不上传）
```

---

## API密钥与配置
- 天气查询功能需配置高德地图API密钥：
  - 在 `Agent/config/settings.py` 中设置 `AMAP_API_KEY = "你的高德API密钥"`
  - 建议用环境变量或 .env 文件管理密钥，避免泄露
- 其他模型、嵌入模型等参数可在 `settings.py` 中自定义

---

## 常见问题
- **Q: 启动报错/依赖冲突？**
  - 建议使用虚拟环境，确保 Python 3.8+，并完整安装 requirements.txt
- **Q: 无法访问模型或天气API？**
  - 检查本地端口、API密钥、网络连接
- **Q: 上传大文件或多文档慢？**
  - 向量化和分块过程较慢，耐心等待，或优化分块参数

---

## 致谢
- Qwen3、DeepSeek 等开源大模型
- LangChain、Streamlit、FAISS 等优秀开源项目
- 高德地图API

---

> 如有建议或问题，欢迎提 issue 或 PR！ 