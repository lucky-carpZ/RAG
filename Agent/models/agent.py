"""
智能体模型类
"""
from typing import Optional, Dict, Any, List
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function
from config.settings import DEFAULT_MODEL, AMAP_API_KEY
from services.weather_tools import WeatherTools
import logging

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    RAG智能体类，封装了与模型交互的功能
    """
    def __init__(self, model_version: str = DEFAULT_MODEL):
        """
        初始化RAG智能体
        
        Args:
            model_version (str): 模型版本名称
        """
        self.model_version = model_version
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """
        创建Agent实例
        
        Returns:
            Agent: 配置好的Agent实例
        """
        # 1. 创建天气查询工具
        weather_tools = WeatherTools(AMAP_API_KEY)
        
        # 2. 创建函数对象
        query_weather_function = Function(      #这里就是脱裤子放屁，变成了一种满足AI规则的格式
            name="query_weather",
            description="查询指定城市的天气预报",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询的城市名称"
                    }
                },
                "required": ["city"]
            },
            entrypoint=weather_tools.query_weather          ##实际调用函数
        )
        
        return Agent(
            name="Qwen 3 RAG Agent",
            model=Ollama(id=self.model_version),
            instructions="""你是一个智能助手，可以回答用户的各种问题。
                            【重要规则】
                            - 在RAG模式下，你只能基于提供的文档内容作答
                            - 在普通对话模式下，你可以基于你的知识回答问题
                            - 请确保对接收的内容以及需要做出的判断进行思考
                            - 回答要简明、准确、有帮助
                            - 如果用户询问天气相关信息，请使用query_weather工具查询天气

                            【决策流程】
                            1. 收到问题后，首先思考该问题是否可以通过你已有的知识回答
                            2. 将你的思考过程包含在<think></think>标签中
                            3. 对于天气查询，使用query_weather工具，例如：query_weather(city="深圳")
                            """,
            tools=[
                ReasoningTools(add_instructions=True),
                query_weather_function  # 使用Function实例
            ],
            show_tool_calls=True,
            markdown=True,
        )
            
    def run(self, prompt: str, context: Optional[str] = None) -> str:
        """
        运行智能体处理查询
        
        Args:
            prompt (str): 用户输入的提示
            context (Optional[str]): 可选的文档上下文
            
        Returns:
            str: 智能体的响应
        """
        if context:
            # RAG模式：有上下文的情况
            full_prompt = f"""【检索内容】\n{context}\n\n【用户问题】\n{prompt}\n\n请严格按照【检索内容】作答。但如果是询问天气相关信息，请直接使用query_weather工具获取实时天气数据。"""
        else:
            # 普通对话模式：无上下文的情况
            full_prompt = f"【用户问题】\n{prompt}\n\n请提供准确、有帮助的回答。如果用户询问天气，请使用query_weather工具。"
        
        # 让Agent处理请求，会自动判断是否使用天气查询工具
        response = self.agent.run(full_prompt)
        return response.content