"""Base LLM Agent Module

该模块定义了LLM代理的基础抽象类，提供统一的接口和通用方法。
主要功能包括：
- 定义代理核心接口
- 提供通用的行为实现
- 支持不同LLM的扩展
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from diplomacy import Game


@dataclass
class LLMConfig:
    """LLM配置数据类"""
    model_name: str
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class BaseLLMAgent(ABC):
    """LLM代理基类"""
    
    def __init__(self,
                 power: str,
                 llm_config: LLMConfig):
        """初始化LLM代理基类
        
        Args:
            power: 控制的势力
            llm_config: LLM配置
        """
        self.power = power
        self.llm_config = llm_config
        
        # 初始化LLM模型
        self.llm = self._init_llm()
        
        # 初始化提示模板
        self._init_prompts()
    
    @abstractmethod
    def _init_llm(self) -> BaseChatModel:
        """初始化LLM模型
        
        Returns:
            BaseChatModel: 初始化的LLM模型实例
        """
        pass
    
    @abstractmethod
    def _init_prompts(self):
        """初始化提示模板，由子类实现具体的提示词"""
        pass
    
    @abstractmethod
    def get_llm_response(self, game: Game, max_retries: int = 3) -> str:
            """生成LLM回答

            Args:
                game: 当前游戏实例
                max_retries: 最大重试次数
                
            Returns:
                str: 生成的回答
                """
            pass

    @abstractmethod
    def parse_actions(self, game: Game, max_retries: int = 3) -> List[str]:
        """解析单位行动
        
        Args:
            game: 当前游戏实例
            max_retries: 最大重试次数
            
        Returns:
            List[str]: 生成的行动命令列表
        """
        pass
        


    