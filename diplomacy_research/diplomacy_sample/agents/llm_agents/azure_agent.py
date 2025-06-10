"""Azure OpenAI Agent Module

该模块实现基于Azure OpenAI的外交游戏代理。
主要功能包括：
- 实现基类定义的抽象方法
- 处理Azure OpenAI的配置和调用
- 定义特定的提示词模板
"""
import sys
import os
sys.path.append('/Your/project/diplomacy-research')
from diplomacy_research.models.state_space import get_map_powers
import asyncio
from typing import Any, Dict, List, Optional
import re
import time
import json
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain  
from diplomacy import Game
from .base_agent import BaseLLMAgent, LLMConfig



class AzureOpenAIAgent(BaseLLMAgent):
    """Azure OpenAI代理类"""
    
    def __init__(self, power: str, llm_config: LLMConfig):
        super().__init__(power, llm_config)
        self.current_state = None
        self.llm_response = None


    def _init_llm(self) -> BaseChatModel:
        """初始化Azure OpenAI模型
        
        Returns:
            BaseChatModel: 初始化的Azure OpenAI模型实例
        """
        return AzureChatOpenAI(
            azure_deployment=self.llm_config.model_name,
            openai_api_key=self.llm_config.api_key,
            azure_endpoint=self.llm_config.api_endpoint,
            api_version=self.llm_config.api_version,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            timeout=None,
            max_retries=2
        )
    
    
    def _init_prompts(self):
        """初始化Azure OpenAI特定的提示模板"""
        prompt_file = "diplomacy_research/diplomacy_sample/prompt.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.template = f.read()
        #
            # self.action_generation_prompt = ChatPromptTemplate.from_template(template)
        except Exception as e:
            print(f"读取提示词模板文件失败: {str(e)}")
            raise

    async def get_units_deployment(self, game: Game) -> Dict[str, List[str]]:
        """获取当前游戏中各国家单位的部署情况
        
        Args:
            game: 当前游戏实例
            
        Returns:
            Dict[str, List[str]]: 各国家单位部署字典
        """
        units_deployment = {}
        for power in game.powers.values():
            if power.units:
                units_deployment[power.name] = list(power.units)
        return units_deployment
    
    async def get_supply_centers_deployment(self, game: Game) -> Dict[str, List[str]]:
        """获取当前游戏中各国家控制的供应中心分布
        
        Args:
            game: 当前游戏实例
            
        Returns:
            Dict[str, List[str]]: 各国家控制的供应中心字典
        """
        centers_deployment = {}
        for power in game.powers.values():
            if power.centers:
                centers_deployment[power.name] = list(power.centers)
        return centers_deployment
    
    
    async def get_llm_response(self, game: Game) -> str:
        """生成LLM回答

        Args:
            game: 当前游戏实例

        Returns:
            str: 生成的回答
        """

        # 获取当前单位部署情况
        units_deployment = await self.get_units_deployment(game)
        if not units_deployment.get(self.power):
            print(f"No units found for power {self.power}")
            return []
        # 获取当前供应中心分布情况
        supply_centers_deployment = await self.get_supply_centers_deployment(game)
        # 获取当前未被占领的供应中心
        neutral_centers = [center for center in game.map.scs if not any(
            center in power.centers for power in game.powers.values()
        )] 
        # 分析当前游戏状态
        current_phase = game.get_current_phase()
        current_year = int(current_phase[1:5])
        season = current_phase[0]
        phase_type = current_phase[-1]
        current_season = 'Spring' if 'S' in current_phase else ('Fall' if 'F' in current_phase else 'Winter')
        build_disband_units_number = abs(len(supply_centers_deployment.get(self.power, [])) - len(units_deployment[self.power]))
        print(f" -----Generating actions for {self.power} in {current_phase}----- ")

        # 添加延迟以避免速率限制
        await asyncio.sleep(0.2)
        
        self.action_generation_prompt = ChatPromptTemplate.from_template(self.template)

        # 构建提示词
        chain = LLMChain(llm=self.llm, prompt=self.action_generation_prompt)
        response = await chain.arun(
            power=self.power,
            units_deployment=str(units_deployment),
            supply_centers_deployment=str(supply_centers_deployment),
            neutral_centers=str(neutral_centers),
            current_year=current_year,
            current_season=current_season,
            phase_type=game.phase_type,
            build_disband_units_number=build_disband_units_number,
        )
        
        self.llm_response = response
        print(f"response:{response}")
        return response

    def parse_actions(self, text: str = None) -> List[str]:
        """解析LLM响应中的行动命令
        
        Args:
            text: LLM响应文本,如果为None则使用最近一次的响应
            
        Returns:
            List[str]: 解析出的行动命令列表
        """
        text = text or self.llm_response
        if not text:
            return []
            
        # 直接使用LLM返回的命令列表
        try:
            # 将字符串形式的列表转换为实际的列表
            orders = eval(text)
            if isinstance(orders, list):
                return [order.strip() for order in orders if order.strip()]
        except:
            pass
            
        return []
        
    async def get_active_powers(self, game: Game):
        """Returns a list of powers that are still active in the game."""
        powers = get_map_powers(game.map)
        current_phase = game.get_current_phase()
        phase_type = current_phase[-1]  # 'M' for Movement, 'R' for Retreat, 'A' for Adjustment
        
        active_powers = []
        units_deployment = await self.get_units_deployment(game)
        centers_deployment = await self.get_supply_centers_deployment(game)
        
        for power in game.powers.values():
            units = units_deployment.get(power.name, [])
            centers = centers_deployment.get(power.name, [])
            # In Movement or Retreat phase, power is active if it has any units
            if phase_type in ['M', 'R']:
                if len(units) > 0:
                    active_powers.append(power.name)
            # In Adjustment phase, power is active ONLY if it has centers (units may exist but don't matter)
            elif phase_type == 'A':
                if len(centers) > 0:
                    active_powers.append(power.name)
        
        return active_powers

