from diplomacy import Game
import asyncio
import sys
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
sys.path.append('/Your/project/diplomacy-research')
from diplomacy_research.models.state_space import get_map_powers
from diplomacy_research.players.rule_based_player import RuleBasedPlayer
from diplomacy_research.diplomacy_sample.agents.llm_agents.azure_agent import AzureOpenAIAgent as AzureAgent
from diplomacy_research.diplomacy_sample.agents.llm_agents.base_agent import LLMConfig
from diplomacy_research.players.rulesets import easy_ruleset
import time


def init_rule_based_agents(powers):
    """初始化规则基础代理
    
    Args:
        powers: 需要控制的势力列表
        
    Returns:
        Dict[str, RuleBasedPlayer]: 势力到代理的映射
    """
    return {power: RuleBasedPlayer(ruleset=easy_ruleset) for power in powers}


async def run_test_game(
    game: Game,
    llm_agent: AzureAgent,
    llm_power: str,
):

    """运行测试游戏
    
    Args:
        game: 游戏实例
        llm_agent: LLM代理
        llm_power: LLM代理控制的势力
    """
    # 获取所有势力
    powers = get_map_powers(game.map)

    # 初始化规则基础代理
    rule_based_agents = init_rule_based_agents([p for p in powers if p != llm_power])

   

   
    # 游戏主循环
    while not game.is_game_done:
        active_powers = await llm_agent.get_active_powers(game)
        # 初始化当前回合的actions字典
        actions = {power: [] for power in active_powers}
        units_deployment = await llm_agent.get_units_deployment(game)
        supply_centers_deployment = await llm_agent.get_supply_centers_deployment(game)
        # 处理每个未被淘汰势力的行动
        for power in active_powers:
            if power == llm_power:
                # 获取当前游戏状态
                current_phase = game.get_current_phase()
                build_disband_units_number = abs(len(supply_centers_deployment.get(power, [])) - len(units_deployment[power]))
                # LLM需要代理行动
                if not (build_disband_units_number == 0 and current_phase[-1] == 'A'):
                    response = await llm_agent.get_llm_response(game)
                    await asyncio.sleep(0.3)  # 添加延迟以避免请求过快
    
                    # 解析响应获取命令
                    orders_llm = llm_agent.parse_actions(response)
                    game.set_orders(power, orders_llm)
                    # 记录LLM代理的命令
                    actions[power] = orders_llm
                else:
                    orders_llm = [] 
                    actions[power] = orders_llm # 由于LLM agent跳过决策 设置空命令列表
                    
            else:
                # 规则基础代理行动
                orders = await rule_based_agents[power].get_orders(game, power)  # 添加 await
                game.set_orders(power, orders)
                # 记录规则基础代理的命令
                actions[power] = orders
        

        # 处理当前回合
        game.process()

    if game.is_game_done:
        print("\n游戏结束!")
        print(f"FRANCE最终供应中心数量: {len(game.get_centers(llm_power))}")
        supply_centers_deployment = await llm_agent.get_supply_centers_deployment(game)
        print(f"最终供应中心部署情况: {supply_centers_deployment}")

        

    


if __name__ == "__main__":
    # 检查必要的环境变量
    required_env_vars = [
        'AZURE_API_KEY',
        'AZURE_API_ENDPOINT',
        'AZURE_API_VERSION',
        'AZURE_DEPLOYMENT_NAME'
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f'环境变量 {var} 未设置')
    
    # 创建LLM配置
    llm_config = LLMConfig(
        model_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
        api_key=os.getenv('AZURE_API_KEY'),
        api_endpoint=os.getenv('AZURE_API_ENDPOINT'),
        api_version=os.getenv('AZURE_API_VERSION'),
        temperature=0.3,  # 降低随机性以获得更稳定的输出
        max_tokens=4000  # 增加最大token数以确保完整的响应
    )
    
    # 创建游戏实例
    game = Game()
    
    # 创建LLM代理
    llm_agent = AzureAgent(power="FRANCE", llm_config=llm_config)
    
    # 运行测试游戏
    asyncio.run(run_test_game(
        game=game,
        llm_agent=llm_agent,
        llm_power="FRANCE"
    ))
