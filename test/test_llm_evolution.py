"""测试LLM进化能力"""

import sys
import os

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)

from funsearch.implementation.sampler import LLM

def test_llm_evolution():
    """测试LLM是否能基于前一轮结果生成新策略"""
    print("测试LLM进化能力...")
    
    # 创建LLM实例
    llm = LLM(samples_per_prompt=1)
    
    # 初始策略
    initial_strategy = """def investment_strategy(market_state: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
    # Initial template: equal weight
    # 处理portfolio为None的情况
    if portfolio is None:
        portfolio = market_state[-5:] if market_state is not None else np.ones(5)
    return np.ones(len(portfolio)) / len(portfolio)
"""
    
    print("初始策略:")
    print(initial_strategy)
    print("-" * 60)
    
    # 第一轮生成
    print("第一轮生成:")
    prompt1 = initial_strategy + "\n# 请生成改进版本："
    samples1 = llm.draw_samples(prompt1)
    first_strategy = samples1[0]
    print(f"生成的策略 1:\n{first_strategy}")
    print("-" * 60)
    
    # 第二轮生成（基于第一轮结果）
    print("第二轮生成（基于第一轮结果）:")
    prompt2 = f"""{initial_strategy}

# 上一轮生成的策略：
{first_strategy}

# 请基于上一轮策略生成更优的版本："""
    samples2 = llm.draw_samples(prompt2)
    second_strategy = samples2[0]
    print(f"生成的策略 2:\n{second_strategy}")
    print("-" * 60)
    
    # 第三轮生成（基于第二轮结果）
    print("第三轮生成（基于第二轮结果）:")
    prompt3 = f"""{initial_strategy}

# 上一轮生成的策略：
{second_strategy}

# 请基于上一轮策略生成更优的版本："""
    samples3 = llm.draw_samples(prompt3)
    third_strategy = samples3[0]
    print(f"生成的策略 3:\n{third_strategy}")
    print("-" * 60)
    
    print("测试完成！")
    return [initial_strategy, first_strategy, second_strategy, third_strategy]

if __name__ == "__main__":
    test_llm_evolution()