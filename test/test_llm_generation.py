"""测试LLM生成策略功能"""

import sys
import os

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)

from funsearch.implementation.sampler import LLM

def test_llm_generation():
    """测试LLM是否能正常生成策略代码"""
    print("测试LLM生成策略功能...")
    
    # 创建LLM实例
    llm = LLM(samples_per_prompt=1)
    
    # 创建一个简单的prompt
    prompt = """def investment_strategy(market_state: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
    # Initial template: equal weight
    # 处理portfolio为None的情况
    if portfolio is None:
        portfolio = market_state[-5:] if market_state is not None else np.ones(5)
    return np.ones(len(portfolio)) / len(portfolio)

# 请生成改进版本："""
    
    print(f"Prompt:")
    print(prompt)
    print("-" * 50)
    
    # 生成样本
    samples = llm.draw_samples(prompt)
    
    print(f"生成的样本:")
    for i, sample in enumerate(samples):
        print(f"样本 {i}:")
        print(repr(sample))
        print("-" * 30)
    
    return samples

if __name__ == "__main__":
    test_llm_generation()