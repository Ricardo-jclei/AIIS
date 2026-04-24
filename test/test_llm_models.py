"""测试不同LLM模型生成策略能力"""

import sys
import os

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)

from funsearch.implementation.sampler import LLM

class TestLLM(LLM):
    """测试用的LLM类，支持切换模型"""
    def __init__(self, samples_per_prompt: int, model: str):
        super().__init__(samples_per_prompt)
        self.model = model
    
    def _draw_sample(self, prompt: str) -> str:
        """使用指定模型生成代码"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert investment strategy developer. Generate innovative, diverse, and effective investment strategies. Use market_state[:, :5] (first 5 features) for 5 stocks. Return only pure function body code, no explanation. Include:\n1. Market trend analysis using recent price movements\n2. Volatility-based weighting\n3. Mean-reversion strategies\n4. Momentum-based approaches\n5. Risk control mechanisms\n6. Diverse weighting schemes\n7. Proper handling of portfolio=None case\n8. Return shape must be (5,)\n9. Weights must be non-negative and sum to 1\n10. Use only numpy operations"},
                    {"role": "user", "content": f"Complete this investment strategy function. Use market_state[:, :5] for 5 stocks. Generate diverse, innovative strategies. Examples of good strategies:\n\nExample 1: Trend-following with volatility adjustment\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\ntrend = market_state[-1, :5] - market_state[0, :5]\nvolatility = np.std(market_state[:, :5], axis=0)\nweights = np.exp(trend) / (volatility + 1e-8)\nweights = weights / np.sum(weights)\nreturn weights\n\nExample 2: Mean-reversion with dynamic adjustment\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\nmean_prices = np.mean(market_state[:, :5], axis=0)\ncurrent_prices = market_state[-1, :5]\nweights = mean_prices / (current_prices + 1e-8)\nweights = weights / np.sum(weights)\nreturn weights\n\nExample 3: Momentum-based with risk parity\nif portfolio is None:\n    portfolio = market_state[-1, :5] if market_state is not None else np.ones(5)\nmomentum = market_state[-5:, :5].mean(axis=0) - market_state[:5, :5].mean(axis=0)\nvolatility = np.std(market_state[:, :5], axis=0)\nweights = momentum / (volatility + 1e-8)\nweights = np.maximum(0, weights)\nweights = weights / np.sum(weights)\nreturn weights\n\nNow complete:\n{prompt}"}
                ],
                temperature=0.8,
                max_tokens=400
            )
            generated_code = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[API Error] {str(e)} → Using fallback strategy")
            generated_code = ""
        
        print(f"[LLM] Model: {self.model}")
        print(f"[LLM] Generated code:\n{generated_code}")
        return generated_code

def test_different_models():
    """测试不同LLM模型"""
    print("测试不同LLM模型生成策略能力...")
    
    # 测试的模型列表
    models = [
        "gpt-5-nano",
        "gpt-4o-mini",
        "gemini-3-flash",
        "gemini-3.1-flash-lite"
    ]
    
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
    print("-" * 80)
    
    results = {}
    
    for model in models:
        print(f"\n测试模型: {model}")
        print("-" * 60)
        
        try:
            llm = TestLLM(samples_per_prompt=1, model=model)
            samples = llm.draw_samples(prompt)
            results[model] = samples[0]
        except Exception as e:
            print(f"测试失败: {e}")
            results[model] = f"Error: {e}"
    
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    for model, code in results.items():
        print(f"\n模型: {model}")
        print("-" * 40)
        print(code)
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    test_different_models()