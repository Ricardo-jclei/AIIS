# AIIS/test_strategy_template.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np

# 加载specification
from funsearch_specification import investment_strategy

# 测试策略模板
feats = np.random.rand(224).astype(np.float32)
portfolio = np.array([100, 200, 300, 400, 500])

weights = investment_strategy(feats, portfolio)

print(f"策略输出形状: {weights.shape}")  # 应输出 (5,)
print(f"策略输出和: {np.sum(weights):.4f}")  # 应接近 1.0
print(f"✅ 策略模板验证通过！")