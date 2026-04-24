"""Test the new package structure."""

import sys
import os

# 添加funsearch目录到路径
funsearch_dir = os.path.abspath('funsearch')
sys.path.insert(0, funsearch_dir)

print("=== FunSearch包结构测试 ===\n")

# 1. 测试funsearch包导入
print("1. 测试funsearch包导入...")
try:
    import funsearch
    print("✅ import funsearch - 成功")
except Exception as e:
    print(f"❌ import funsearch - 失败: {e}")

# 2. 测试implementation子包导入
print("\n2. 测试implementation子包导入...")
try:
    from funsearch import implementation
    print("✅ from funsearch import implementation - 成功")
except Exception as e:
    print(f"❌ from funsearch import implementation - 失败: {e}")

# 3. 测试具体模块导入
print("\n3. 测试具体模块导入...")
try:
    from funsearch.implementation import code_manipulation
    print("✅ from funsearch.implementation import code_manipulation - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import code_manipulation - 失败: {e}")

try:
    from funsearch.implementation import config
    print("✅ from funsearch.implementation import config - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import config - 失败: {e}")

try:
    from funsearch.implementation import evaluator
    print("✅ from funsearch.implementation import evaluator - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import evaluator - 失败: {e}")

try:
    from funsearch.implementation import funsearch
    print("✅ from funsearch.implementation import funsearch - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import funsearch - 失败: {e}")

try:
    from funsearch.implementation import programs_database
    print("✅ from funsearch.implementation import programs_database - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import programs_database - 失败: {e}")

try:
    from funsearch.implementation import sampler
    print("✅ from funsearch.implementation import sampler - 成功")
except Exception as e:
    print(f"❌ from funsearch.implementation import sampler - 失败: {e}")

# 4. 测试本地LLM加载
print("\n4. 测试本地LLM加载...")
try:
    from funsearch.implementation.sampler import LLM
    llm = LLM(samples_per_prompt=1)
    print("✅ 本地LLM加载成功")
except Exception as e:
    print(f"❌ 本地LLM加载失败: {e}")

print("\n=== FunSearch包结构测试完成 ===")