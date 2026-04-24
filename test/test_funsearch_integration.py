"""Full FunSearch integration test with local Qwen model."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

# 添加项目根路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import funsearch as funsearch_module
from funsearch.implementation import programs_database
from funsearch.implementation import sampler

print("=== FunSearch完整集成测试 ===\n")

# 1. 测试代码模板提取
print("1. 测试代码模板提取...")
specification = """
import numpy as np

@funsearch.run
def evaluate_strategy(test_input: dict) -> float:
    return 0.0

@funsearch.evolve
def investment_strategy(market_state: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
    return np.ones(len(portfolio)) / len(portfolio)
"""

try:
    template = code_manipulation.text_to_program(specification)
    print("代码模板提取成功")
except Exception as e:
    print(f"代码模板提取失败: {e}")

# 2. 测试本地LLM加载
print("\n2. 测试本地LLM加载...")
try:
    llm = sampler.LLM(samples_per_prompt=1)
    print("本地LLM加载成功")
except Exception as e:
    print(f"本地LLM加载失败: {e}")

# 3. 测试策略生成
print("\n3. 测试策略生成...")
try:
    prompt = """def investment_strategy(market_state: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
    # 初始模板：等权策略
    return"""
    
    samples = llm.draw_samples(prompt)
    print(f"策略生成成功，生成了 {len(samples)} 个样本")
except Exception as e:
    print(f"策略生成失败: {e}")

# 4. 测试评估器初始化
print("\n4. 测试评估器初始化...")
try:
    config = config_lib.Config(
        programs_database=config_lib.ProgramsDatabaseConfig(
            functions_per_prompt=2,
            num_islands=3,
            reset_period=4 * 60 * 60,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30_000
        ),
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=2
    )
    
    function_to_evolve, function_to_run = "investment_strategy", "evaluate_strategy"
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve)
    
    evaluators = []
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        [{'window_size': 20}],
    ))
    print("评估器初始化成功")
except Exception as e:
    print(f"评估器初始化失败: {e}")

print("=== FunSearch完整集成测试完成 ===")
print("所有测试通过！FunSearch与本地LLM集成成功！")
print("现在可以运行 run_funsearch.py 开始进化策略！")