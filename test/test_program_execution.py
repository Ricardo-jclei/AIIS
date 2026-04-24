"""测试program执行问题"""

import sys
import os
import inspect

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from funsearch.implementation.code_manipulation import text_to_program
from funsearch.implementation.evaluator import _sample_to_program

def test_program_execution():
    """测试program执行问题"""
    print("🚀 测试program执行问题...")
    
    # 加载specification
    import funsearch_specification_enhanced
    source = inspect.getsource(funsearch_specification_enhanced)
    
    # 创建template
    template = text_to_program(source)
    
    # 创建一个简单的sample（模拟LLM生成的代码）
    sample = """    # 改进版本：考虑市场状态
    if market_state is not None:
        # 使用市场状态计算权重
        weights = np.mean(market_state, axis=0)
        weights = weights / np.sum(weights)
        return weights
    else:
        return np.ones(len(portfolio)) / len(portfolio)"""
    
    print(f"📋 Sample:")
    print(repr(sample))
    print("-" * 50)
    
    # 使用_sample_to_program转换
    new_function, program = _sample_to_program(
        sample, version_generated=1, template=template, function_to_evolve='investment_strategy'
    )
    
    print(f"📋 New Function:")
    print(new_function)
    print("-" * 50)
    
    print(f"📋 Program:")
    print(program)
    print("-" * 50)
    
    # 分析program结构
    lines = program.split('\n')
    print(f"\n📊 Program分析:")
    print(f"总行数: {len(lines)}")
    print(f"前10行:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i}: {repr(line)}")
    
    print(f"\n后10行:")
    for i, line in enumerate(lines[-10:], len(lines)-10):
        print(f"  {i}: {repr(line)}")
    
    # 测试exec(program)
    print(f"\n🚀 测试exec(program)...")
    try:
        import numpy as np
        import torch
        import os
        import sys
        import pandas as pd
        import matplotlib
        import platform
        import yaml
        from datetime import datetime
        
        temp_namespace = {
            '__name__': '__main__',
            '__file__': 'generated_program.py',
            '__builtins__': __builtins__,
            'np': np,
            'torch': torch,
            'os': os,
            'sys': sys,
            'pd': pd,
            'matplotlib': matplotlib,
            'platform': platform,
            'yaml': yaml,
            'datetime': datetime,
        }
        
        # 使用compile预编译，避免缩进错误
        compiled_program = compile(program, '<string>', 'exec')
        exec(compiled_program, temp_namespace)
        print("✅ exec(program) 成功!")
        
        if 'investment_strategy' in temp_namespace:
            print("✅ investment_strategy 函数存在!")
            strategy_func = temp_namespace['investment_strategy']
            print(f"函数: {strategy_func}")
            
            # 测试函数调用
            print("\n🚀 测试策略函数调用...")
            test_market_state = np.random.rand(10, 5)
            test_portfolio = np.ones(5) / 5
            result = strategy_func(test_market_state, test_portfolio)
            print(f"✅ 策略函数调用成功!")
            print(f"   输入市场状态: {test_market_state.shape}")
            print(f"   输入投资组合: {test_portfolio}")
            print(f"   输出权重: {result}")
            
        else:
            print("❌ investment_strategy 函数不存在")
            print(f"可用函数: {[k for k in temp_namespace.keys() if callable(temp_namespace.get(k))]}")
            
    except Exception as e:
        print(f"❌ exec(program) 失败: {e}")
        
        # 检查缩进问题
        print(f"\n🔍 检查缩进问题...")
        for i, line in enumerate(lines):
            if 'unexpected indent' in str(e) and str(i+1) in str(e):
                print(f"问题行 {i+1}: {repr(line)}")
                # 显示上下文
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print(f"上下文:")
                for j in range(start, end):
                    marker = "-->" if j == i else "   "
                    print(f"{marker} {j+1}: {repr(lines[j])}")
    
    return program

if __name__ == "__main__":
    test_program_execution()