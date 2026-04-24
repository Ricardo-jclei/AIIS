"""测试FunSearch的prompt生成"""

import sys
import os
import inspect

# 添加路径
funsearch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "funsearch"))
sys.path.insert(0, funsearch_path)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from funsearch.implementation.code_manipulation import text_to_program
from funsearch.implementation.programs_database import ProgramsDatabase
from funsearch.implementation.config import ProgramsDatabaseConfig

def test_funsearch_prompt():
    """测试FunSearch生成的prompt"""
    print("🚀 测试FunSearch的prompt生成...")
    
    # 加载specification
    import funsearch_specification_enhanced
    source = inspect.getsource(funsearch_specification_enhanced)
    
    # 创建template
    template = text_to_program(source)
    
    # 创建ProgramsDatabase
    config = ProgramsDatabaseConfig(
        functions_per_prompt=2,
        num_islands=10,
        reset_period=4 * 60 * 60,
        cluster_sampling_temperature_init=0.1,
        cluster_sampling_temperature_period=30_000
    )
    
    database = ProgramsDatabase(config, template, 'investment_strategy')
    
    # 先添加初始策略版本到数据库
    print("📝 添加初始策略版本到数据库...")
    initial_function = template.get_function('investment_strategy')
    initial_score = 0.5  # 初始分数
    
    # 为每个island添加初始版本
    for island_id in range(config.num_islands):
        database.register_program(
            program=initial_function,  # 传递Function对象，不是字符串
            island_id=island_id,
            scores_per_test={str({'window_size': 20, 'island_id': island_id}): initial_score}
        )
    
    print("✅ 初始策略版本添加完成")
    
    # 获取prompt
    prompt_obj = database.get_prompt()
    
    print(f"📋 FunSearch生成的prompt:")
    print(prompt_obj.code)
    print("-" * 50)
    print(f"版本: {prompt_obj.version_generated}")
    print(f"岛屿ID: {prompt_obj.island_id}")
    
    # 分析prompt结构
    lines = prompt_obj.code.split('\n')
    print(f"\n📊 Prompt分析:")
    print(f"总行数: {len(lines)}")
    print(f"前10行:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i}: {repr(line)}")
    
    print(f"\n后10行:")
    for i, line in enumerate(lines[-10:], len(lines)-10):
        print(f"  {i}: {repr(line)}")
    
    return prompt_obj.code

if __name__ == "__main__":
    test_funsearch_prompt()
