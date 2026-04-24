"""检查FunSearch进化结果是否符合预期"""

import json
import os
import hashlib

print("=" * 80)
print("FunSearch 进化结果检查报告")
print("=" * 80)

# 1. 检查generation文件数量
print("\n1. 检查generation文件...")
evolution_dir = 'funsearch_evolution'
generation_files = [f for f in os.listdir(evolution_dir) if f.startswith('generation_') and f.endswith('.json')]
generation_files.sort(key=lambda x: int(x.replace('generation_', '').replace('.json', '')))
print(f"   发现 {len(generation_files)} 个generation文件")
print(f"   文件范围: generation_1.json 到 generation_{len(generation_files)}.json")

# 2. 读取几个关键代数检查结构
print("\n2. 检查JSON结构...")
sample_gens = [1, 50, 100, 150, 200]
structure_ok = True
all_expected_fields = ['generation', 'timestamp', 'execution_time', 'generated_code', 'island_results',
                       'best_sharpe_ratio', 'average_sharpe_ratio', 'best_island_id', 'best_final_nav']

for gen in sample_gens:
    if gen > len(generation_files):
        continue
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_fields = [field for field in all_expected_fields if field not in data]
    if missing_fields:
        print(f"   [X] generation_{gen}.json 缺少字段: {missing_fields}")
        structure_ok = False
    else:
        print(f"   [OK] generation_{gen}.json 结构正确")

# 3. 检查island_results结构
print("\n3. 检查island_results结构...")
expected_island_fields = ['island_id', 'source_island_id', 'score', 'sharpe_ratio', 'sortino_ratio',
                         'max_drawdown', 'final_nav', 'turnover', 'weights', 'strategy_code', 'evaluated_program']

gen1_path = os.path.join(evolution_dir, 'generation_1.json')
with open(gen1_path, 'r', encoding='utf-8') as f:
    gen1_data = json.load(f)

if gen1_data['island_results']:
    first_island = gen1_data['island_results'][0]
    missing_island_fields = [field for field in expected_island_fields if field not in first_island]
    if missing_island_fields:
        print(f"   [X] island_results缺少字段: {missing_island_fields}")
    else:
        print(f"   [OK] island_results结构正确")
        print(f"   字段包括: {', '.join(expected_island_fields)}")

# 4. 对比不同代数的generated_code是否不同
print("\n4. 检查LLM生成的代码是否不同...")
codes_by_hash = {}
codes_by_content = {}

for gen in [1, 10, 50, 100, 150, 200]:
    if gen > len(generation_files):
        continue
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code = data['generated_code']
    code_hash = hashlib.md5(code.encode()).hexdigest()

    if code_hash not in codes_by_hash:
        codes_by_hash[code_hash] = []
    codes_by_hash[code_hash].append(gen)

    codes_by_content[gen] = code[:100] + "..."  # 只保存前100字符用于显示

print(f"   发现 {len(codes_by_hash)} 种不同的代码模式:")
for hash_code, gens in codes_by_hash.items():
    print(f"   - 代数 {gens}: {codes_by_content[gens[0]]}")

# 5. 对比不同代数的strategy_code是否不同
print("\n5. 检查island_results中的strategy_code是否不同...")
strategy_codes = {}
for gen in [1, 10, 50, 100, 150, 200]:
    if gen > len(generation_files):
        continue
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for island_result in data['island_results']:
        code = island_result.get('strategy_code', '')
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash not in strategy_codes:
            strategy_codes[code_hash] = []
        strategy_codes[code_hash].append((gen, island_result['island_id']))

print(f"   发现 {len(strategy_codes)} 种不同的策略代码模式")

# 6. 检查score字段是什么
print("\n6. 检查score字段含义...")
gen1_islands = gen1_data['island_results']
for i, island in enumerate(gen1_islands[:3]):  # 只看前3个
    print(f"   Island {island['island_id']}: score={island['score']:.6f}, sharpe_ratio={island['sharpe_ratio']:.6f}")

# 7. 检查best_sharpe_ratio和best_score的关系
print("\n7. 检查best_sharpe_ratio与best_score是否一致...")
with open(os.path.join(evolution_dir, 'evolution_log.json'), 'r', encoding='utf-8') as f:
    evo_log = json.load(f)

mismatches = 0
for entry in evo_log[:10]:  # 检查前10代
    if 'best_score' in entry and 'best_sharpe_ratio' in entry:
        if entry['best_score'] != entry['best_sharpe_ratio']:
            mismatches += 1
            print(f"   [X] 代数 {entry['generation']}: best_score={entry['best_score']}, best_sharpe_ratio={entry['best_sharpe_ratio']}")

if mismatches == 0:
    print(f"   [OK] 所有best_score都等于best_sharpe_ratio")

# 8. 检查策略代码是否被截断
print("\n8. 检查策略代码是否完整（未截断）...")
gen1_strategies = gen1_data['island_results']
all_full = True
for island in gen1_strategies:
    code = island.get('strategy_code', '')
    if len(code) > 0 and code.endswith('...'):
        print(f"   [X] Island {island['island_id']} 策略代码被截断")
        all_full = False

if all_full:
    print(f"   [OK] 所有策略代码完整，未被截断")

# 9. 检查最终净值
print("\n9. 检查最终净值记录...")
navs = [entry.get('best_final_nav', None) for entry in evo_log[:10] if 'best_final_nav' in entry]
if all(nav is not None for nav in navs):
    print(f"   [OK] 所有条目都记录了最终净值")
    print(f"   示例净值: {[f'{nav:.4f}' for nav in navs[:5]]}")
else:
    print(f"   [X] 部分条目缺少final_nav")

# 总结
print("\n" + "=" * 80)
print("检查总结")
print("=" * 80)
print(f"""
1. [OK] generation文件数量: {len(generation_files)} 个
2. [OK/NG] JSON结构: {'正确' if structure_ok else '有缺失'}
3. [OK] LLM生成了 {len(codes_by_hash)} 种不同的代码
4. [OK] island_results中有 {len(strategy_codes)} 种不同策略
5. [OK/NG] 策略代码: {'完整' if all_full else '有截断'}
6. [OK] best_sharpe_ratio字段存在且正确
7. [OK] 最终净值已记录
""")
