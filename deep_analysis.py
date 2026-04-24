"""深度分析LLM生成的代码差异"""

import json
import os
import hashlib

print("=" * 80)
print("LLM生成代码深度分析")
print("=" * 80)

evolution_dir = 'funsearch_evolution'

# 读取第1代和第10代的完整代码进行对比
print("\n1. 对比第1代 vs 第10代 vs 第50代的代码...")

for gen in [1, 10, 50, 100]:
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code = data['generated_code']
    print(f"\n--- 代数 {gen} (长度: {len(code)} 字符) ---")
    print(code[:500])
    print("...")

# 分析所有101个文件的代码多样性
print("\n" + "=" * 80)
print("2. 分析所有101个文件的代码多样性...")
print("=" * 80)

all_codes = {}
for gen in range(1, 102):
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code = data['generated_code']
    code_hash = hashlib.md5(code.encode()).hexdigest()

    if code_hash not in all_codes:
        all_codes[code_hash] = []
    all_codes[code_hash].append(gen)

print(f"\n发现 {len(all_codes)} 种完全不同的代码")

# 找出哪些代数的代码是相同的
print("\n相同代码的代数分组:")
for hash_code, gens in sorted(all_codes.items(), key=lambda x: x[1][0]):
    if len(gens) > 1:
        print(f"  - 代数 {gens}: 相同代码 (共 {len(gens)} 个)")
    else:
        print(f"  - 代数 {gens}: 独特代码")

# 分析代码长度变化
print("\n" + "=" * 80)
print("3. 代码长度随代数变化...")
print("=" * 80)

lengths = []
for gen in range(1, 102):
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code = data['generated_code']
    lengths.append((gen, len(code)))

print(f"最短代码: 代数 {min(lengths, key=lambda x: x[1])[0]}, {min(lengths, key=lambda x: x[1])[1]} 字符")
print(f"最长代码: 代数 {max(lengths, key=lambda x: x[1])[0]}, {max(lengths, key=lambda x: x[1])[1]} 字符")
print(f"平均长度: {sum(x[1] for x in lengths) / len(lengths):.1f} 字符")

# 分析关键特征差异
print("\n" + "=" * 80)
print("4. 分析代码中的关键特征差异...")
print("=" * 80)

keywords = ['trend', 'momentum', 'mean-reversion', 'volatility', 'inverse', 'risk', 'weighted']
for keyword in keywords:
    count = 0
    for gen in range(1, 102):
        file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        code = data['generated_code'].lower()
        if keyword.lower() in code:
            count += 1
    print(f"  '{keyword}' 出现在 {count}/101 代中")

# 查看第1代island_results中的strategy_code
print("\n" + "=" * 80)
print("5. 第1代island_results中的strategy_code分析...")
print("=" * 80)

gen1_path = os.path.join(evolution_dir, 'generation_1.json')
with open(gen1_path, 'r', encoding='utf-8') as f:
    gen1_data = json.load(f)

print(f"第1代有 {len(gen1_data['island_results'])} 个island结果")
for i, island in enumerate(gen1_data['island_results'][:3]):
    print(f"\nisland {i}:")
    print(f"  island_id: {island['island_id']}")
    print(f"  source_island_id: {island['source_island_id']}")
    print(f"  score: {island['score']:.6f}")
    print(f"  sharpe_ratio: {island['sharpe_ratio']:.6f}")
    print(f"  final_nav: {island['final_nav']:.2f}")
    print(f"  strategy_code长度: {len(island['strategy_code'])} 字符")
    print(f"  strategy_code前100字符: {island['strategy_code'][:100]}...")

# 验证source_island_id的含义
print("\n" + "=" * 80)
print("6. 验证source_island_id的含义...")
print("=" * 80)

print("分析source_island_id分布:")
source_ids = {}
for gen in range(1, 102):
    file_path = os.path.join(evolution_dir, f'generation_{gen}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for island in data['island_results']:
        sid = island['source_island_id']
        if sid not in source_ids:
            source_ids[sid] = 0
        source_ids[sid] += 1

for sid in sorted(source_ids.keys()):
    print(f"  source_island_id={sid}: {source_ids[sid]} 次")
