"""提取完整的最佳策略代码"""

import json

# 读取最佳策略的具体代码
best_gen = 131
best_gen_file = f'funsearch_evolution/generation_{best_gen}.json'
with open(best_gen_file, 'r', encoding='utf-8') as f:
    best_gen_data = json.load(f)

# 找到该代中得分最高的island
best_island = None
best_island_score = -float('inf')
for r in best_gen_data['island_results']:
    if r['score'] > best_island_score:
        best_island_score = r['score']
        best_island = r

print("最佳策略完整代码:\n")
print(best_island['strategy_code'])

# 保存到文件
with open('funsearch_results/best_strategy_code.txt', 'w', encoding='utf-8') as f:
    f.write(best_island['strategy_code'])
