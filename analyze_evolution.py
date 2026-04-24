"""分析FunSearch进化日志"""

import json
import os
import numpy as np
from datetime import datetime

# 读取进化日志
with open('funsearch_evolution/evolution_log.json', 'r', encoding='utf-8') as f:
    evolution_log = json.load(f)

print(f"共 {len(evolution_log)} 代进化数据")

# 提取数据
generations = []
best_scores = []
average_scores = []
all_scores_history = []

for entry in evolution_log:
    gen = entry['generation']
    generations.append(gen)
    best_scores.append(entry['best_score'])
    average_scores.append(entry['average_score'])
    
    # 收集每代所有island的得分
    island_scores = [r['score'] for r in entry['island_results']]
    all_scores_history.extend([(gen, s) for s in island_scores])

print(f"\n=== 整体统计 ===")
print(f"总进化代数: {len(generations)}")
print(f"历史最高得分: {max(best_scores):.6f}")
print(f"历史最低得分: {min(best_scores):.6f}")
print(f"平均得分: {np.mean(best_scores):.6f}")

# 找到得分最高的代数
max_score_idx = np.argmax(best_scores)
best_gen = generations[max_score_idx]
best_score = best_scores[max_score_idx]

print(f"\n=== 最佳表现 ===")
print(f"最佳代数: 第 {best_gen} 代")
print(f"最高分: {best_score:.6f}")

# 检查该代的island结果
best_gen_entry = evolution_log[max_score_idx]
print(f"\n=== 第 {best_gen} 代的所有Island结果 ===")
for r in best_gen_entry['island_results']:
    print(f"Island {r['island_id']}: {r['score']:.6f}")

# 找到历史上所有得分最高的策略
print(f"\n=== 历史上得分最高的策略记录 ===")
all_scores = []
for entry in evolution_log:
    for r in entry['island_results']:
        all_scores.append({
            'gen': entry['generation'],
            'island': r['island_id'],
            'score': r['score'],
            'code': r['strategy_code']
        })

# 按得分排序
all_scores.sort(key=lambda x: x['score'], reverse=True)
top10 = all_scores[:10]
for i, s in enumerate(top10):
    print(f"Top {i+1}: Gen {s['gen']}, Island {s['island']}, Score {s['score']:.6f}")

# 保存分析结果
analysis = {
    'total_generations': len(generations),
    'best_gen': best_gen,
    'best_score': best_score,
    'all_scores': best_scores,
    'top_strategies': top10
}

with open('funsearch_evolution/analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"\n分析完成，结果已保存到 funsearch_evolution/analysis.json")

# 创建最佳策略的代码
best_strategy = top10[0]
print(f"\n=== 最佳策略代码（Gen {best_strategy['gen']}, Island {best_strategy['island']}, Score {best_strategy['score']:.6f}） ===")
print(best_strategy['code'])
