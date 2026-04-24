"""可视化FunSearch进化过程"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取分析数据
with open('funsearch_evolution/analysis.json', 'r', encoding='utf-8') as f:
    analysis = json.load(f)

# 读取进化日志
with open('funsearch_evolution/evolution_log.json', 'r', encoding='utf-8') as f:
    evolution_log = json.load(f)

# 提取数据
generations = []
best_scores = []
average_scores = []
all_island_scores = []

for entry in evolution_log:
    gen = entry['generation']
    generations.append(gen)
    best_scores.append(entry['best_score'])
    average_scores.append(entry['average_score'])
    
    # 收集所有island得分
    for r in entry['island_results']:
        all_island_scores.append({'gen': gen, 'score': r['score']})

# 创建可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1：进化得分趋势
ax1 = axes[0]
ax1.plot(generations, best_scores, label='Best Score (per Gen)', linewidth=2, color='#d62728', alpha=0.8)
ax1.plot(generations, average_scores, label='Average Score (per Gen)', linewidth=2, color='#1f77b4', alpha=0.8)

# 标记最佳点
best_gen = analysis['best_gen']
best_score = analysis['best_score']
ax1.scatter(best_gen, best_score, s=200, c='#ff7f0e', zorder=5, label=f'Best (Gen {best_gen}: {best_score:.6f})')
ax1.axhline(y=best_score, color='#ff7f0e', linestyle='--', alpha=0.7)
ax1.axvline(x=best_gen, color='#ff7f0e', linestyle='--', alpha=0.7)

ax1.set_title('FunSearch Evolution Progress', fontsize=16, fontweight='bold')
ax1.set_xlabel('Generation', fontsize=12)
ax1.set_ylabel('Score (Sharpe Ratio)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.6, 1.0])

# 图2：散点图（所有island得分）
ax2 = axes[1]
# 散点图
x_scatter = [s['gen'] for s in all_island_scores]
y_scatter = [s['score'] for s in all_island_scores]
scatter = ax2.scatter(x_scatter, y_scatter, alpha=0.3, s=20, c='#9467bd')

# 滚动平均（平滑线）
window_size = 50
if len(best_scores) > window_size:
    rolling_avg = np.convolve(best_scores, np.ones(window_size)/window_size, mode='valid')
    rolling_gens = generations[window_size-1:]
    ax2.plot(rolling_gens, rolling_avg, linewidth=3, color='#17becf', label=f'Rolling Avg ({window_size} gen)')

ax2.scatter(best_gen, best_score, s=200, c='#ff7f0e', zorder=5, label=f'Best (Gen {best_gen})')
ax2.set_title('All Island Scores', fontsize=16, fontweight='bold')
ax2.set_xlabel('Generation', fontsize=12)
ax2.set_ylabel('Score (Sharpe Ratio)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.6, 1.0])

plt.tight_layout()
plt.savefig('funsearch_results/evolution_analysis.png', dpi=200, bbox_inches='tight')
print("可视化已保存到: funsearch_results/evolution_analysis.png")

# 分阶段分析
print("\n=== 分阶段性能分析 ===")
stages = [
    ("初期 (1-100代)", 0, 100),
    ("中期 (101-500代)", 100, 500),
    ("后期 (501-1000代)", 500, 1000),
    ("末期 (1001-2000代)", 1000, 2000),
]

for name, start, end in stages:
    if start < len(best_scores):
        end_idx = min(end, len(best_scores))
        stage_scores = best_scores[start:end_idx]
        print(f"{name}:")
        print(f"  平均分: {np.mean(stage_scores):.6f}")
        print(f"  最高分: {np.max(stage_scores):.6f}")
        print(f"  最低分: {np.min(stage_scores):.6f}")
        print()
