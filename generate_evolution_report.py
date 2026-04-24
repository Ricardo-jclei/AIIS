"""FunSearch进化结果自动分析报告生成脚本"""

import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_evolution_report():
    """生成完整的进化分析报告"""

    print("=" * 80)
    print("FunSearch 进化分析报告生成")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查日志文件
    log_file = 'funsearch_evolution/evolution_log.json'
    if not os.path.exists(log_file):
        print(f"[错误] 进化日志文件不存在: {log_file}")
        return None

    # 读取进化日志
    print("读取进化日志...")
    with open(log_file, 'r', encoding='utf-8') as f:
        evolution_log = json.load(f)

    total_generations = len(evolution_log)
    print(f"共 {total_generations} 代进化数据")

    # 创建输出目录
    output_dir = 'funsearch_results'
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 提取数据 ====================
    print("\n提取进化数据...")

    generations = []
    best_sharpe_ratios = []
    avg_sharpe_ratios = []
    best_final_navs = []
    all_island_data = []

    for entry in evolution_log:
        generations.append(entry['generation'])
        best_sharpe_ratios.append(entry.get('best_sharpe_ratio', entry.get('best_score', 0)))
        avg_sharpe_ratios.append(entry.get('average_sharpe_ratio', entry.get('average_score', 0)))
        best_final_navs.append(entry.get('best_final_nav', 1.0))

        # 收集所有island的数据
        for island_result in entry.get('island_results', []):
            all_island_data.append({
                'generation': entry['generation'],
                'island_id': island_result.get('island_id', 0),
                'source_island_id': island_result.get('source_island_id', None),
                'score': island_result.get('score', 0),
                'sharpe_ratio': island_result.get('sharpe_ratio', 0),
                'sortino_ratio': island_result.get('sortino_ratio', 0),
                'max_drawdown': island_result.get('max_drawdown', 0),
                'final_nav': island_result.get('final_nav', 1.0),
                'turnover': island_result.get('turnover', 0),
            })

    generations = np.array(generations)
    best_sharpe_ratios = np.array(best_sharpe_ratios)
    avg_sharpe_ratios = np.array(avg_sharpe_ratios)
    best_final_navs = np.array(best_final_navs)

    # ==================== 2. 统计分析 ====================
    print("进行统计分析...")

    # 历史最佳
    best_sharpe_idx = np.argmax(best_sharpe_ratios)
    best_sharpe = best_sharpe_ratios[best_sharpe_idx]
    best_sharpe_gen = generations[best_sharpe_idx]

    best_nav_idx = np.argmax(best_final_navs)
    best_nav = best_final_navs[best_nav_idx]
    best_nav_gen = generations[best_nav_idx]

    # 分阶段统计
    stages = [
        ("初期", 1, min(100, total_generations)),
        ("中期", min(101, total_generations), min(500, total_generations)),
        ("后期", min(501, total_generations), min(1000, total_generations)),
        ("末期", min(1001, total_generations), total_generations),
    ]

    stage_stats = []
    for name, start, end in stages:
        if start < len(best_sharpe_ratios):
            end_idx = min(end, len(best_sharpe_ratios))
            stage_scores = best_sharpe_ratios[start:end_idx]
            if len(stage_scores) > 0:
                stage_stats.append({
                    'name': name,
                    'avg_sharpe': np.mean(stage_scores),
                    'max_sharpe': np.max(stage_scores),
                    'min_sharpe': np.min(stage_scores),
                    'std_sharpe': np.std(stage_scores)
                })

    # ==================== 3. 生成可视化 ====================
    print("生成可视化图表...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # 图1: 夏普比率进化趋势
    ax1 = axes[0, 0]
    ax1.plot(generations, best_sharpe_ratios, 'r-', linewidth=2, label='Best Sharpe', alpha=0.8)
    ax1.plot(generations, avg_sharpe_ratios, 'b-', linewidth=2, label='Avg Sharpe', alpha=0.8)
    ax1.scatter([best_sharpe_gen], [best_sharpe], s=200, c='orange', zorder=5, marker='*', label=f'Best: {best_sharpe:.4f}')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title('Sharpe Ratio Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.1])

    # 图2: 最终净值进化趋势
    ax2 = axes[0, 1]
    ax2.plot(generations, best_final_navs, 'g-', linewidth=2, label='Best NAV')
    ax2.scatter([best_nav_gen], [best_nav], s=200, c='orange', zorder=5, marker='*', label=f'Best: {best_nav:.4f}')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Final NAV', fontsize=12)
    ax2.set_title('Final Net Asset Value Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 图3: 夏普比率分布（所有Island）
    ax3 = axes[1, 0]
    all_sharpes = [d['sharpe_ratio'] for d in all_island_data]
    ax3.hist(all_sharpes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(all_sharpes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_sharpes):.4f}')
    ax3.axvline(x=np.median(all_sharpes), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(all_sharpes):.4f}')
    ax3.set_xlabel('Sharpe Ratio', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Sharpe Ratio Distribution (All Islands)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 图4: 滚动平均趋势
    ax4 = axes[1, 1]
    window_size = min(50, len(generations) // 10)
    if len(best_sharpe_ratios) > window_size:
        rolling_avg = np.convolve(best_sharpe_ratios, np.ones(window_size)/window_size, mode='valid')
        rolling_gens = generations[window_size-1:]
        ax4.plot(rolling_gens, rolling_avg, 'purple', linewidth=3, label=f'Rolling Avg (window={window_size})')
        ax4.fill_between(rolling_gens, rolling_avg - 0.05, rolling_avg + 0.05, alpha=0.2, color='purple')
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Sharpe Ratio', fontsize=12)
    ax4.set_title('Rolling Average Sharpe Ratio', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 图5: 夏普 vs 净值散点图
    ax5 = axes[2, 0]
    all_sharpes = [d['sharpe_ratio'] for d in all_island_data]
    all_navs = [d['final_nav'] for d in all_island_data]
    scatter = ax5.scatter(all_sharpes, all_navs, alpha=0.5, s=20, c=range(len(all_sharpes)), cmap='viridis')
    ax5.set_xlabel('Sharpe Ratio', fontsize=12)
    ax5.set_ylabel('Final NAV', fontsize=12)
    ax5.set_title('Sharpe Ratio vs Final NAV', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Generation')
    ax5.grid(True, alpha=0.3)

    # 图6: 最大回撤分布
    ax6 = axes[2, 1]
    all_mdd = [d['max_drawdown'] for d in all_island_data]
    ax6.hist(all_mdd, bins=50, color='crimson', alpha=0.7, edgecolor='black')
    ax6.axvline(x=np.mean(all_mdd), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_mdd):.4f}')
    ax6.set_xlabel('Max Drawdown', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Maximum Drawdown Distribution', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/evolution_analysis_report.png', dpi=200, bbox_inches='tight')
    print(f"  保存: {output_dir}/evolution_analysis_report.png")

    # ==================== 4. 生成Markdown报告 ====================
    print("生成Markdown报告...")

    report_content = f"""# FunSearch 进化分析报告

## 1. 概览

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总进化代数**: {total_generations}
- **每代Island数量**: 10
- **总评估次数**: {total_generations * 10}

## 2. 性能统计

### 2.1 整体表现

| 指标 | 值 |
|------|-----|
| **最佳夏普比率** | {best_sharpe:.6f} |
| **最佳夏普代数** | 第 {best_sharpe_gen} 代 |
| **最佳最终净值** | {best_nav:.6f} |
| **最佳净值代数** | 第 {best_nav_gen} 代 |
| **平均夏普比率** | {np.mean(best_sharpe_ratios):.6f} |
| **夏普比率标准差** | {np.std(best_sharpe_ratios):.6f} |

### 2.2 分阶段统计

| 阶段 | 代数范围 | 平均夏普 | 最高夏普 | 最低夏普 | 标准差 |
|------|---------|---------|---------|---------|--------|
"""

    for stage in stage_stats:
        report_content += f"| {stage['name']} | {stage['avg_sharpe']:.4f} | {stage['max_sharpe']:.4f} | {stage['min_sharpe']:.4f} | {stage['std_sharpe']:.4f} |\n"

    report_content += f"""
### 2.3 所有Island统计

| 指标 | 值 |
|------|-----|
| **总策略评估数** | {len(all_island_data)} |
| **平均夏普比率** | {np.mean([d['sharpe_ratio'] for d in all_island_data]):.6f} |
| **平均最终净值** | {np.mean([d['final_nav'] for d in all_island_data]):.6f} |
| **平均最大回撤** | {np.mean([d['max_drawdown'] for d in all_island_data]):.6f} |

## 3. 进化趋势分析

### 3.1 夏普比率趋势

"""

    if len(stage_stats) >= 2:
        first_stage = stage_stats[0]
        last_stage = stage_stats[-1]
        improvement = ((last_stage['avg_sharpe'] - first_stage['avg_sharpe']) / first_stage['avg_sharpe'] * 100) if first_stage['avg_sharpe'] != 0 else 0
        report_content += f"- 从{first_stage['name']}到{last_stage['name']}，平均夏普比率变化: {improvement:+.2f}%\n"
        report_content += f"- {'进化效果良好' if improvement > 0 else '进化效果不明显或退化'}\n"

    report_content += f"""
- 最佳夏普比率出现在第 {best_sharpe_gen} 代: {best_sharpe:.6f}
- 最佳最终净值出现在第 {best_nav_gen} 代: {best_nav:.6f}

### 3.2 稳定性分析

- 夏普比率变异系数 (CV): {np.std(best_sharpe_ratios) / np.mean(best_sharpe_ratios) * 100:.2f}%
- {'进化过程相对稳定' if np.std(best_sharpe_ratios) / np.mean(best_sharpe_ratios) < 0.1 else '进化过程波动较大'}

## 4. 可视化

### 4.1 进化过程总览
![Evolution Analysis](evolution_analysis_report.png)

### 4.2 夏普比率进化趋势
- 红色线: 每代最佳夏普比率
- 蓝色线: 每代平均夏普比率
- 橙色星号: 历史最佳点

### 4.3 最终净值进化趋势
- 展示每代最佳策略的最终净值变化

### 4.4 夏普比率分布
- 展示所有Island评估结果的夏普比率分布
- 红色虚线: 均值
- 橙色虚线: 中位数

### 4.5 滚动平均趋势
- 使用滑动窗口平滑后的夏普比率趋势
- 阴影区域: ±0.05范围

### 4.6 夏普 vs 净值
- 散点图展示夏普比率和最终净值的关系
- 颜色表示代数（从暗到亮表示从初期到后期）

### 4.7 最大回撤分布
- 展示所有策略的最大回撤分布

## 5. 最佳策略详情

### 5.1 最佳夏普策略
- **代数**: {best_sharpe_gen}
- **夏普比率**: {best_sharpe:.6f}

"""

    # 查找最佳策略的详细信息
    if best_sharpe_gen <= len(evolution_log):
        best_gen_entry = evolution_log[best_sharpe_gen - 1]
        best_island = None
        best_score = -float('inf')
        for island_result in best_gen_entry.get('island_results', []):
            if island_result.get('score', 0) > best_score:
                best_score = island_result.get('score', 0)
                best_island = island_result

        if best_island:
            report_content += f"""- **Sortino比率**: {best_island.get('sortino_ratio', 0):.6f}
- **最大回撤**: {best_island.get('max_drawdown', 0):.6f}
- **最终净值**: {best_island.get('final_nav', 0):.6f}
- **换手率**: {best_island.get('turnover', 0):.6f}

### 5.2 策略代码

```
{best_island.get('strategy_code', 'N/A')[:1000]}...
```

"""

    report_content += f"""
## 6. 结论与建议

### 6.1 进化效果评估

"""

    if best_sharpe > 0.9:
        report_content += "✅ **优秀**: 进化过程成功找到了高夏普比率的策略（>0.9）\n"
    elif best_sharpe > 0.8:
        report_content += "✅ **良好**: 进化过程找到了较好夏普比率的策略（>0.8）\n"
    elif best_sharpe > 0.7:
        report_content += "⚠️ **一般**: 进化过程夏普比率在可接受范围内（>0.7）\n"
    else:
        report_content += "❌ **需改进**: 进化效果不理想，建议调整参数或策略模板\n"

    if len(stage_stats) >= 2:
        if stage_stats[-1]['avg_sharpe'] > stage_stats[0]['avg_sharpe']:
            report_content += "✅ 后期平均表现优于初期，进化有正向效果\n"
        else:
            report_content += "⚠️ 后期平均表现未明显优于初期，可能存在退化或陷入局部最优\n"

    report_content += f"""
### 6.2 建议

1. **参数调优**: 考虑调整进化参数（温度、样本数等）以提高进化效果
2. **策略模板**: 当前策略模板可能需要更多样化以促进进化
3. **提前停止**: 当性能不再显著提升时可提前停止进化节省时间
4. **集成学习**: 可考虑将多个表现良好的策略进行集成

## 7. 文件清单

- `evolution_analysis_report.png`: 进化过程可视化图表
- `evolution_log.json`: 完整进化日志
- `generation_*.json`: 每代详细数据
- `funsearch_report.md`: 本报告

---

*报告由 FunSearch 自动分析系统生成*
"""

    # 保存报告
    report_path = f'{output_dir}/funsearch_evolution_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  保存: {report_path}")

    # ==================== 5. 保存分析数据 ====================
    print("保存分析数据...")

    analysis_data = {
        'total_generations': total_generations,
        'best_sharpe': float(best_sharpe),
        'best_sharpe_gen': int(best_sharpe_gen),
        'best_nav': float(best_nav),
        'best_nav_gen': int(best_nav_gen),
        'avg_sharpe': float(np.mean(best_sharpe_ratios)),
        'std_sharpe': float(np.std(best_sharpe_ratios)),
        'stage_stats': stage_stats,
        'all_island_stats': {
            'count': len(all_island_data),
            'avg_sharpe': float(np.mean([d['sharpe_ratio'] for d in all_island_data])),
            'avg_nav': float(np.mean([d['final_nav'] for d in all_island_data])),
            'avg_mdd': float(np.mean([d['max_drawdown'] for d in all_island_data])),
        }
    }

    analysis_path = f'{output_dir}/evolution_analysis_data.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    print(f"  保存: {analysis_path}")

    # ==================== 6. 打印总结 ====================
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n📊 生成的文件:")
    print(f"  1. {output_dir}/evolution_analysis_report.png - 可视化图表")
    print(f"  2. {output_dir}/funsearch_evolution_report.md - 详细报告")
    print(f"  3. {output_dir}/evolution_analysis_data.json - 分析数据")

    print(f"\n📈 关键发现:")
    print(f"  • 最佳夏普比率: {best_sharpe:.4f} (第 {best_sharpe_gen} 代)")
    print(f"  • 最佳最终净值: {best_nav:.4f} (第 {best_nav_gen} 代)")
    print(f"  • 平均夏普比率: {np.mean(best_sharpe_ratios):.4f}")
    print(f"  • 夏普比率标准差: {np.std(best_sharpe_ratios):.4f}")

    print("\n" + "=" * 80)

    return analysis_data


if __name__ == '__main__':
    generate_evolution_report()
