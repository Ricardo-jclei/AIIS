"""从进化日志中提取最佳策略并重新评估"""

import json
import os
import sys
import numpy as np
import types

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import funsearch_specification_enhanced as fs

print("=" * 80)
print("FunSearch 最佳策略重评估")
print("=" * 80)

# 读取分析数据
with open('funsearch_evolution/analysis.json', 'r', encoding='utf-8') as f:
    analysis = json.load(f)

print(f"\n历史最佳策略信息:")
print(f"  代数: {analysis['best_gen']}")
print(f"  得分: {analysis['best_score']:.6f}")

# 读取最佳策略的具体代码
best_gen_file = f'funsearch_evolution/generation_{analysis["best_gen"]}.json'
with open(best_gen_file, 'r', encoding='utf-8') as f:
    best_gen_data = json.load(f)

# 找到该代中得分最高的island
best_island = None
best_island_score = -float('inf')
for r in best_gen_data['island_results']:
    if r['score'] > best_island_score:
        best_island_score = r['score']
        best_island = r

print(f"  最佳 Island: {best_island['island_id']}")
print(f"  该 Island 得分: {best_island['score']:.6f}")

# 评估所有历史top策略
print(f"\n\n开始评估历史Top 10策略...")

top_strategies = analysis['top_strategies']
evaluated_strategies = []

for i, strategy_info in enumerate(top_strategies):
    gen = strategy_info['gen']
    island = strategy_info['island']
    score = strategy_info['score']
    
    print(f"\n--- 评估 Top {i+1}: Gen {gen}, Island {island} (FunSearch得分: {score:.6f}) ---")
    
    # 读取该代的具体策略代码
    gen_file = f'funsearch_evolution/generation_{gen}.json'
    try:
        with open(gen_file, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)
        
        # 找到该island的策略代码
        strategy_code = None
        for r in gen_data['island_results']:
            if r['island_id'] == island:
                strategy_code = r['strategy_code']
                break
        
        if strategy_code is None:
            print(f"  [警告] 未找到该Island的策略代码")
            continue
        
        # 创建临时模块并执行
        temp_module = types.ModuleType('temp_strategy')
        full_code = f"""
import numpy as np

def candidate_strategy(market_state, portfolio):
{strategy_code}
        """
        
        exec(full_code, temp_module.__dict__)
        
        # 使用策略进行回测
        weights_list = []
        window_size = 20
        
        for t in range(window_size, len(fs.multi_factor_array)):
            try:
                weights_t = temp_module.candidate_strategy(fs.multi_factor_array[t-window_size:t], None)
                # 确保权重有效
                if weights_t is None:
                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                elif not isinstance(weights_t, np.ndarray):
                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                elif len(weights_t) != len(fs.stock_list):
                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                elif np.any(np.isnan(weights_t)) or np.any(np.isinf(weights_t)):
                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                # 确保权重归一化
                weights_t = np.clip(weights_t, 0, 1)
                weights_sum = np.sum(weights_t)
                if weights_sum == 0:
                    weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                else:
                    weights_t = weights_t / weights_sum
                weights_list.append(weights_t)
            except Exception as e:
                weights_t = np.ones(len(fs.stock_list)) / len(fs.stock_list)
                weights_list.append(weights_t)
        
        # 计算平均权重
        if weights_list:
            weights = np.mean(np.array(weights_list), axis=0)
            # 确保权重归一化
            weights = np.clip(weights, 0, 1)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                weights = np.ones(len(fs.stock_list)) / len(fs.stock_list)
            else:
                weights = weights / weights_sum
            
            # 回测
            nav, _, sr, so, mdd = fs.backtest(weights, fs.price_array)
            
            # 综合得分
            combined_score = sr - abs(mdd) * 0.5
            
            print(f"  回测结果:")
            print(f"    夏普比率: {sr:.6f}")
            print(f"    索提诺比率: {so:.6f}")
            print(f"    最大回撤: {mdd:.6f}")
            print(f"    综合得分: {combined_score:.6f}")
            
            evaluated_strategies.append({
                'rank': i+1,
                'gen': gen,
                'island': island,
                'funsearch_score': score,
                'sharpe': sr,
                'sortino': so,
                'mdd': mdd,
                'combined_score': combined_score,
                'weights': weights,
                'nav': nav,
                'strategy_code': strategy_code
            })
        
    except Exception as e:
        print(f"  [错误] 评估失败: {e}")
        import traceback
        traceback.print_exc()

# 找出真正最优的
print(f"\n\n" + "=" * 80)
print("策略评估结果汇总")
print("=" * 80)

print(f"\n按 FunSearch得分 排名:")
for s in sorted(evaluated_strategies, key=lambda x: x['funsearch_score'], reverse=True):
    print(f"  Gen {s['gen']}, Island {s['island']}:")
    print(f"    FunSearch: {s['funsearch_score']:.6f}, 回测夏普: {s['sharpe']:.6f}")

print(f"\n按 回测夏普 排名:")
for s in sorted(evaluated_strategies, key=lambda x: x['sharpe'], reverse=True):
    print(f"  Gen {s['gen']}, Island {s['island']}:")
    print(f"    回测夏普: {s['sharpe']:.6f}, FunSearch: {s['funsearch_score']:.6f}")

print(f"\n按 综合得分 排名:")
for s in sorted(evaluated_strategies, key=lambda x: x['combined_score'], reverse=True):
    print(f"  Gen {s['gen']}, Island {s['island']}:")
    print(f"    综合: {s['combined_score']:.6f}, 夏普: {s['sharpe']:.6f}, 回撤: {s['mdd']:.6f}")

# 选择最终策略 - 按FunSearch得分
best_by_funsearch = max(evaluated_strategies, key=lambda x: x['funsearch_score'])
best_by_sharpe = max(evaluated_strategies, key=lambda x: x['sharpe'])
best_by_combined = max(evaluated_strategies, key=lambda x: x['combined_score'])

print(f"\n\n" + "=" * 80)
print("最终建议")
print("=" * 80)
print(f"\n按FunSearch得分最佳:")
print(f"  Gen {best_by_funsearch['gen']}, Island {best_by_funsearch['island']}")
print(f"  FunSearch得分: {best_by_funsearch['funsearch_score']:.6f}")
print(f"  回测夏普: {best_by_funsearch['sharpe']:.6f}")

print(f"\n按回测夏普最佳:")
print(f"  Gen {best_by_sharpe['gen']}, Island {best_by_sharpe['island']}")
print(f"  回测夏普: {best_by_sharpe['sharpe']:.6f}")
print(f"  FunSearch得分: {best_by_sharpe['funsearch_score']:.6f}")

# 使用按FunSearch得分最佳的策略作为最终结果
selected_strategy = best_by_funsearch

# 创建对比数据
nav_dict = {}
metrics_dict = {}

# 等权策略
weights_equal = np.ones(len(fs.stock_list)) / len(fs.stock_list)
nav_equal, _, sr_equal, so_equal, mdd_equal = fs.backtest(weights_equal, fs.price_array)
nav_dict['等权'] = nav_equal
metrics_dict['等权'] = {'sharpe_ratio': sr_equal, 'sortino_ratio': so_equal, 'max_drawdown': mdd_equal}

# 最小方差策略
weights_minvar = fs.minvar_weights(fs.price_array)
nav_minvar, _, sr_minvar, so_minvar, mdd_minvar = fs.backtest(weights_minvar, fs.price_array)
nav_dict['最小方差'] = nav_minvar
metrics_dict['最小方差'] = {'sharpe_ratio': sr_minvar, 'sortino_ratio': so_minvar, 'max_drawdown': mdd_minvar}

# 最大夏普策略
weights_maxsharpe = fs.maxsharpe_weights(fs.price_array)
nav_maxsharpe, _, sr_maxsharpe, so_maxsharpe, mdd_maxsharpe = fs.backtest(weights_maxsharpe, fs.price_array)
nav_dict['最大夏普'] = nav_maxsharpe
metrics_dict['最大夏普'] = {'sharpe_ratio': sr_maxsharpe, 'sortino_ratio': so_maxsharpe, 'max_drawdown': mdd_maxsharpe}

# LSTM+PPO动态RL策略
try:
    nav_lstmppo, sr_lstmppo, so_lstmppo, _ = fs.lstm_ppo_dynamic_backtest(fs.price_array, fs.multi_factor_array, fs.ppo_model, window_size=20, sharpe_window=20, lstm_input_size=fs.multi_factor_array.shape[1])
    peak = np.maximum.accumulate(nav_lstmppo)
    mdd_lstmppo = np.min((nav_lstmppo - peak) / (peak + 1e-8))
    nav_dict['LSTM+PPO动态RL'] = nav_lstmppo
    metrics_dict['LSTM+PPO动态RL'] = {'sharpe_ratio': sr_lstmppo, 'sortino_ratio': so_lstmppo, 'max_drawdown': mdd_lstmppo}
except Exception as e:
    print(f"[警告] LSTM+PPO动态RL回测失败: {e}")

# 添加FunSearch策略
nav_dict['FunSearch'] = selected_strategy['nav']
metrics_dict['FunSearch'] = {'sharpe_ratio': selected_strategy['sharpe'], 'sortino_ratio': selected_strategy['sortino'], 'max_drawdown': selected_strategy['mdd']}

# 可视化
print(f"\n\n生成可视化...")
fs.plot_comparison(nav_dict)

# 生成报告
fs.generate_report(nav_dict, metrics_dict)

# 保存最佳策略代码
best_strategy_file = 'funsearch_results/best_strategy.py'
with open(best_strategy_file, 'w', encoding='utf-8') as f:
    f.write('"""FunSearch发现的最佳策略"""\n')
    f.write('import numpy as np\n\n')
    f.write('def best_strategy(market_state, portfolio):\n')
    f.write(selected_strategy['strategy_code'])
print(f"最佳策略代码已保存到: {best_strategy_file}")

# 打印最终结果
print(f"\n\n" + "=" * 60)
print("最终对比结果")
print("=" * 60)
for name, metrics in metrics_dict.items():
    print(f"\n{name}:")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.6f}")
    print(f"  索提诺比率: {metrics['sortino_ratio']:.6f}")
    print(f"  最大回撤: {metrics['max_drawdown']:.6f}")

print(f"\n结果已保存到: funsearch_results/")
