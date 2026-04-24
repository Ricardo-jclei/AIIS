#!/usr/bin/env python3
"""
验证对比策略的真实性
确保所有策略使用相同的数据和回测逻辑
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import funsearch_specification_enhanced as fs


def validate_comparison():
    """验证对比策略的真实性"""
    print("开始验证对比策略的真实性...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 验证数据一致性
    print("\n1. 验证数据一致性...")
    print(f"价格数据形状: {fs.price_array.shape}")
    print(f"多因子数据形状: {fs.multi_factor_array.shape}")
    print(f"股票列表: {fs.stock_list}")
    print(f"数据长度: {len(fs.price_array)} 时间步")
    
    # 2. 验证回测参数一致性
    print("\n2. 验证回测参数一致性...")
    initial_cash = 1e7
    fee_rate = 0.001
    slippage_rate = 0.0
    print(f"初始资金: {initial_cash:.2f}")
    print(f"交易费率: {fee_rate:.4f}")
    print(f"滑点费率: {slippage_rate:.4f}")
    
    # 3. 运行所有对比策略
    print("\n3. 运行所有对比策略...")
    strategies = {}
    
    # 等权策略
    print("   等权策略")
    weights_equal = np.ones(len(fs.stock_list)) / len(fs.stock_list)
    nav_equal, returns_equal, sr_equal, so_equal, mdd_equal = fs.backtest(
        weights_equal, fs.price_array, initial_cash, fee_rate, slippage_rate
    )
    strategies['等权'] = {
        'weights': weights_equal,
        'nav': nav_equal,
        'returns': returns_equal,
        'sharpe_ratio': sr_equal,
        'sortino_ratio': so_equal,
        'max_drawdown': mdd_equal
    }
    print(f"   夏普比率: {sr_equal:.4f}")
    
    # 最小方差策略
    print("   最小方差策略")
    weights_minvar = fs.minvar_weights(fs.price_array)
    nav_minvar, returns_minvar, sr_minvar, so_minvar, mdd_minvar = fs.backtest(
        weights_minvar, fs.price_array, initial_cash, fee_rate, slippage_rate
    )
    strategies['最小方差'] = {
        'weights': weights_minvar,
        'nav': nav_minvar,
        'returns': returns_minvar,
        'sharpe_ratio': sr_minvar,
        'sortino_ratio': so_minvar,
        'max_drawdown': mdd_minvar
    }
    print(f"   夏普比率: {sr_minvar:.4f}")
    
    # 最大夏普策略
    print("   最大夏普策略")
    weights_maxsharpe = fs.maxsharpe_weights(fs.price_array)
    nav_maxsharpe, returns_maxsharpe, sr_maxsharpe, so_maxsharpe, mdd_maxsharpe = fs.backtest(
        weights_maxsharpe, fs.price_array, initial_cash, fee_rate, slippage_rate
    )
    strategies['最大夏普'] = {
        'weights': weights_maxsharpe,
        'nav': nav_maxsharpe,
        'returns': returns_maxsharpe,
        'sharpe_ratio': sr_maxsharpe,
        'sortino_ratio': so_maxsharpe,
        'max_drawdown': mdd_maxsharpe
    }
    print(f"   夏普比率: {sr_maxsharpe:.4f}")
    
    # LSTM+PPO策略
    print("   LSTM+PPO策略")
    try:
        lstm_ppo_weights_result = fs.lstm_ppo_weights(fs.multi_factor_array, fs.lstm_model, fs.ppo_model, window_size=20)
        nav_lstm_ppo, returns_lstm_ppo, sr_lstm_ppo, so_lstm_ppo, mdd_lstm_ppo = fs.backtest(
            lstm_ppo_weights_result, fs.price_array, initial_cash, fee_rate, slippage_rate
        )
        strategies['LSTM+PPO'] = {
            'weights': lstm_ppo_weights_result,
            'nav': nav_lstm_ppo,
            'returns': returns_lstm_ppo,
            'sharpe_ratio': sr_lstm_ppo,
            'sortino_ratio': so_lstm_ppo,
            'max_drawdown': mdd_lstm_ppo
        }
        print(f"   夏普比率: {sr_lstm_ppo:.4f}")
    except Exception as e:
        print(f"   LSTM+PPO失败: {e}")
    
    # 4. 验证结果一致性
    print("\n4. 验证结果一致性...")
    all_nav_lengths = []
    for name, data in strategies.items():
        nav_length = len(data['nav'])
        all_nav_lengths.append(nav_length)
        print(f"   {name}: 净值长度 = {nav_length}")
    
    # 检查所有净值长度是否一致
    if len(set(all_nav_lengths)) == 1:
        print("   所有策略的净值长度一致")
    else:
        print("   策略的净值长度不一致")
    
    # 5. 验证权重计算
    print("\n5. 验证权重计算...")
    for name, data in strategies.items():
        weights = data['weights']
        weights_sum = np.sum(weights)
        print(f"   {name}: 权重和 = {weights_sum:.4f}, 权重形状 = {weights.shape}")
        if abs(weights_sum - 1.0) < 1e-6:
            print(f"   {name} 权重归一化正确")
        else:
            print(f"   {name} 权重未正确归一化")
    
    # 6. 生成验证报告
    print("\n6. 生成验证报告...")
    output_dir = 'funsearch_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存验证结果
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': fs.price_array.shape,
        'stock_list': fs.stock_list,
        'backtest_params': {
            'initial_cash': initial_cash,
            'fee_rate': fee_rate,
            'slippage_rate': slippage_rate
        },
        'strategies': {}
    }
    
    for name, data in strategies.items():
        validation_results['strategies'][name] = {
            'sharpe_ratio': data['sharpe_ratio'],
            'sortino_ratio': data['sortino_ratio'],
            'max_drawdown': data['max_drawdown'],
            'final_nav': data['nav'][-1] if len(data['nav']) > 0 else 0,
            'weights_sum': float(np.sum(data['weights']))
        }
    
    import json
    with open(os.path.join(output_dir, 'comparison_validation.json'), 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    # 7. 打印验证结果
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    
    for name, data in strategies.items():
        print(f"\n{name}:")
        print(f"  夏普比率: {data['sharpe_ratio']:.4f}")
        print(f"  索提诺比率: {data['sortino_ratio']:.4f}")
        print(f"  最大回撤: {data['max_drawdown']:.4f}")
        print(f"  最终净值: {data['nav'][-1]:.2f}")
    
    print("\n验证完成！")
    print(f"验证报告已保存到: {output_dir}/comparison_validation.json")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return strategies


if __name__ == '__main__':
    validate_comparison()
