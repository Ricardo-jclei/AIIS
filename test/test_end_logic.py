#!/usr/bin/env python3
"""
快速测试FunSearch结束逻辑
验证报告生成、图表生成和策略对比
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import funsearch_specification_enhanced as fs


def test_end_logic():
    """测试FunSearch结束逻辑"""
    print("开始测试FunSearch结束逻辑...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    print("加载测试数据...")
    # 数据已经在funsearch_specification_enhanced.py中加载
    print(f"   价格数据形状: {fs.price_array.shape}")
    print(f"   多因子数据形状: {fs.multi_factor_array.shape}")
    print(f"   股票数量: {len(fs.stock_list)}")
    
    # 2. 生成对比策略
    print("生成对比策略...")
    nav_dict = {}
    metrics_dict = {}
    
    # 等权策略
    print("   等权策略")
    weights_equal = np.ones(len(fs.stock_list)) / len(fs.stock_list)
    nav_equal, _, sr_equal, so_equal, mdd_equal = fs.backtest(weights_equal, fs.price_array)
    nav_dict['等权'] = nav_equal
    metrics_dict['等权'] = {'sharpe_ratio': sr_equal, 'sortino_ratio': so_equal, 'max_drawdown': mdd_equal}
    print(f"   夏普比率: {sr_equal:.4f}")
    
    # 最小方差策略
    print("   最小方差策略")
    weights_minvar = fs.minvar_weights(fs.price_array)
    nav_minvar, _, sr_minvar, so_minvar, mdd_minvar = fs.backtest(weights_minvar, fs.price_array)
    nav_dict['最小方差'] = nav_minvar
    metrics_dict['最小方差'] = {'sharpe_ratio': sr_minvar, 'sortino_ratio': so_minvar, 'max_drawdown': mdd_minvar}
    print(f"   夏普比率: {sr_minvar:.4f}")
    
    # 最大夏普策略
    print("   最大夏普策略")
    weights_maxsharpe = fs.maxsharpe_weights(fs.price_array)
    nav_maxsharpe, _, sr_maxsharpe, so_maxsharpe, mdd_maxsharpe = fs.backtest(weights_maxsharpe, fs.price_array)
    nav_dict['最大夏普'] = nav_maxsharpe
    metrics_dict['最大夏普'] = {'sharpe_ratio': sr_maxsharpe, 'sortino_ratio': so_maxsharpe, 'max_drawdown': mdd_maxsharpe}
    print(f"   夏普比率: {sr_maxsharpe:.4f}")
    
    # LSTM+PPO策略
    print("   LSTM+PPO策略")
    try:
        lstm_ppo_weights_result = fs.lstm_ppo_weights(fs.multi_factor_array, fs.lstm_model, fs.ppo_model, window_size=20)
        nav_lstm_ppo, _, sr_lstm_ppo, so_lstm_ppo, mdd_lstm_ppo = fs.backtest(lstm_ppo_weights_result, fs.price_array)
        nav_dict['LSTM+PPO'] = nav_lstm_ppo
        metrics_dict['LSTM+PPO'] = {'sharpe_ratio': sr_lstm_ppo, 'sortino_ratio': so_lstm_ppo, 'max_drawdown': mdd_lstm_ppo}
        print(f"   夏普比率: {sr_lstm_ppo:.4f}")
    except Exception as e:
        print(f"   LSTM+PPO失败: {e}")
    
    # 模拟FunSearch策略（使用随机权重）
    print("   FunSearch策略")
    try:
        # 生成随机权重（模拟FunSearch结果）
        weights_funsearch = np.random.rand(len(fs.stock_list))
        weights_funsearch = weights_funsearch / np.sum(weights_funsearch)
        nav_funsearch, _, sr_funsearch, so_funsearch, mdd_funsearch = fs.backtest(weights_funsearch, fs.price_array)
        nav_dict['FunSearch'] = nav_funsearch
        metrics_dict['FunSearch'] = {'sharpe_ratio': sr_funsearch, 'sortino_ratio': so_funsearch, 'max_drawdown': mdd_funsearch}
        print(f"   夏普比率: {sr_funsearch:.4f}")
    except Exception as e:
        print(f"   FunSearch失败: {e}")
    
    # 3. 生成可视化
    print("\n生成可视化...")
    try:
        fs.plot_comparison(nav_dict)
        print("   可视化生成成功")
    except Exception as e:
        print(f"   可视化失败: {e}")
    
    # 4. 生成报告
    print("\n生成报告...")
    try:
        fs.generate_report(nav_dict, metrics_dict)
        print("   报告生成成功")
    except Exception as e:
        print(f"   报告生成失败: {e}")
    
    # 5. 打印最终结果
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    
    for name, metrics in metrics_dict.items():
        print(f"\n{name}:")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  索提诺比率: {metrics['sortino_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.4f}")
    
    print("\n测试完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return nav_dict, metrics_dict


if __name__ == '__main__':
    test_end_logic()
