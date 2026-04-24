"""Enhanced FunSearch specification with full evaluation and visualization."""

import numpy as np
import torch
import os
import sys
import pandas as pd
import matplotlib
import json
import platform
import yaml
from datetime import datetime

# ====== 0. Set matplotlib font to avoid Colab warnings ======
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model.enhanced_lstm import EnhancedLSTMModel
from src.eval.metrics import sharpe_ratio, sortino_ratio, max_drawdown
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv
from stable_baselines3 import PPO


# ====== 1. 加载LSTM模型 ======
lstm_model_path = 'model_ckpt/best_lstm_multi_asset.pth'
lstm_input_size = 224
lstm_hidden_size = 128
lstm_num_layers = 3
lstm_output_size = 224
lstm_model = EnhancedLSTMModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size, 0.3)
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location='cpu'))
lstm_model.eval()


# ====== 2. 加载数据 ======
stock_list = ['600519', '600030', '600036', '601318', '601988']
mf_data_dir_tpl = 'data/processed/{}/multi_factor.csv'
mf_dfs = []
for code in stock_list:
    path = mf_data_dir_tpl.format(code)
    df = pd.read_csv(path)
    df = df.rename(columns={col: f'{code}_{col}' for col in df.columns if col != '日期'})
    mf_dfs.append(df)
mf_df = mf_dfs[0]
for df in mf_dfs[1:]:
    mf_df = mf_df.merge(df, on='日期', how='inner')
multi_factor_array = mf_df.drop(columns=['日期']).values

# 加载价格数据
data_dir_tpl = 'data/processed/{}/market/daily/20220425_20250424_processed.csv'
price_dfs = []
for code in stock_list:
    path = data_dir_tpl.format(code)
    if not os.path.exists(path):
        # 尝试其他可能的路径
        alt_path = f'data/processed/{code}/20220425_20250424_processed.csv'
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"价格数据文件不存在: {path}")
    df = pd.read_csv(path, usecols=['日期', '收盘'])
    df = df.rename(columns={'收盘': code})
    price_dfs.append(df)

# 合并价格数据
price_df = price_dfs[0]
for df in price_dfs[1:]:
    price_df = price_df.merge(df, on='日期', how='inner')

# 确保数据有效
if price_df.empty:
    raise ValueError("价格数据合并后为空")

price_array = price_df[stock_list].values
print(f"价格数据加载完成，形状: {price_array.shape}")


# ====== 3. 定义funsearch装饰器 ======
class funsearch:
    @staticmethod
    def run(func):
        return func
    
    @staticmethod
    def evolve(func):
        return func


# ====== 4. 回测函数（与user_demo.py一致） ======
def backtest(weights, price_array, initial_cash=1e7, fee_rate=0.001, slippage_rate=0.0):
    """
    Backtest strategy and return NAV and metrics.
    
    Args:
        weights: Strategy weights
        price_array: Price array
        initial_cash: Initial cash
        fee_rate: Transaction fee rate
        slippage_rate: Slippage rate
    
    Returns:
        nav: Net asset value
        returns: Returns
        sr: Sharpe ratio
        so: Sortino ratio
        mdd: Max drawdown
    """
    # 验证输入
    if price_array is None or len(price_array) == 0:
        return np.array([initial_cash]), np.array([]), 0.0, 0.0, 0.0
    
    if weights is None:
        weights = np.ones(price_array.shape[1]) / price_array.shape[1]
    
    # 确保权重归一化
    weights = np.clip(weights, 0, 1)
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights_sum
    
    nav = [initial_cash]
    
    for t in range(1, len(price_array)):
        try:
            # 计算收益率
            price_change = (price_array[t] - price_array[t-1]) / (price_array[t-1] + 1e-8)
            ret = np.dot(price_change, weights)
            
            # 调试：检查 ret 是否有效
            if isinstance(ret, (np.ndarray, list)):
                if hasattr(ret, 'shape') and len(ret.shape) > 0 and ret.shape[0] > 1:
                    print(f"[backtest] ret is array with shape {ret.shape} at t={t}，使用第一个元素")
                    ret = ret[0]
            
            # 应用滑点
            if slippage_rate > 0:
                # 假设滑点对收益率的影响
                ret = ret * (1 - slippage_rate)
            
            nav_value = nav[-1] * (1 + ret)
            
            # 调试：检查 nav_value 是否有效
            if isinstance(nav_value, (np.ndarray, list)):
                print(f"[backtest] nav_value is array with shape {nav_value.shape} at t={t}，使用标量值")
                nav_value = float(nav_value.flatten()[0]) if hasattr(nav_value, 'size') and nav_value.size > 0 else nav[-1]
            
            if np.isnan(nav_value) or np.isinf(nav_value):
                print(f"[backtest] nav_value is NaN or Inf at t={t}，使用前一个值")
                nav_value = nav[-1]
            
            nav.append(nav_value)
        except Exception as e:
            print(f"[backtest] 计算错误: {e} at t={t}，使用前一个值")
            nav.append(nav[-1])

    nav = np.array(nav)
    returns = (nav[1:] - nav[:-1]) / nav[:-1] if len(nav) > 1 else np.array([])

    # Calculate metrics
    sr = sharpe_ratio(returns) if len(returns) > 0 else 0.0
    so = sortino_ratio(returns) if len(returns) > 0 else 0.0
    mdd = max_drawdown(nav) if len(nav) > 0 else 0.0

    return nav, returns, sr, so, mdd


# ====== 5. 基准策略（与user_demo.py一致） ======
def minvar_weights(price_array):
    """Minimum variance portfolio."""
    returns = (price_array[1:] - price_array[:-1]) / price_array[:-1]
    cov = np.cov(returns.T)
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ np.ones(len(stock_list))
    w = w / np.sum(w)
    return w

def maxsharpe_weights(price_array):
    """Maximum Sharpe ratio portfolio."""
    returns = (price_array[1:] - price_array[:-1]) / price_array[:-1]
    mean_ret = np.mean(returns, axis=0)
    cov = np.cov(returns.T)
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mean_ret
    w = w / np.sum(w)
    return w

# ====== 5.1 加载LSTM+PPO模型 ======
ppo_results_path = 'output/ppo_reward_compare/ppo_reward_compare_results.csv'
ppo_model_path = 'model_ckpt/ppo_multi_asset_sharpe_turnover_penalty.zip'  # 默认
if os.path.exists(ppo_results_path):
    df = pd.read_csv(ppo_results_path)
    if 'Sharpe' in df.columns and 'weight_path' in df.columns:
        idx = df['Sharpe'].astype(float).idxmax()
        best_weight = df.loc[idx, 'weight_path']
        if isinstance(best_weight, str) and os.path.exists(best_weight):
            ppo_model_path = best_weight
ppo_model = PPO.load(ppo_model_path)

# ====== 5.2 LSTM+PPO静态权重（与user_demo.py一致） ======
def lstm_ppo_weights(multi_factor_array, lstm_model, ppo_model, window_size=20):
    """LSTM+PPO静态权重（仅用于展示）"""
    N = 5
    position = np.zeros(N)
    cash = 1e7
    weights_list = []
    for t in range(window_size, len(multi_factor_array)):
        window = multi_factor_array[t-window_size:t]
        window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
        X = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            feats = lstm_model(X).cpu().numpy().flatten()
        prices = price_array[t-1]
        total_value = cash + np.sum(position * prices)
        position_ratio = (position * prices) / (total_value + 1e-8)
        cash_ratio = cash / (total_value + 1e-8)
        state = np.concatenate([feats, position_ratio, [cash_ratio]]).astype(np.float32)
        action, _ = ppo_model.predict(state, deterministic=True)
        w = np.clip(action, 0, 1)
        w = w / (np.sum(w) + 1e-8)
        weights_list.append(w)
    weights = np.mean(np.array(weights_list), axis=0)
    return weights

# ====== 5.3 LSTM+PPO动态RL（与user_demo.py一致） ======
def lstm_ppo_dynamic_backtest(price_array, feature_array, ppo_model, window_size=20, sharpe_window=20, lstm_input_size=224):
    """LSTM+PPO动态RL推理（真实RL策略回测）"""
    config_path = os.path.join(os.path.dirname(__file__), 'src/config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '../src/config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    fee_rate = config.get('fee_rate', 0.001)
    slippage_rate = config.get('slippage_rate', 0.0)
    enable_slippage = config.get('enable_slippage', False)
    
    env = MultiAssetTradingEnv(
        price_array=price_array,
        feature_array=feature_array,
        window_size=window_size,
        sharpe_window=sharpe_window,
        reward_type='sharpe_turnover_penalty',
        lstm_input_size=lstm_input_size,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        enable_slippage=enable_slippage
    )
    obs, info = env.reset()
    nav = [env.initial_cash]
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        nav.append(env.portfolio_value)
        done = terminated or truncated
    nav = np.array(nav)
    returns = (nav[1:] - nav[:-1]) / nav[:-1]
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    calmar = -max_drawdown(nav)
    return nav, sharpe, sortino, calmar


# ====== 6. 增强版评估函数 ======
# 用于存储一轮中所有island的评估结果
island_results = []

@funsearch.run
def evaluate_strategy(test_input: dict) -> dict:
    """
    Enhanced evaluation function that returns comprehensive metrics.

    Args:
        test_input: Contains 'strategy_func', 'window_size', and 'island_id'

    Returns:
        Dictionary containing all performance metrics:
        - sharpe_ratio: Sharpe ratio (used for evolution)
        - sortino_ratio: Sortino ratio
        - max_drawdown: Maximum drawdown
        - final_nav: Final net asset value
        - turnover: Portfolio turnover rate
        - weights: Average portfolio weights
    """
    strategy_func = test_input['strategy_func']
    window_size = test_input['window_size']
    island_id = test_input.get('island_id', 0)
    
    # 只在第一个island打印策略信息
    if island_id == 0:
            print(f"[Evaluator] 策略信息:")
            print(f"   策略函数: {strategy_func.__name__}")
            print(f"   策略模块: {strategy_func.__module__}")
            # 重置结果列表
            global island_results
            island_results = []
    
    # 使用进化策略进行回测
    # 生成策略权重序列
    weights_list = []
    error_count = 0
    max_errors = 5  # 最大错误打印次数
    fallback_count = 0  # 跟踪等权回退次数
    
    for t in range(window_size, len(multi_factor_array)):
        try:
            weights_t = strategy_func(multi_factor_array[t-window_size:t], None)
            
            # 检查 weights_t 是否有效
            if weights_t is None:
                if error_count < max_errors:
                    print(f"[evaluate_strategy] weights_t is None at t={t}，使用等权回退")
                    error_count += 1
                weights_t = np.ones(len(stock_list)) / len(stock_list)
                fallback_count += 1
            elif not isinstance(weights_t, np.ndarray):
                if error_count < max_errors:
                    print(f"[evaluate_strategy] weights_t type error: {type(weights_t)} at t={t}，使用等权回退")
                    error_count += 1
                weights_t = np.ones(len(stock_list)) / len(stock_list)
                fallback_count += 1
            elif len(weights_t) != len(stock_list):
                if error_count < max_errors:
                    print(f"[evaluate_strategy] weights_t length error: {len(weights_t)} at t={t}，使用等权回退")
                    error_count += 1
                weights_t = np.ones(len(stock_list)) / len(stock_list)
                fallback_count += 1
            elif np.any(np.isnan(weights_t)) or np.any(np.isinf(weights_t)):
                if error_count < max_errors:
                    print(f"[evaluate_strategy] weights_t contains NaN or Inf at t={t}，使用等权回退")
                    error_count += 1
                weights_t = np.ones(len(stock_list)) / len(stock_list)
                fallback_count += 1
            
            # 确保权重归一化
            weights_t = np.clip(weights_t, 0, 1)
            weights_sum = np.sum(weights_t)
            if weights_sum == 0:
                weights_t = np.ones(len(stock_list)) / len(stock_list)
                fallback_count += 1
            else:
                weights_t = weights_t / weights_sum
            
            weights_list.append(weights_t)
        except Exception as e:
            if error_count < max_errors:
                print(f"[evaluate_strategy] 策略执行错误: {e} at t={t}，使用等权回退")
                error_count += 1
            weights_t = np.ones(len(stock_list)) / len(stock_list)
            fallback_count += 1
            weights_list.append(weights_t)
            
    # 计算平均权重作为静态策略
    if len(weights_list) == 0:
        print(f"[evaluate_strategy] weights_list is empty，使用等权")
        weights = np.ones(len(stock_list)) / len(stock_list)
    else:
        weights_array = np.array(weights_list)
        if np.any(np.isnan(weights_array)) or np.any(np.isinf(weights_array)):
            print(f"[evaluate_strategy] weights_list contains NaN or Inf，使用等权回退")
            weights = np.ones(len(stock_list)) / len(stock_list)
        else:
            weights = np.mean(weights_array, axis=0)
            
            # 验证 weights
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                print(f"[evaluate_strategy] weights contains NaN or Inf，使用等权回退")
                weights = np.ones(len(stock_list)) / len(stock_list)
            elif weights is None:
                print(f"[evaluate_strategy] weights is None，使用等权回退")
                weights = np.ones(len(stock_list)) / len(stock_list)
            else:
                # 确保权重归一化
                weights = np.clip(weights, 0, 1)
                weights_sum = np.sum(weights)
                if weights_sum == 0:
                    weights = np.ones(len(stock_list)) / len(stock_list)
                else:
                    weights = weights / weights_sum
    
    # Backtest
    nav, returns, sr, so, mdd = backtest(weights, price_array, fee_rate=0.001, slippage_rate=0.0)
    
    # Calculate turnover
    positions = np.array(weights_list)
    turnover = np.mean(np.abs(np.diff(positions, axis=0))) if len(positions) > 1 else 0.0
    
    # 统计有效权重数量
    valid_weights_count = len([w for w in weights_list if not np.any(np.isnan(w)) and not np.any(np.isinf(w))])
    if island_id == 0:
        print(f"   有效权重数量: {valid_weights_count}/{len(weights_list)}")
        print(f"   等权回退次数: {fallback_count}")
    
    # 只在第一个island打印详细信息
    if island_id == 0:
        # 打印前几个权重作为示例
        if len(weights_list) > 0:
            print(f"   首个权重: {weights_list[0]}")
            print(f"   权重数量: {len(weights_list)}")
    
    # 存储评估结果
    island_results.append({
        'island_id': island_id,
        'sharpe_ratio': sr,
        'sortino_ratio': so,
        'max_drawdown': mdd,
        'turnover': turnover,
        'final_nav': nav[-1]
    })
    
    # 当处理完所有island后，打印表格
    # 假设一轮有10个island
    if island_id == 9:  # 最后一个island
        print("\n" + "="*80)
        print("本轮评估结果汇总")
        print("="*80)
        # 打印表头
        print(f"{'Island':<8} {'夏普比率':<10} {'索提诺比率':<10} {'最大回撤':<10} {'换手率':<10} {'最终净值':<12}")
        print("-"*80)
        # 打印每个island的结果
        for result in island_results:
            print(f"{result['island_id']:<8} {result['sharpe_ratio']:.4f}     {result['sortino_ratio']:.4f}     {result['max_drawdown']:.4f}     {result['turnover']:.4f}     {result['final_nav']:.2f}")
        print("="*80 + "\n")

    # Return dictionary with all metrics for comprehensive evaluation
    return {
        'sharpe_ratio': sr,
        'sortino_ratio': so,
        'max_drawdown': mdd,
        'final_nav': nav[-1],
        'turnover': turnover,
        'weights': weights.tolist()
    }


# ====== 7. 策略进化函数 ======
@funsearch.evolve
def investment_strategy(market_state: np.ndarray, portfolio: np.ndarray) -> np.ndarray:
    """
    Enhanced initial strategy template with diversity to promote evolution.
    
    Args:
        market_state: Current market state array (shape: [window_size, 224])
        portfolio: Current portfolio weights (shape: [5])
    
    Returns:
        New portfolio weights (shape: [5])
    """
    # 处理portfolio为None的情况
    if portfolio is None:
        # multi_factor_array的形状是[时间步, 224]，我们需要提取股票相关的特征
        # 假设前5个特征对应5个股票
        if market_state is not None and market_state.shape[1] >= 5:
            # 使用前5个特征作为股票代理
            portfolio = market_state[-1, :5] if len(market_state) > 0 else np.ones(5)
        else:
            portfolio = np.ones(5)
    
    # 基础等权策略
    base_weights = np.ones(len(portfolio)) / len(portfolio)
    
    # 如果market_state有效，使用更智能的策略
    if market_state is not None and len(market_state) > 0:
        # 方法1：基于市场状态的均值策略（使用前5个特征）
        if market_state.shape[1] >= 5:
            mean_weights = np.mean(market_state[:, :5], axis=0)  # 只使用前5个特征
            mean_weights = np.clip(mean_weights, 0, None)  # 确保非负
            mean_sum = np.sum(mean_weights)
            mean_weights = mean_weights / mean_sum if mean_sum > 0 else base_weights
        else:
            mean_weights = base_weights
        
        # 方法2：基于波动率的策略
        if market_state.shape[1] >= 5:
            volatility = np.std(market_state[:, :5], axis=0)
            inv_vol_weights = 1.0 / (volatility + 1e-8)
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
        else:
            inv_vol_weights = base_weights
        
        # 方法3：基于趋势的策略（添加数值稳定性）
        if market_state.shape[1] >= 5 and len(market_state) > 1:
            recent_trend = market_state[-1, :5] - market_state[0, :5]
            # 限制趋势值范围，避免指数溢出
            recent_trend = np.clip(recent_trend, -10, 10)
            trend_weights = np.exp(recent_trend)
            trend_weights = trend_weights / np.sum(trend_weights)
        else:
            trend_weights = base_weights
        
        # 方法4：动量策略（添加数值稳定性）
        if market_state.shape[1] >= 5 and len(market_state) > 1:
            # 计算最近收益
            recent_returns = (market_state[-1, :5] - market_state[-2, :5]) / (market_state[-2, :5] + 1e-8)
            # 限制收益值范围
            recent_returns = np.clip(recent_returns, -1, 1)
            # 减小放大因子，避免指数溢出
            momentum_weights = np.exp(recent_returns * 5)
            momentum_weights = momentum_weights / np.sum(momentum_weights)
        else:
            momentum_weights = base_weights
        
        # 方法5：反转策略（添加数值稳定性）
        if market_state.shape[1] >= 5 and len(market_state) > 1:
            # 计算最近收益
            recent_returns = (market_state[-1, :5] - market_state[-2, :5]) / (market_state[-2, :5] + 1e-8)
            # 限制收益值范围
            recent_returns = np.clip(recent_returns, -1, 1)
            # 减小放大因子，避免指数溢出
            reversal_weights = np.exp(-recent_returns * 5)
            reversal_weights = reversal_weights / np.sum(reversal_weights)
        else:
            reversal_weights = base_weights
        
        # 组合多种策略（随机选择一种以促进多样性）
        strategies = [mean_weights, inv_vol_weights, trend_weights, momentum_weights, reversal_weights, base_weights]
        selected_strategy = np.random.choice(len(strategies))
        
        # 添加随机扰动以促进进化
        noise = np.random.normal(0, 0.05, len(portfolio))
        final_weights = strategies[selected_strategy] + noise
        
        # 确保权重有效
        final_weights = np.clip(final_weights, 0, 1)
        weights_sum = np.sum(final_weights)
        if weights_sum == 0:
            final_weights = base_weights
        else:
            final_weights = final_weights / weights_sum
        
        return final_weights
    
    # 如果market_state无效，使用等权策略
    return base_weights


# ====== 8. 可视化函数 ======
def plot_comparison(nav_dict, output_dir='funsearch_results'):
    """
    Plot comparison of different strategies.
    
    Args:
        nav_dict: Dictionary of {strategy_name: nav_array}
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    color_map = {
        'Equal Weight': '#1f77b4',
        'Minimum Variance': '#2ca02c',
        'Maximum Sharpe': '#d62728',
        'LSTM+PPO': '#7f7f7f',
        'LSTM+PPO Dynamic RL': '#ff7f0e',
        'PPO Dynamic RL (No LSTM)': '#800080',
        'FunSearch': '#e377c2'
    }
    
    import matplotlib.pyplot as plt
    
    # NAV curve
    plt.figure(figsize=(12, 7))
    for name, nav in nav_dict.items():
        if nav is not None and len(nav) > 0:
            plt.plot(nav, label=name, linewidth=2, color=color_map.get(name, None))
    
    plt.legend(fontsize=13)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Net Asset Value', fontsize=14)
    plt.title('Strategy NAV Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_nav_compare.png', dpi=180)
    plt.close()
    
    # Drawdown curve
    plt.figure(figsize=(12, 7))
    for name, nav in nav_dict.items():
        if nav is not None and len(nav) > 0:
            peak = np.maximum.accumulate(nav)
            drawdown = (nav - peak) / (peak + 1e-8)
            plt.plot(drawdown, label=name, linewidth=2, color=color_map.get(name, None))
    
    plt.legend(fontsize=13)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Drawdown', fontsize=14)
    plt.title('Strategy Drawdown Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_drawdown_compare.png', dpi=180)
    plt.close()


def plot_evolution_progress(log_dir='funsearch_evolution', output_dir='funsearch_results'):
    """
    Plot evolution progress from log files.
    
    Args:
        log_dir: Directory containing evolution logs
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evolution logs
    log_file = os.path.join(log_dir, 'evolution_log.json')
    if not os.path.exists(log_file):
        print(f"Evolution log file not found: {log_file}")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            evolution_log = json.load(f)
    except Exception as e:
        print(f"Failed to load evolution log: {e}")
        return
    
    if not evolution_log:
        print("Evolution log is empty")
        return
    
    import matplotlib.pyplot as plt
    
    # Extract data
    generations = []
    best_scores = []
    average_scores = []
    execution_times = []
    
    for entry in evolution_log:
        generations.append(entry.get('generation', 0))
        best_scores.append(entry.get('best_sharpe_ratio', entry.get('best_score', 0)))
        average_scores.append(entry.get('average_sharpe_ratio', entry.get('average_score', 0)))
        execution_times.append(entry.get('execution_time', 0))
    
    # Evolution score trend
    plt.figure(figsize=(12, 7))
    plt.plot(generations, best_scores, 'o-', label='Best Score', linewidth=2, color='#d62728')
    plt.plot(generations, average_scores, 's-', label='Average Score', linewidth=2, color='#1f77b4')
    plt.legend(fontsize=13)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Score (Sharpe Ratio)', fontsize=14)
    plt.title('FunSearch Evolution Progress', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evolution_progress.png', dpi=180)
    plt.close()
    
    # Execution time trend
    plt.figure(figsize=(12, 5))
    plt.plot(generations, execution_times, '^-', label='Execution Time', linewidth=2, color='#2ca02c')
    plt.legend(fontsize=13)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Execution Time per Generation', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evolution_execution_time.png', dpi=180)
    plt.close()
    
    print(f"Evolution progress visualization saved to: {output_dir}/")


# ====== 9. 生成报告 ======
def generate_report(nav_dict, metrics_dict, output_dir='funsearch_results'):
    """
    Generate markdown report.
    
    Args:
        nav_dict: Dictionary of {strategy_name: nav_array}
        metrics_dict: Dictionary of {strategy_name: {metric: value}}
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/funsearch_report.md'
    
    # Calculate additional performance metrics
    extended_metrics = {}
    for name, nav in nav_dict.items():
        if nav is not None and len(nav) > 1:
            returns = (nav[1:] - nav[:-1]) / nav[:-1]
            sr = sharpe_ratio(returns)
            so = sortino_ratio(returns)
            mdd = max_drawdown(nav)
            # Calculate annualized return
            annual_return = (nav[-1] / nav[0]) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            # Calculate Calmar ratio
            calmar_ratio = annual_return / (-mdd) if mdd < 0 else 0
            # Calculate win rate
            win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
            
            extended_metrics[name] = {
                'annual_return': annual_return,
                'sharpe_ratio': sr,
                'sortino_ratio': so,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': mdd,
                'win_rate': win_rate
            }
        else:
            extended_metrics[name] = {
                'annual_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# FunSearch Portfolio Optimization Report\n\n')
        f.write(f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## Strategy Overview\n\n')
        f.write(f'- Number of assets: {len(stock_list)}\n')
        f.write(f'- Backtest period: {len(next(iter(nav_dict.values())))} time steps\n')
        f.write(f'- Stock list: {stock_list}\n\n')
        
        f.write('## Performance Metrics Comparison\n\n')
        f.write('| Strategy | Annual Return | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Max Drawdown | Win Rate |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        
        for name, metrics in extended_metrics.items():
            f.write(f'| {name} | {metrics.get("annual_return", 0):.4f} | {metrics.get("sharpe_ratio", 0):.4f} | {metrics.get("sortino_ratio", 0):.4f} | {metrics.get("calmar_ratio", 0):.4f} | {metrics.get("max_drawdown", 0):.4f} | {metrics.get("win_rate", 0):.4f} |\n')
        
        f.write('\n## NAV Curve\n\n')
        f.write(f'![NAV Curve](portfolio_nav_compare.png)\n\n')
        
        f.write('## Drawdown Curve\n\n')
        f.write(f'![Drawdown Curve](portfolio_drawdown_compare.png)\n\n')
        
        f.write('## Return Distribution\n\n')
        f.write(f'![Return Distribution](portfolio_returns_distribution.png)\n\n')
        
        f.write('## Performance Metrics Radar\n\n')
        f.write(f'![Performance Metrics Radar](portfolio_metrics_radar.png)\n\n')
        
        f.write('## Weight Distribution\n\n')
        f.write(f'![Weight Distribution](portfolio_weights_compare.png)\n\n')
        
        f.write('## Strategy Analysis\n\n')
        f.write('### Best Strategy\n')
        # Find strategy with highest Sharpe ratio
        best_strategy = max(extended_metrics, key=lambda x: extended_metrics[x]['sharpe_ratio'])
        f.write(f'- Best strategy: {best_strategy}\n')
        f.write(f'- Annual return: {extended_metrics[best_strategy]["annual_return"]:.4f}\n')
        f.write(f'- Sharpe ratio: {extended_metrics[best_strategy]["sharpe_ratio"]:.4f}\n')
        f.write(f'- Max drawdown: {extended_metrics[best_strategy]["max_drawdown"]:.4f}\n\n')
        
        f.write('### Strategy Recommendations\n')
        f.write('- Based on historical data, FunSearch-optimized strategies perform well\n')
        f.write('- Consider dynamically adjusting strategy parameters based on market conditions\n')
        f.write('- Regularly retrain models to adapt to market changes\n')
        f.write('- Implement risk management mechanisms to control maximum drawdown\n')
    
    print(f"Report saved to: {report_path}")