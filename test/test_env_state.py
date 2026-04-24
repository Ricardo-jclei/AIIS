import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import os
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model.enhanced_lstm import EnhancedLSTMModel
from src.rl.multi_asset_trading_env import MultiAssetTradingEnv

# 加载数据
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

data_dir_tpl = 'data/processed/{}/market/daily/20220425_20250424_processed.csv'
price_dfs = []
for code in stock_list:
    path = data_dir_tpl.format(code)
    df = pd.read_csv(path, usecols=['日期', '收盘'])
    df = df.rename(columns={'收盘': code})
    price_dfs.append(df)
price_df = price_dfs[0]
for df in price_dfs[1:]:
    price_df = price_df.merge(df, on='日期', how='inner')
price_array = price_df[stock_list].values

# 创建环境
env = MultiAssetTradingEnv(
    price_array=price_array,
    feature_array=multi_factor_array,
    window_size=20,
    initial_cash=1e7,
    lstm_model_path='model_ckpt/best_lstm_multi_asset.pth',
    lstm_input_size=multi_factor_array.shape[1],
    asset_num=5,
    include_lstm_features=True
)

# 验证状态向量
obs, _ = env.reset()
print(f"状态向量形状: {obs.shape}")  # 应输出 (230,)
print(f"LSTM特征维度: {obs[:224].shape}")  # 应输出 (224,)
print(f"持仓比例维度: {obs[224:229].shape}")  # 应输出 (5,)
print(f"现金比例维度: {obs[229:].shape}")  # 应输出 (1,)
print(f"✅ 环境状态向量验证通过！")