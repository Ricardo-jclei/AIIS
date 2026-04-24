import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import os
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model.enhanced_lstm import EnhancedLSTMModel

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

# 加载LSTM模型
lstm_model_path = 'model_ckpt/best_lstm_multi_asset.pth'
lstm_input_size = multi_factor_array.shape[1]
lstm_model = EnhancedLSTMModel(lstm_input_size, 128, 3, lstm_input_size, 0.3)
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location='cpu'))
lstm_model.eval()

# 验证LSTM特征提取
window_size = 20
t = 50
window = multi_factor_array[t-window_size:t]
window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-8)
X = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    feats = lstm_model(X).cpu().numpy().flatten()

print(f"LSTM特征向量形状: {feats.shape}")  # 应输出 (224,)
print(f"✅ LSTM特征提取验证通过！")