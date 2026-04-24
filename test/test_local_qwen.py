# AIIS/test_local_qwen.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 使用本地缓存路径
cache_dir = "C:/Users/Administrator/.cache/huggingface/hub"

print("正在加载Qwen2.5-1.5B-Instruct模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/Administrator/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "C:/Users/Administrator/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("✅ 模型加载成功！")

# 生成策略代码
prompt = """请生成一个投资策略函数，使用LSTM特征和当前持仓来生成资产权重：

要求：
1. 函数名为 investment_strategy
2. 输入参数：market_state (LSTM特征向量), portfolio (当前持仓)
3. 输出：新的资产权重数组
4. 初始模板可以使用简单的等权策略

请只返回函数代码，不要返回其他内容。"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=200, 
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== 生成的策略代码 ===")
print(response)