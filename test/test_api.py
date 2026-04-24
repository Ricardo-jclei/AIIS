# AIIS/test_api.py
import os
from openai import OpenAI

# 设置API Key
client = OpenAI(
    api_key="sk-917afa319bea49979aa8f1d483e16742",  # 你的最新DeepSeek API Key
    base_url="https://api.deepseek.com/v1"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=0.7,
        max_tokens=50
    )
    print("✅ DeepSeek API works!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API Error: {e}")