# FunSearch Optimization Plan

## 1. 项目分析

### 1.1 当前问题分析

1. **参数设置问题**：
   - 目前参数是硬编码的，用户无法在运行时调整
   - 没有明确的参数设置提示和说明

2. **语言问题**：
   - 输出内容主要是中文，在Colab环境中可能导致字体警告
   - 图片中的文字也是中文，可能无法正常显示

3. **用户体验问题**：
   - 没有明确的运行时间估计
   - 没有清晰的停止条件说明

### 1.2 目标优化

1. **参数设置优化**：
   - 添加运行时参数设置提示
   - 明确说明最大评估次数是主要停止条件
   - 提供运行时间估计

2. **语言优化**：
   - 将所有输出内容改为英文
   - 确保图片中的文字也是英文
   - 避免Colab中的字体警告

3. **用户体验优化**：
   - 提供清晰的运行状态信息
   - 优化输出格式和结构

## 2. 实施计划

### 2.1 参数设置优化

#### 2.1.1 添加命令行参数支持
- 修改 `run_funsearch_enhanced.py` 添加 `argparse` 支持
- 添加 `--max_time_hours` 和 `--max_evaluations` 参数
- 设置合理的默认值

#### 2.1.2 添加运行时参数提示
- 在脚本开始时添加交互式参数设置
- 提供默认值和运行时间估计
- 明确说明最大评估次数是主要停止条件

### 2.2 语言优化

#### 2.2.1 输出内容英文化
- 将所有中文输出改为英文
- 确保错误信息和提示信息也是英文
- 保持专业术语的准确性

#### 2.2.2 图片文字英文化
- 修改 `funsearch_specification_enhanced.py` 中的 `plot_comparison` 函数
- 将图表标题、轴标签等改为英文
- 确保图例和注释也是英文

#### 2.2.3 字体设置优化
- 在 `plot_comparison` 函数中设置明确的英文字体
- 避免使用可能在Colab中不存在的中文字体
- 确保图表在不同环境中都能正常显示

### 2.3 用户体验优化

#### 2.3.1 运行状态信息
- 添加更详细的运行状态信息
- 提供评估进度和时间估计
- 优化输出格式，使其更清晰易读

#### 2.3.2 结果展示优化
- 优化最终结果的展示格式
- 提供更详细的性能指标
- 确保报告内容也是英文

## 3. 技术实现细节

### 3.1 参数设置实现

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run FunSearch with enhanced evaluation')
    parser.add_argument('--max_time_hours', type=float, default=4, help='Maximum run time in hours')
    parser.add_argument('--max_evaluations', type=int, default=2000, help='Maximum number of evaluations (primary stopping condition)')
    parser.add_argument('--non_interactive', action='store_true', help='Run without interactive prompts')
    return parser.parse_args()

def get_user_input():
    print("\n=== FunSearch Configuration ===")
    print("Note: Maximum evaluations is the primary stopping condition")
    print("Approximately 100 evaluations take about 30 minutes")
    
    max_time = input(f"Enter maximum run time in hours [default: 4]: ")
    max_evals = input(f"Enter maximum evaluations [default: 2000]: ")
    
    max_time = float(max_time) if max_time else 4
    max_evals = int(max_evals) if max_evals else 2000
    
    print(f"\nConfiguration:")
    print(f"- Maximum run time: {max_time} hours")
    print(f"- Maximum evaluations: {max_evals}")
    print(f"- Estimated time: ~{max_evals/2} minutes")
    
    return max_time, max_evals
```

### 3.2 语言和字体设置实现

#### 3.2.1 输出内容英文化
- 将所有 `print` 语句中的中文改为英文
- 确保错误信息和提示信息也是英文

#### 3.2.2 图片文字和字体设置

```python
def plot_comparison(nav_dict):
    """Plot comparison of strategies"""
    plt.figure(figsize=(12, 8))
    
    # Set English font to avoid Colab warnings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    for name, nav in nav_dict.items():
        plt.plot(nav, label=name)
    
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Asset Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('funsearch_results', exist_ok=True)
    plt.savefig('funsearch_results/strategy_comparison.png')
    plt.close()
```

### 3.3 运行状态信息实现

```python
def run_funsearch_with_evaluation(max_time_hours=4, max_evaluations=1000):
    """Run FunSearch with evaluation and save results."""
    print("Starting FunSearch Strategy Evolution (Enhanced)...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Stopping conditions:")
    print(f"   - Maximum run time: {max_time_hours} hours")
    print(f"   - Maximum evaluations: {max_evaluations} (primary)")
    print(f"Estimated run time: ~{max_evaluations/2} minutes")
    print("Configuring FunSearch...")
    
    # Rest of the function...
```

## 4. 风险控制

### 4.1 潜在风险

1. **参数设置错误**：用户可能输入无效的参数值
2. **字体设置问题**：不同环境中可能有不同的可用字体
3. **运行时间估计不准确**：实际运行时间可能与估计有差异

### 4.2 缓解措施

1. **参数验证**：添加参数验证逻辑，确保输入值有效
2. **字体 fallback**：设置多个字体选项，确保至少有一个可用
3. **动态时间估计**：根据实际运行速度调整时间估计

## 5. 执行时间表

1. **参数设置优化**：30分钟
2. **语言和字体优化**：45分钟
3. **用户体验优化**：15分钟
4. **测试和验证**：30分钟

## 6. 预期成果

1. **优化后的run_funsearch_enhanced.py**：
   - 支持命令行参数
   - 提供交互式参数设置
   - 所有输出内容为英文
   - 清晰的运行状态信息和时间估计

2. **优化后的图表**：
   - 英文标题和标签
   - 避免Colab字体警告
   - 清晰易读的格式

3. **更好的用户体验**：
   - 明确的参数设置提示
   - 清晰的停止条件说明
   - 准确的运行时间估计
   - 专业的英文输出内容