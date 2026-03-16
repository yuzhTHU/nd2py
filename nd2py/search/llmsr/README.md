# LLMSR - LLM-guided Symbolic Regression

LLMSR 是一个使用大型语言模型 (LLM) 引导的符号回归算法。它通过迭代地让 LLM 改进方程结构，结合数值优化评估方程质量，从而发现数据的数学表达式。

本项目参考 [LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR) (ICLR 2025 Oral) 实现。

## 快速开始

### 导入

```python
import numpy as np
from nd2py.search.llmsr import LLMSR
```

### 支持的 LLM Provider

| Provider | 模型 | 类型 |
|----------|------|------|
| SiliconFlow | `Qwen3-8B` | **免费** |
| SiliconFlow | `Deepseek-V3` | 付费 |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | 付费 |
| OpenAI | `gpt-4o-mini`, `gpt-5-mini` | 付费 |
| Gemini | `gemini-2.5-pro`, `gemini-2.5-flash` | 付费 |
| OpenRouter | `kimi-k2`, `gemini-2.5-pro` | 付费 |

### 环境变量配置

```bash
# SiliconFlow (推荐，Qwen3-8B 免费)
export SILICONFLOW_API_KEY="your-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# OpenAI/Azure
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_VERSION="2024-xx-xx"
export OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

# Gemini
export GEMINI_API_KEY="your-api-key"
```

## 完整示例

以下示例可以直接复制到 Jupyter Notebook 或 Python 脚本中运行：

```python
from nd2py.search.llmsr import LLMSR
import numpy as np

# ============================================================
# 1. 定义问题：发现 y = 2*sin(x) + 0.5*x^2 的数学表达式
# ============================================================
prompt = """Find the mathematical function skeleton that best fits the data.
You should generate `def equation(...)` directly, without any additional comments or explanations.
The function should take x (input) and params (coefficients) as arguments."""


def evaluate(x: np.ndarray, y: np.ndarray, maxn_params: int = 10) -> float:
    """
    评估方程的质量（负 MSE）。

    Args:
        x: 输入数据
        y: 目标值
        maxn_params: 最大参数数量

    Returns:
        负均方误差（越大越好）
    """
    from scipy.optimize import minimize

    def loss(params):
        y_pred = equation(x, params)
        return np.mean((y_pred - y) ** 2)

    result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")

    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return float('-inf')
    return -loss_val  # 负 MSE 作为分数


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """种子方程：简单的线性模型（LLMSR 会迭代改进它）"""
    return params[0] * x


# ============================================================
# 2. 准备数据
# ============================================================
np.random.seed(42)
N = 100
x = np.linspace(-np.pi, np.pi, N)
true_y = 2 * np.sin(x) + 0.5 * x**2 + np.random.normal(0, 0.1, N)  # 添加噪声
data = {"x": x, "y": true_y}


# ============================================================
# 3. 配置并运行 LLMSR
# ============================================================
est = LLMSR(
    prompt=prompt,
    eval_program=evaluate,
    seed_program=equation,
    namespace={"np": np},
    llm_provider="SiliconFlow",   # LLM Provider（需要申请 API Key 并设置 SILICONFLOW_API_KEY 环境变量）
    llm_model="Qwen3-8B",         # 使用 Qwen3-8B 模型
    n_islands=10,                 # 岛屿数量（并行种群）
    n_iter=50,                    # 迭代次数
    programs_per_prompt=2,        # 每次请求生成的程序数量
    temperature_init=0.1,         # 初始温度（用于 Boltzmann 选择）
    temperature_period=30,        # 温度周期
    log_per_iter=1,               # 每 N 次迭代记录日志
    save_path="./logs/llmsr_demo",  # 结果保存路径（设为 None 则不保存）
    log_detailed_speed=True,      # 记录详细的速度信息
)
print("开始 LLMSR 搜索...")
est.fit(data)

# ============================================================
# 4. 查看结果
# ============================================================
print("\n" + "=" * 60)
print("发现的最优方程:")
print("=" * 60)
print(est.best_model.program)
```

## 参数说明

### LLMSR 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | - | 给 LLM 的提示词 |
| `eval_program` | callable | - | 评估函数，返回负 MSE |
| `seed_program` | callable | - | 种子方程（初始方程骨架） |
| `template` | str | `"{prompt}\n\n{eval_program}\n\n{seed_programs}"` | Prompt 模板 |
| `namespace` | dict | `{}` | 执行方程时的命名空间 |
| `n_islands` | int | `10` | 岛屿数量（并行种群） |
| `n_iter` | int | `1000` | 迭代次数 |
| `programs_per_prompt` | int | `2` | 每次请求生成的程序数量 |
| `temperature_init` | float | `0.1` | 初始温度 |
| `temperature_period` | int | `30000` | 温度周期 |
| `random_state` | int | `None` | 随机种子 |
| `log_per_iter` | int | `1` | 日志记录频率（次） |
| `log_per_sec` | float | `None` | 日志记录频率（秒） |
| `save_path` | str | `None` | 结果保存路径 |
| `llm_provider` | str | `"SiliconFlow"` | LLM Provider 名称 |
| `llm_model` | str | `"Qwen3-8B"` | LLM 模型名称 |

### 评估函数要求

```python
def evaluate(x: np.ndarray, y: np.ndarray, maxn_params: int = 10) -> float:
    """
    评估方程的质量。

    该函数会使用全局变量 `equation`（由 LLMSR 动态生成）。

    Returns:
        分数（越大越好），通常使用负 MSE
    """
    # ... 使用 equation(x, params) 进行预测
    # ... 使用 scipy.optimize.minimize 优化参数
    # ... 返回负 MSE
```

### 种子方程要求

```python
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    种子方程：LLMSR 会迭代改进这个方程的结构。

    Args:
        x: 输入数据（可以是多个变量）
        params: 可优化的参数

    Returns:
        预测值
    """
    return params[0] * x  # 简单的线性模型作为起点
```

## 高级用法

### 多变量符号回归

```python
# 定义多变量的评估函数和数据
def evaluate(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, maxn_params=10) -> float:
    from scipy.optimize import minimize

    def loss(params):
        y_pred = equation(x1, x2, params)
        return np.mean((y_pred - y) ** 2)

    result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")
    return -result.fun if np.isfinite(result.fun) else float('-inf')

def equation(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """种子方程：简单的加法模型"""
    return params[0] * x1 + params[1] * x2

# 准备多变量数据
data = {
    "x1": np.random.random(100),
    "x2": np.random.random(100),
    "y": 2 * np.random.random(100) + 3 * np.random.random(100),
}

est = LLMSR(
    prompt="Find a function of x1 and x2.",
    eval_program=evaluate,
    seed_program=equation,
    namespace={"np": np},
    llm_provider="SiliconFlow",
    llm_model="Qwen3-8B",
)
est.fit(data)
```

### 切换不同 LLM Provider

```python
# SiliconFlow Qwen3-8B (免费)
est = LLMSR(llm_provider="SiliconFlow", llm_model="Qwen3-8B", ...)

# SiliconFlow DeepSeek-V3 (付费)
est = LLMSR(llm_provider="SiliconFlow", llm_model="Deepseek-V3", ...)

# DeepSeek
est = LLMSR(llm_provider="DeepSeek", llm_model="deepseek-chat", ...)

# OpenAI
est = LLMSR(llm_provider="OpenAI", llm_model="gpt-4o-mini", ...)

# Gemini
est = LLMSR(llm_provider="Gemini", llm_model="gemini-2.5-pro", ...)

# OpenRouter
est = LLMSR(llm_provider="OpenRouter", llm_model="kimi-k2", ...)
```

### 日志输出说明

启用 `log_detailed_speed=True` 后，日志会显示：

- **Time Usage**: 各阶段耗时（Tournament、Generate Prompt、Generate Children、Set Score）
- **Token Usage**: Token 消耗统计（prompt/answer/reason/total）
- **Money Usage**: 费用统计（根据各 Provider 的定价自动计算）
- **Best Program**: 当前最优方程

示例输出：
```
Iter: 10 | Score: -0.006735 | Program Length: 144 | Population Size: 25 |
Time Usage: 5.32 min (Other Stuff=2.1 min [39%]; Set Score=1.8 min [34%]; ...) |
Token Usage: 13.3 ktoken (answer=8.2 ktoken [62%]; prompt=4.1 ktoken [31%]; ...) |
Money Usage: $0.15 (answer=$0.09 [60%]; prompt=$0.06 [40%]; ...)
```

## 运行测试

```bash
# 激活虚拟环境
conda activate ./venv

# 运行 SiliconFlow API 测试（Qwen3-8B 免费）
./venv/bin/python -m pytest tests/search/llmsr/test_siliconflow_api.py -v --run-slow

# 运行集成测试（需要 SILICONFLOW_API_KEY）
./venv/bin/python -m pytest tests/search/llmsr/test_llmsr_integration.py -v --run-slow

# 运行 demo 脚本
./venv/bin/python demo/llmsr.py
```

## 常见问题

### Q: 为什么使用 `eval_program` 和 `seed_program` 函数而不是直接传字符串？

A: 使用函数可以：
1. 利用 Python 的类型检查和代码高亮
2. 更容易调试和维护
3. 自动获取源代码作为 prompt 模板

### Q: 如何提高搜索质量？

A: 尝试以下方法：
1. 增加 `n_iter`（更多迭代）
2. 增加 `n_islands`（更多并行种群）
3. 调整 `temperature_init`（更高的温度增加探索）
4. 改进 `prompt`（更清晰的指示）
5. 使用更强大的 LLM（如 DeepSeek-V3 或 GPT-4）

### Q: 搜索很慢怎么办？

A: 可以尝试：
1. 减少 `n_iter` 和 `n_islands`
2. 减少 `programs_per_prompt`
3. 使用更快的 LLM（Qwen3-8B 比 DeepSeek-V3 快）
4. 启用 `log_detailed_speed=True` 分析瓶颈
