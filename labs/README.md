# Labs：从公式到可观察机制

本目录包含 6 个基础实验与 22 个 Frontier mechanism-level labs。当前脚本只依赖 **NumPy、Matplotlib 与 Python 标准库**，设计目标是让教材中的数学机制可以在普通 CPU 环境中被观察、修改和质疑，而不是追求 benchmark。

## 基础实验

| 脚本 | 主题 |
|---|---|
| `01_generalization.py` | 多项式过拟合与泛化 |
| `02_nullspace_edit.py` | 零空间约束与知识编辑 toy model |
| `03_conditioning.py` | 条件数与优化轨迹 |
| `04_attention.py` | 手写 causal attention |
| `05_best_of_n.py` | test-time sampling 的收益曲线 |
| `06_reward_hacking.py` | proxy reward 与 specification gaming |

## Frontier Labs

`frontier/ch01_*.py` 到 `frontier/ch22_*.py` 与教材 22 章逐章对应，主题覆盖 grokking、AlphaEdit、twisted SMC、矩阵 polar、logit rank、feature learning、gradient influence、context entropy、tokenizer scaling、gated attention、RoPE、scaling/coverage、test-time compute、Bayesian ICL、DPO/RLVR、agentic retrieval、POMDP、planning/replanning、memory/self-play 等。

> **Evidence boundary**：这些脚本属于机制级复现（mechanism-level reproduction）。它们用于验证某个数学结构、构造反例或观察定性现象，**不能替代论文规模实验，也不能单独证明论文的经验结论**。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python labs/frontier/ch13_gated_attention.py
```

部分脚本会在仓库根目录创建 `figures/generated/`。该目录默认被 `.gitignore` 忽略。
