# 《从机器学习到智能体》· 分章阅读

> v2.4 Final · 5 篇 · 22 章 · 300 页完整版拆分

为了让 GitHub 成为可以直接使用的教学入口，教材按章节提供 PDF。每章 PDF 保留最终版的公式、图、Research Frontier、Worked Example 与习题版式；配套的机制级实验位于 [`../labs/frontier/`](../labs/frontier/)。

## Part I · 学习的数学骨架

1. [机器究竟怎样“学习”？——经验风险、泛化与归纳偏置](chapters/ch01.pdf) · [Frontier Lab](../labs/frontier/ch01_spectral_grokking.py)
2. [向量、矩阵与子空间——表示学习的几何语言](chapters/ch02.pdf) · [Frontier Lab](../labs/frontier/ch02_alphaedit_nullspace.py)
3. [概率、似然与贝叶斯推断——模型怎样表示不确定性](chapters/ch03.pdf) · [Frontier Lab](../labs/frontier/ch03_twisted_smc.py)
4. [信息论与交叉熵——从最短编码到语言建模](chapters/ch04.pdf) · [Frontier Lab](../labs/frontier/ch04_compression_elasticity.py)
5. [优化与学习动力学——参数究竟怎样被找到](chapters/ch05.pdf) · [Frontier Lab](../labs/frontier/ch05_matrix_polar.py)

## Part II · 从线性模型到深度学习

6. [线性回归不是“简单模型”——投影、隐式偏置与谱动力学](chapters/ch06.pdf) · [Frontier Lab](../labs/frontier/ch06_spectral_filters.py)
7. [Logistic、Softmax 与 Logit 几何——分类如何变成概率学习](chapters/ch07.pdf) · [Frontier Lab](../labs/frontier/ch07_logit_rank.py)
8. [神经网络与表示学习——从固定特征到 Feature Learning](chapters/ch08.pdf) · [Frontier Lab](../labs/frontier/ch08_feature_learning.py)
9. [反向传播与自动微分——从链式法则到学习影响](chapters/ch09.pdf) · [Frontier Lab](../labs/frontier/ch09_gradient_influence.py)
10. [深度网络为什么能训起来——信号尺度、优化器状态与 Scaling](chapters/ch10.pdf) · [Frontier Lab](../labs/frontier/ch10_lowrank_momentum.py)

## Part III · 从序列概率到 Transformer

11. [语言的概率模型——从链式法则、熵率到 Next-Token Prediction](chapters/ch11.pdf) · [Frontier Lab](../labs/frontier/ch11_context_entropy.py)
12. [Tokenization 与 Embedding——离散符号怎样变成可学习的几何](chapters/ch12.pdf) · [Frontier Lab](../labs/frontier/ch12_tokenizer_scaling.py)
13. [Attention——从核回归到可微分内容寻址](chapters/ch13.pdf) · [Frontier Lab](../labs/frontier/ch13_gated_attention.py)
14. [Transformer——位置、残差与可扩展序列计算的完整组装](chapters/ch14.pdf) · [Frontier Lab](../labs/frontier/ch14_position_kvcache.py)

## Part IV · 大语言模型理论

15. [LLM 预训练与 Scaling——从交叉熵到计算最优与 Coverage](chapters/ch15.pdf) · [Frontier Lab](../labs/frontier/ch15_scaling_coverage.py)
16. [LLM 解码与测试时计算——搜索、验证与计算分配](chapters/ch16.pdf) · [Frontier Lab](../labs/frontier/ch16_testtime_compute.py)
17. [能力、表示与上下文学习——LLM 为什么会“像是在推理”？](chapters/ch17.pdf) · [Frontier Lab](../labs/frontier/ch17_icl_bayes.py)
18. [后训练、偏好优化与 RLVR——概率质量如何被重新塑形](chapters/ch18.pdf) · [Frontier Lab](../labs/frontier/ch18_dpo_rlvr.py)

## Part V · 从 LLM 到 Agent

19. [RAG、Agentic Search 与外部证据——从“查到资料”到主动获取信息](chapters/ch19.pdf) · [Frontier Lab](../labs/frontier/ch19_agentic_retrieval.py)
20. [MDP/POMDP——Agent 的序贯决策数学](chapters/ch20.pdf) · [Frontier Lab](../labs/frontier/ch20_belief_pomdp.py)
21. [Tool Use 与 Planning——搜索、层级动作和闭环执行](chapters/ch21.pdf) · [Frontier Lab](../labs/frontier/ch21_planning_tools.py)
22. [Memory、Multi-Agent 与 Agent Learning——让智能体跨时间积累](chapters/ch22.pdf) · [Frontier Lab](../labs/frontier/ch22_memory_selfplay.py)

## 推荐阅读方式

第一次学习按 1→22 顺序推进；每章先读 Prerequisite Recall、Visual Derivation、Foundation 和 Worked Example，再做 A/B 级题。研究导向学习再进入 Research Frontier，并运行对应 `chXX_*.py`。

> **Evidence boundary:** `labs/frontier/` 是 mechanism-level reproduction，用来观察论文背后的数学机制，不等价于 paper-scale replication。论文级结论仍应回到论文原文、官方代码与对应实验预算核验。
