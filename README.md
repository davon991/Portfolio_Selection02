## `README.md`
```
# CtR 主目标 + CtB 风控约束（Stage 9A Minimal Runnable）

本目录是“模块 9：代码实现与跑通”的 **9A 最小可运行实现**。

## 当前实现目标

- 按月再平衡；
- 支持 `EW / GMV / ERC / CTB_ONLY / RB_CTB_BAND`；
- 使用 long-only、全投资、单资产权重上限；
- 支持样本协方差与 Ledoit-Wolf 收缩协方差；
- 支持 `CtR`、`CtB`、`D_R`、`D_B`、换手、成本、回撤等核心指标；
- 支持训练/验证/测试切分与 `(delta, eta, gamma)` 的规则化校准；
- 每次 run 同时输出论文资产与 GPT 可读分析资产。

## 目录

- `configs/`：运行配置
- `scripts/run_experiment.py`：单一入口
- `src/`：数据、风险、指标、求解、回测、报告
- `results/`：结果目录

## 运行

```powershell
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/main.yaml
```

## 说明

本阶段优先保证：

1. 口径正确；
2. 输出契约完整；
3. 代码可跑通；
4. 为下一阶段“冻结校准协议”和“按协议跑完整实验”提供反馈。

正式的 unified solver family（Projected Gradient + Active-set Damped Newton）保留为下一阶段增强；本阶段采用稳健的 SLSQP / constrained optimization 方案完成最小可运行实现。