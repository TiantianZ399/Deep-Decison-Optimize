# DDO-MD vs SPO / SPO+ / MSE：枚举路径基准总结

## 实验设定
- 任务：将一个小型 layered shortest-path 图完全枚举成可行路径集合，然后在同一批路径上比较 `mse / spo / spo+ / ddo-md`。
- 公平性：所有方法共享同一张图、同一 synthetic teacher、同一个线性 student、同一训练/验证/测试划分规则、同一 Adam 优化器和同一调参预算。
- 图规模：33 条边、81 条可行路径；所有路径长度一致。
- 数据：train=96, val=128, test=512, feature dim=16; synthetic teacher 含线性 + 正弦非线性，线性 student 刻意 misspecified。
- 调参：在验证集上用标准 path regret（standard SPO loss）选超参；调参种子 2 个，最终报告 5 个 seeds。
- 预算：每组超参调参训练 8 epochs；最终五种子汇总训练 12 epochs。

## Selected hyperparameters
- ddo-md: lr=0.1, tau=0.05
- mse: lr=0.03, tau=0.0
- spo: lr=0.001, tau=0.0
- spo+: lr=0.03, tau=0.0

## Five-seed summary
| METHOD   | STANDARD SPO LOSS (= PATH REGRET)   | PATH ACCURACY   | EDGE OVERLAP    | RUNTIME (S)     |   REGRET RANK |
|:---------|:------------------------------------|:----------------|:----------------|:----------------|--------------:|
| ddo-md   | 0.5704 ± 0.0270                     | 0.0742 ± 0.0154 | 0.3524 ± 0.0217 | 0.0291 ± 0.0015 |             1 |
| mse      | 0.6255 ± 0.0322                     | 0.0734 ± 0.0088 | 0.3266 ± 0.0122 | 0.0243 ± 0.0011 |             2 |
| spo+     | 0.6393 ± 0.0289                     | 0.0586 ± 0.0135 | 0.3181 ± 0.0119 | 0.0306 ± 0.0018 |             3 |
| spo      | 1.0593 ± 0.0499                     | 0.0090 ± 0.0033 | 0.1908 ± 0.0159 | 0.0290 ± 0.0009 |             4 |

## Main takeaways
- 最优 mean standard SPO loss / path regret 是 **ddo-md = 0.5704**，优于第二名 **mse = 0.6255**。
- 在这个 collaborator-style 的枚举路径基准里，**ddo-md** 的 mean path regret 低于 `spo` (0.5704 vs 1.0593)、`spo+` (0.5704 vs 0.6393)，也低于 `mse` (0.5704 vs 0.6255)。
- Path accuracy 上，`ddo-md` 为 0.0742；`mse` 为 0.0734。这个实验里，regret 改善比 exact path match 更明显。
- Edge overlap 上，`ddo-md` 为 0.3524，说明即便没有显著拉开 exact path accuracy，它也更稳定地把更多正确边放进了最终路径。
- 直接 `spo` 行应当只被看作 heuristic baseline：它不是 exact gradient method，而是 regret-scaled 的 straight-through 方向。

## 文件说明
- `scripts/run_benchmark_comparison.py`：一键复现实验、表格和图。
- `tables/`：调参结果、五种子 summary、LaTeX 表格。
- `figures/`：path regret、decision quality、validation curves。
- `main.tex`：可编译的简短实验报告模板。
