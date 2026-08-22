# AGENTS.md — 给 AI 协作者的说明

本文件面向任何要**阅读、调用或修改** `stock_predictor.py` 的 AI 编码助手
（Claude Code / Cursor / Copilot / Qwen Code 等）。人类贡献者也可参考。

---

## 0. 一句话理解本项目

一个单文件的 A 股预测研究平台，靠一个**统一的模型接口 `BaseModel` + 注册表 `ALGO_REGISTRY`**
把 34 种算法插拔在同一套"取数→特征→训练→评估→可视化"流水线上。

---

## 1. 架构地图（改代码前必读）

```
stock_predictor.py
├─ StockDataFetcher     取数：fetch()真实 / generate_synthetic_data()合成 / parquet 缓存
├─ FeatureEngineer      技术指标 → 滑动窗口样本 → 标准化 → 时间序列切分
├─ Metrics             12 个评价指标，统一入口 calc_all()，含方向准确率 DA
├─ BaseModel (ABC)      所有算法的父类：fit / predict / get_hpo_space
│   └─ 34 个子类 → 注册进 ALGO_REGISTRY / ALGO_AVAILABILITY
├─ run_hpo + 5 个优化器 PSO / GA / FA / SOA / BO(默认)
├─ TrainingPipeline     调度：prepare_data / run_single / run_batch / _naive_baseline / predict_next
├─ MainWindow (PySide6) 图形界面（GUI 只读上面的东西，不含业务逻辑）
└─ run_experiment / run_cli / main   命令行与可编程入口
```

**数据契约**：整条流水线只认列名为 `date, open, close, high, low, volume` 且**按时间升序**的
`pandas.DataFrame`。只要产出这样的 DataFrame，数据源可以随意替换。

---

## 2. 不可触碰的红线（违反 = 制造出"看起来很准、实际造假"的模型）

1. **时间序列绝不能随机 shuffle 切分**。必须用 `FeatureEngineer.time_series_split`（前段训练、后段测试）。
   用 sklearn 默认 shuffle 会用未来预测过去，是严重的数据泄漏。
2. **标准化器只在训练集上 `fit`**，测试集只 `transform`（见 `prepare_data`）。绝不在全量或测试集上 fit。
3. **绝不删除 Naive(前值) 朴素基准和 DA 方向准确率**，也不得为了让数字好看而绕过它们。
   任何"提升精度"的改动，都必须仍然和朴素基准做公平对比。
4. **不得把目标价格泄漏进特征**。当前特征只用"当天及以前可获得"的信息，新增指标时保持这一点。
5. 新增依赖必须放进 `try/except ImportError` 的可选依赖区，缺失时**降级而非崩溃**。

---

## 3. 常见修改配方

### 配方 A：新增一个算法
1. 写一个 `BaseModel` 子类，设置 `name` 和 `category`（三选一：
   `"基础与传统模型"` / `"先进与前沿模型"` / `"公式拟合/演化计算"`）。
2. 实现 `fit(self, X, y)`（返回 self）和 `predict(self, X)`（返回一维 np.ndarray）。
3. 可选：实现 `get_hpo_space()` 返回 `{参数名: (下界, 上界)}` 以支持超参数寻优。
4. 在 `ALGO_REGISTRY` 里加一行 `"名字": 你的类`。若依赖可选库，在 `ALGO_AVAILABILITY` 里登记开关。
5. 自测：`python stock_predictor.py --cli --synthetic --algos 你的名字`
6. 深度学习模型请继承 `TorchModelBase`，只实现 `_build_network(n_features)`；它会自动帮你把展平的
   X 还原成 `(样本, 时间步, 特征)`——所以你的 `__init__` 必须接受并透传 `window_size`。

### 配方 B：新增一个评价指标
1. 在 `Metrics` 里加一个 `@staticmethod`，签名 `(y_true, y_pred)`。
2. 在 `Metrics._FUNC_MAP` 里注册 `"简称": 你的函数.__func__`。
3. 如需在 GUI 勾选，把简称加进 `_build_metric_group` 的 `metric_names`。

### 配方 C：换数据源
改 `StockDataFetcher.fetch()`，或在外部构造好 DataFrame 直接传给
`TrainingPipeline(...).run_batch(algos, raw_df)`。保持数据契约（第 1 节）即可，其它一律不用动。

---

## 4. 如何自测（没有 GPU / 没联网也能测）

```bash
# 最快：合成数据 + 几个纯 sklearn 算法（秒级）
python stock_predictor.py --cli --synthetic --algos "RF,SVR,GBRT" --json out.json

# 带向前预测
python stock_predictor.py --cli --synthetic --algos "RF" --forecast

# 验证语法/导入（不跑训练）
python -c "import stock_predictor"
```
判断改动是否正确的最低标准：CLI 能跑通、结果表里有 `Naive(前值)` 一行、你的模型 R² 不是 NaN。

---

## 5. 已知的"近似实现"与可改进点（欢迎 AI 认领）

- **Kstar / M5Rules / GEP / MEP**：目前是数学上合理的近似（见各类注释）。可接入
  `python-weka-wrapper3`（Kstar/M5Rules）或 `geppy`（GEP）做精确实现。
- **返回值预测模式**：当前默认预测"价格水平"。可新增 `target_mode="return"` 改为预测"次日收益率"，
  再由前收盘价重建价格——这通常是更严谨的做法，是很好的下一步。（改 `FeatureEngineer` 与 `TrainingPipeline`，
  务必同步更新 Naive 基准与 DA 的计算口径。）
- **多步预测 / 多标的**：`horizon>1` 时 Naive 基准与 DA 仅对 `horizon=1` 严格成立，可扩展。
- **回测层**：目前只做逐点误差评估，可加"按预测方向模拟买卖"的收益回测（含手续费/滑点）。

改进时请回到第 2 节红线核对一遍。
