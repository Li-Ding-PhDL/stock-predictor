# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指引。

## 项目概述

单文件 A 股预测研究平台。所有功能（~6000 行）集中在 `stock_predictor.py`，三种使用方式：图形界面 / 命令行 / 可编程 API。

**核心信条**：诚实优先——真实数据、不造假、双基准强制对比、DA 显著性检验、多重比较偏差警示。

---

## 常用命令

```bash
# 图形界面（GUI）
python stock_predictor.py

# 合成数据冒烟测试（无需联网，秒级）
python stock_predictor.py --cli --synthetic --algos "RF,SVR,GBRT"

# 合成数据 + 向前预测
python stock_predictor.py --cli --synthetic --algos "RF" --forecast

# 真实股票 + 未来一周逐日预测 + 回测
python stock_predictor.py --cli --code 600519 --week --backtest

# 仅验证导入语法（不跑训练）
python -c "import stock_predictor"

# 输出 JSON 结果
python stock_predictor.py --cli --synthetic --algos "RF,SVR,GBRT" --json out.json
```

没有独立的 lint 或 test 命令；改动正确的最低标准是：CLI 冒烟能跑通、结果表里有 `Naive(前值)` 一行、目标模型 R² 不是 NaN。

### Docker / 自动化

```bash
docker build -t stock-predictor . && docker run --rm stock-predictor --cli --synthetic --algos "RF,SVR,GBRT"
# cron 每日 15:30 自动核对到期预测
30 15 * * 1-5 /path/to/scripts/daily_verify.sh >> verify.log 2>&1
```

---

## 代码架构

`stock_predictor.py` 按节编号，改任何东西前先定位所在节：

| 节 | 行范围（约） | 内容 |
|---|---|---|
| 一 | 1–201 | 依赖导入（核心 + try/except 可选依赖） |
| 二 | 202–251 | 全局常量、`TrainConfig`（dataclass 配置） |
| 三 | 252–1406 | `StockDataFetcher`：多源取数、parquet 缓存、估值、资金流、新闻、实时盘口 |
| 四 | 1407–1734 | `FeatureEngineer`：技术指标 → 滑动窗口 → 标准化 → 时间序列三分 |
| 五 | 1735–1889 | `Metrics`：12 个指标，统一入口 `calc_all()`，含 DA 方向准确率 |
| 六 | 1890–2765 | `BaseModel` (ABC) + 35 个子类 + `ALGO_REGISTRY` / `ALGO_AVAILABILITY` |
| 七 | 2766–2987 | 超参数优化：PSO / GA / FA / SOA / BO(optuna) |
| 八 | 2988–6583 | `TrainingPipeline`（调度 + CV 前扩窗口）+ 回测(含 Beta/Jensen's Alpha) + 未来预测 + 风险 + 因子选股(含盈余质量/税率异常/因子权重优化) + 预测跟踪 + 校准检验 + 尾盘选股 + 监管观察名单 + 模拟交易账户 |
| 九 | 6584–11508 | `MainWindow`（PySide6 GUI，13 页签，含「模拟交易」） |
| 十 | 11509–11621 | `run_experiment`（可编程 API） |
| 十一 | 11622– | CLI 解析 + `main()` 入口 |

### 关键设计点

- **数据契约**：整条流水线只认 `date, open, close, high, low, volume` 列、按时间升序的 `pandas.DataFrame`。替换数据源只需改 `StockDataFetcher.fetch()`，其余不动。
- **算法插件**：继承 `BaseModel`，实现 `fit(X, y)` 和 `predict(X)`，在 `ALGO_REGISTRY` 加一行即完成注册。可选依赖须在 `ALGO_AVAILABILITY` 登记开关，缺失时置灰而非崩溃。
- **深度学习**：继承 `TorchModelBase`，只实现 `_build_network(n_features)`；`__init__` 必须接受并透传 `window_size`。
- **多源容错**：东方财富（akshare）失败自动切 baostock；北交所代码（4/8/9 开头）只走 akshare，跳过不支持北交所的 baostock。
- **GUI 与业务解耦**：`MainWindow` 只调用上层 pipeline，不含业务逻辑，改算法/数据不需动 GUI。

---

## 不可触碰的红线

**违反以下任一条 = 制造"看起来很准、实际造假"的模型**，改代码前必须逐条自查（见 `AGENTS.md` 第 2 节完整清单）：

1. **时序不 shuffle**：必须用 `FeatureEngineer.time_series_split`（前段训练、后段测试）。交叉验证必须用前扩窗口 walk-forward（`_cross_validate`），不得用未来折数据训练。
2. **标准化只在训练集 fit**：`scaler.fit(X_train)` 后对验证/测试集只 `transform`，绝不在全量或测试集上 fit（见 `prepare_data`）。
3. **不删诚实基准**：`Naive(前值)` 朴素基准和 DA 方向准确率不得删除，任何"提升精度"的改动必须仍与基准公平对比。
4. **目标不进特征**：新增特征只能用"当天及以前可获得"的信息，未来收盘/涨跌绝不出现在 X 里。
5. **新增依赖必须可降级**：新库放进 `try/except ImportError`，缺失时降级而非崩溃。
6. **外部数据向后对齐**：`merge_asof(direction="backward")`，只取当天及以前已知的值；财报若做时序特征必须按公告日对齐（非报告期）。
7. **不荐股**：任何对外结论都必须带"非建议、盈亏自负"提示，不得引入"保证收益/买卖信号"表述。

---

## 常见改动配方

**新增算法**：写 `BaseModel` 子类 → 实现 `fit`/`predict` → 在 `ALGO_REGISTRY` 加一行 → 用 `--synthetic` 冒烟。

**新增评价指标**：在 `Metrics` 加 `@staticmethod(y_true, y_pred)` → 在 `_FUNC_MAP` 注册简称。

**换数据源**：改 `StockDataFetcher.fetch()` 或直接在外部构造好 DataFrame 传给 `TrainingPipeline(...).run_batch()`。

---

## Windows 路径注意

用户名含 `&` 时，VS Code 的 ▶ 按钮会让 PowerShell 报 `AmpersandNotAllowed`。解法：在终端用引号路径 `cd "D:\..."; python stock_predictor.py`，不要用 IDE 的运行按钮。
