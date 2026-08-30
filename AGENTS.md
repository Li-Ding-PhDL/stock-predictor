# AGENTS.md — 给 AI 协作者的说明

> ⚠️ **红线（合规）**：本项目仅供交流与学习，**不构成投资建议、绝不荐股**；作者与本产品**不承担任何法律责任与风险**。
> 任何改动都不得引入"荐股/买卖信号/保证收益"的表述，所有对外结论都要带"非建议、盈亏自负"的提示。

本文件面向任何要**阅读、调用或修改** `stock_predictor.py` 的 AI 编码助手
（Claude Code / Cursor / Copilot / Qwen Code 等）。人类贡献者也可参考。

---

## 0. 一句话理解本项目

一个单文件的 A 股预测研究平台，靠一个**统一的模型接口 `BaseModel` + 注册表 `ALGO_REGISTRY`**
把 35 种算法(34 + ARIMA)插拔在同一套"取数→特征→训练→评估→可视化"流水线上。
默认预测**涨跌幅**(`target_mode="return"`)，评估统一还原成价格，并自带 `Naive(前值)` 与 `总是涨(方向基准)` 两条诚实性基准。

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
   用 sklearn 默认 shuffle 会用未来预测过去，是严重的数据泄漏。**交叉验证同理**：`_cross_validate` 必须用
   **前扩窗口 walk-forward**（只用验证折之前的数据训练），绝不能把验证折之后的未来折也拿去训练。
2. **标准化器只在训练集上 `fit`**，测试集只 `transform`（见 `prepare_data`）。绝不在全量或测试集上 fit。
3. **绝不删除 Naive(前值) 朴素基准和 DA 方向准确率**，也不得为了让数字好看而绕过它们。
   任何"提升精度"的改动，都必须仍然和朴素基准做公平对比。
4. **不得把目标价格泄漏进特征**。当前特征只用"当天及以前可获得"的信息，新增指标时保持这一点。
5. 新增依赖必须放进 `try/except ImportError` 的可选依赖区，缺失时**降级而非崩溃**。
6. **财务数据点时(point-in-time)**：F-Score/ROE/营收增速等财报字段目前只用于研判卡展示与横截面因子打分(非时序训练特征)，无泄漏。
   **若将来把它们做成训练特征，必须按『公告日』而非『报告期』对齐**(财报有披露滞后)，否则会用到未来数据 = 泄漏。
7. **幸存者偏差**：单票回测只覆盖『还在上市』的标的，会系统性高估收益。任何"策略有效"的结论都要在界面/文档如实标注此偏差。

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
- **✅ 收益率预测模式**（已完成）：`target_mode="return"` 已实现并设为默认，评估统一还原成价格。
- **✅ ARIMA 经典时序模型**（已完成）：可选 `statsmodels`；目前为多步 forecast，可进一步做 walk-forward 一步预测。
- **✅ 多周期预测**（已完成）：horizon 支持 1/3/5/20/60 日；DA/UP_P 与两条基准在管道内用每个样本的
  `prev_close` 计算，对所有周期都正确（`TrainingPipeline._dir_metrics`）。GUI「预测周期」下拉可选。
- **✅ 周/月/季趋势特征**（已完成）：`add_technical_indicators` 4.1.9，由日线滚动计算，无泄露。
- **✅ 外部数据接入**（已完成）：`StockDataFetcher.enrich()` 用 merge_asof 并入财报估值(乐咕乐股
  `stock_a_indicator_lg`)与大盘环境(沪深300 `stock_zh_index_daily`)；新闻(`stock_news_em`)仅展示不训练。
  失败自动跳过、绝不造假。**注意 merge_asof 两侧 date 必须同分辨率，已统一 datetime64[ns]。**
- **✅ 隔夜美股 / 北向资金**（已完成）：`enrich()` 已并入(`index_us_stock_sina` 日期+1对齐、`stock_hsgt_north_net_flow_in_em`)。
- **✅ 未来一周·逐日预测 + 真实交易日历 + 置信区间**（已完成）：`forecast_curve(horizons=(1,2,3,4,5))` 逐日直接预测(非递归)；
  `next_trading_days()`/`_load_trade_calendar()`(akshare `tool_trade_date_hist_sina`，全会话缓存)跳周末+节假日，失败退回 BDay；
  每点带约80%置信区间(σ·√h)。CLI `--week`。`is_trading_now` 也用日历排节假日。
- **✅ 回测贴合 A 股制度**（已完成）：`backtest_directional` 逐样本 `bar_info`(涨跌停/停牌)→涨停买不进/停牌/跌停卖不出/T+1；
  `board_limit_pct` 按板块定幅度；波动率目标仓位 `vol_target_annual` + 单段止损 `stop_loss_pct`；对比沪深300超额(`fetch_index_close`)。
- **✅ 统计严谨性**（已完成）：`da_significance()` 二项检验(无 scipy 用 erfc)；研判卡"可信度"须过基准+显著；推荐区加多重比较偏差警示。
- **✅ 因子选股 5 因子**（已完成）：`batch_factor_scan` = 价值/动量/资金 + 质量(`fetch_quality` ROE+营收增速) + 低波动。
- **市场情绪 / 舆情**：量价情绪代理已有；真正的新闻/股吧 NLP 情绪仍是扩展——务必真实抓取、不臆造分数。
- **板块/行业**：可加行业分类 + 板块指数(`stock_board_industry_hist_em`)，按 enrich 同款 merge_asof 并入。
- **✅ 组合与仓位 / 因子有效性 / 校准 / 波动率**（已完成，借鉴交易 skill）：GUI「组合与仓位」页 = 相关性分散化 `basket_correlation` + 凯利仓位 `kelly_fraction` + 因子有效性 `factor_ic_test`(IC/ICIR，纯价格因子无泄漏)；预测跟踪「区间覆盖率」校准(pred_lo/hi→in_interval)；`estimate_daily_vol`(EWMA/GARCH)改进置信区间。
- **✅ Walk-Forward / 均值回归股性**（已完成）：`walk_forward_eval`(每折 refit 无泄漏，GUI「机器学习内部」按钮)；`stock_character`(Hurst+z-score 判均值回归/趋势，接入研判卡)。
- **借鉴来源(致谢)**：staskh/trading_skills(Piotroski/财报日历)、yennanliu/InvestSkill(基本面结构化)、agiprolabs/claude-trading-skills(凯利/相关性/组合/IC/WalkForward/波动率/均值回归)、tradermonty/claude-trading-skills(RS/regime)。均只参考公开方法学、未复制代码。
- **✅ 对照 MATLAB RS5 参考平台**（已核对）：Python 的 34 模型/5 优化器已是**真实现**(sklearn/torch/xgboost/lightgbm/catboost/statsmodels + 标准 PSO/GA/真萤火虫/真海鸥/optuna 贝叶斯)；LSSVM=KernelRidge(rbf) 与 RS5 的 `(K+γI)\y` 数学等价；FA/SOA/BO 比 RS5 紧凑器(退化为自适应搜索/启发式)**更faithful**，故不降级对齐。仅借鉴 RS5 的 **HPO『默认超参兜底』**(run_hpo：寻优不优于默认则回退，防小样本过拟合验证集)。RS5 的 b01Train 的抗外推钳制(预测限训练范围±30%)可作为后续可选安全阀。
- **待做**：实时主力资金流(后端 `fetch_realtime_fundflow` 已备，UI 未接；用户要求 skill 做完后再完善)；ARIMA 滚动一步；组合层面收益回测/行业中性；龙虎榜/融资融券/解禁/商誉/质押因子；NLP 舆情；`Kstar/M5Rules/GEP/MEP` 精确实现。

改进时请回到第 2 节红线核对一遍。
