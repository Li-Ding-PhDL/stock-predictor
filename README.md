# 一键式 A 股预测研究软件 · stock-predictor

一个**单文件**（`stock_predictor.py`）的 A 股走势预测与可视化研究平台：一份代码里同时提供
**图形界面（GUI）**、**命令行（CLI）** 和 **可编程 API** 三种用法，内置 35 种算法(含 ARIMA)、5 种超参数
优化方式、多种评价指标(含方向准确率/上涨精确率)，默认**预测涨跌幅**并自带诚实性基准，支持真实行情与合成数据。

> ⚠️ **免责声明**：本软件仅用于机器学习/时间序列建模的**教学与科研**。股票市场受大量不可建模因素
> 影响，任何模型的历史回测表现都**不代表**未来收益。软件输出**不构成任何投资建议**，据此交易风险自负。

---

## 目录
- [为什么要认真看"诚实性"这一节](#诚实性最重要)
- [安装](#安装)
- [三种用法](#三种用法)
- [如何接入真实数据](#如何接入真实数据)
- [如何让其它 AI 调用并完善代码](#如何让其它-ai-调用并完善代码)
- [软件结构](#软件结构)

---

## 诚实性（最重要）

股价是**强自相关**序列。"用今天的收盘价当作明天的预测值"这种什么都没学的**朴素基准**，在"预测价格
水平"任务上就能轻松拿到 **R² > 0.95**。所以：

> **单看某个模型 R²=0.98 是没有意义的。** 必须和朴素基准对比。

本软件为此内置了一整套诚实性机制，默认开启：

1. **默认预测「涨跌幅」而非「价格水平」**（`target_mode="return"`）：价格自相关极强，预测价格会得到
   虚高 R²；预测涨跌幅更平稳、也更贴近"判断涨跌"的真实目标。无论哪种模式，评估与画图都会**统一还原成
   价格**来比较，口径一致。
2. **两条基准线，自动加入每次结果**：
   - `Naive(前值)`：预测明天=今天。价格误差(MAE/RMSE)的参照——模型必须**明显低于**它。
   - `总是涨(方向基准)`：每天都猜涨，它的方向准确率就等于"上涨日占比"。模型的方向指标必须**明显高于**它。
3. **方向类指标（判断涨跌的关键）**：
   - **DA 方向准确率**：预测涨跌方向与真实一致的比例。
   - **UP_P 上涨精确率**：模型**说涨**的那些天里真的涨了的比例——因为你只在"模型说涨"时买入，这个最实用。

看结果表时，请永远先看这三行/几列：**价格上赢过 Naive、方向上赢过"总是涨"，才算真有用。**
在真实股票上，多数模型的 DA 会很接近 50%——这就是诚实的现实，不必粉饰。

---

## 安装

```bash
# 1) 克隆仓库
git clone https://github.com/Li-Ding-PhDL/stock-predictor.git
cd stock-predictor

# 2) 安装核心依赖（必须）
pip install numpy pandas scikit-learn matplotlib pyarrow

# 3) 按需安装可选依赖
pip install PySide6            # 想用图形界面
pip install akshare            # 想用真实 A 股数据
pip install xgboost lightgbm catboost torch pytorch-tabnet gplearn optuna   # 更多算法

# 或一次性安装 requirements.txt 里未注释的部分
pip install -r requirements.txt
```

缺失可选依赖时程序**不会崩溃**：对应算法在界面里自动置灰，akshare 缺失则自动改用合成数据。

---

## 三种用法

### 1) 图形界面（最直观）
```bash
python stock_predictor.py
```
左侧配置股票代码/日期/数据源、勾选算法、选 HPO/评价指标/训练比例，点"运行"，右侧看对比图、指标表、日志。

### 2) 命令行（适合批量实验 / 服务器）
```bash
# 看全部参数
python stock_predictor.py -h

# 合成数据快速跑通
python stock_predictor.py --cli --synthetic --algos "RF,SVR,GBRT"

# 真实数据 + 向前预测下一交易日 + 导出 JSON
python stock_predictor.py --cli --code 600519 --start 20200101 \
    --algos "RF,XGBoost,LightGBM" --forecast --json result.json
```

### 3) 可编程 API（供脚本 / 其它 AI 调用）
```python
from stock_predictor import run_experiment

out = run_experiment(
    code="600519", start="20200101",
    algos=["RF", "XGBoost"], forecast=True,
)
for r in out["results"]:
    print(r["algo"], r["metrics"])
print(out["forecast"])
```
`run_experiment(...)` 返回一个可直接 `json.dumps` 的纯字典，不依赖任何 GUI，见下一节。

---

## 如何接入真实数据

### 默认数据源：多源自动容错（东方财富 → baostock）
本软件默认用 [akshare](https://akshare.akfamily.xyz/)(东方财富) 拉日线行情，**拉不到会自动重试并切换到
[baostock](http://baostock.com/)**(国内稳定、免注册)。强烈建议两个都装：

```bash
pip install akshare baostock
```
装好后，GUI 里"数据源"选"真实数据 (自动: 东方财富/baostock)"，或 CLI 去掉 `--synthetic` 即可。
第一次拉取会缓存到 `data_cache/*.parquet`，之后同一区间秒开。

> 若开着 VPN/代理导致 `ProxyError`：软件默认会**自动绕过系统代理**直连国内行情站（东方财富/baostock
> 都是国内服务器）。若你的网络反而必须走代理才能出网，可在代码里以 `fetch(..., bypass_proxy=False)` 调用。

### 外部数据接入（全部真实来源，失败自动跳过、绝不填假数据）
在 GUI「外部数据接入」区勾选后，真实数据模式下会自动按日期对齐并入以下真实数据：

| 输入 | 真实来源 | akshare 接口 | 说明 |
|---|---|---|---|
| 日线行情 | 东方财富 / baostock | `stock_zh_a_hist` / baostock | 主行情 |
| 周/月/季趋势 | 由日线滚动计算 | — | 各周期涨跌/区间位置/均线偏离，无泄露 |
| 财报估值(日频) | 乐咕乐股 legulegu.com | `stock_a_indicator_lg` | PE/PB/PS/股息率/总市值 |
| 大盘环境 | 东方财富 | `stock_zh_index_daily` | 沪深300 涨跌/均线偏离 |
| **主力资金/主力意图** | 东方财富 | `stock_individual_fund_flow` | 主力/超大单/大单净流入 + 派生：连续进/出场天数、累计净流入、资金-价格背离(吸筹洗盘 vs 派发 代理) |
| 个股新闻 | 东方财富 | `stock_news_em` | **仅在"机器学习内部"页展示真实标题+链接，不作历史训练特征**(避免臆造情绪分) |

> **主力意图 & 买卖手（诚实说明）**：不臆造"主力预言"，而是接入**真实每日主力资金净流入**并派生
> "进场/退场/洗盘"的可学习代理特征，由模型自己学。实时盘口"买1-5/卖1-5(买卖手)"是**快照、无免费历史**，
> 用于回测会泄露，故用可回测的"每日主力/大单净额"作其历史等价物。

> 诚实说明：新闻/舆情要变成可靠的**历史**情绪分需要 NLP 模型且历史覆盖不足，软件不编造分数喂给模型。
> 板块/行业指数为进一步扩展项（见 AGENTS.md）。所有对齐都用 `merge_asof` 向后取最近已知值，**不泄露未来**。

### 多周期涨跌预测
GUI「预测周期」可选 **第二天(1) / 3日 / 一周(5) / 一个月(20) / 三个月(60)**；CLI 用 `--horizon`。
方向准确率(DA)、上涨精确率(UP_P)、以及两条基准，对**所有周期**都用"每个样本的基准日收盘价"正确计算。

对应代码在 `StockDataFetcher.fetch()`（第三部分），核心一行：
```python
df = ak.stock_zh_a_hist(symbol=code, period="daily",
                        start_date=start_date, end_date=end_date, adjust="qfq")
```
- `code`：如 `"600519"`（贵州茅台），不带市场前缀，akshare 自动识别沪深。
- `adjust`：`"qfq"` 前复权（**训练推荐**）/ `"hfq"` 后复权 / `""` 不复权。

### 换成别的数据源（Tushare / 券商 API / 本地 CSV）
只要最终产出一个含 **`date, open, close, high, low, volume`** 这几列的 `pandas.DataFrame`
（按日期升序），就能无缝喂给后续所有流程。三种常见接法：

1. **本地 CSV / Excel**（最简单，不联网）：
   ```python
   import pandas as pd
   from stock_predictor import TrainingPipeline, TrainConfig
   raw_df = pd.read_csv("my_stock.csv", parse_dates=["date"]).sort_values("date")
   # 确保列名是 date/open/close/high/low/volume，不是就先 rename
   results = TrainingPipeline(TrainConfig()).run_batch(["RF", "SVR"], raw_df)
   ```

2. **Tushare**（需注册拿 token）：
   ```python
   import tushare as ts
   pro = ts.pro_api("你的token")
   df = pro.daily(ts_code="600519.SH", start_date="20200101", end_date="20241231")
   df = df.rename(columns={"trade_date": "date", "vol": "volume"})
   df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date")
   ```

3. **改造 `StockDataFetcher`**：直接在 `fetch()` 里替换成你的数据源，保持返回的列名不变即可，
   GUI/CLI/API 全部无需改动。

> 关键约束：**列名统一为英文 `date/open/close/high/low/volume`**，且**按时间升序**。
> 特征工程会自动算出 MA/MACD/RSI/布林带/KDJ 等指标，你不需要自己算。

---

## 如何让其它 AI 调用并完善代码

本仓库特意设计成**对 AI 协作友好**。请配合根目录的 [`AGENTS.md`](AGENTS.md) 一起看——那里给出了
架构地图、扩展点、以及"新增一个算法/指标/数据源"的分步配方与不可触碰的红线。

**给其它 AI（Claude Code / Cursor / Copilot / Qwen 等）的最短上手路径：**

1. **调用**：让 AI 读 `run_experiment()` 的 docstring，然后写脚本调用它、解析返回的 JSON——
   不需要理解 2000 行全文即可使用本软件。
2. **扩展算法**：继承 `BaseModel`，实现 `fit/predict`（可选 `get_hpo_space`），在 `ALGO_REGISTRY`
   里加一行注册，GUI 会**自动**出现对应勾选框。见 `AGENTS.md` 的"新增算法配方"。
3. **红线**：绝不允许用随机 shuffle 切分时间序列、绝不在测试集上 fit 标准化器、绝不删除 Naive 基准与
   DA 指标来"让数字更好看"。任何"提升精度"的改动都必须同时对朴素基准保持公平对比。

一个可以直接丢给其它 AI 的任务示例：
> "请阅读本仓库的 AGENTS.md，在 `stock_predictor.py` 中新增一个 `Informer` 时序模型，
> 继承 TorchModelBase，注册进 ALGO_REGISTRY，并用 `python stock_predictor.py --cli --synthetic
> --algos Informer` 自测跑通。不要改动时间序列切分与朴素基准逻辑。"

---

## 软件结构

`stock_predictor.py` 分为十一个部分（文件内均有分节注释）：

| 部分 | 内容 | 关键类/函数 |
|---|---|---|
| 一~二 | 依赖导入、全局配置 | `TrainConfig` |
| 三 | 数据获取（真实/合成/缓存） | `StockDataFetcher` |
| 四 | 特征工程（技术指标+滑窗+切分） | `FeatureEngineer` |
| 五 | 评价指标（含 DA 方向准确率 / UP_P 上涨精确率） | `Metrics` |
| 六 | 35 种算法(34 + ARIMA) + 可插拔注册表 | `BaseModel`, `ALGO_REGISTRY` |
| 七 | 超参数优化（PSO/GA/FA/SOA/BO） | `run_hpo`, `HPO_REGISTRY` |
| 八 | 训练调度 + 朴素基准 + 向前预测 | `TrainingPipeline` |
| 九 | PySide6 图形界面 | `MainWindow` |
| 十 | 可编程 API | `run_experiment` |
| 十一 | GUI/CLI 双模式入口 | `main`, `run_cli` |

关于其中 4 个"近似实现"算法（Kstar / M5Rules / GEP / MEP）的说明，见对应类的注释——它们在 Python
生态里没有和 WEKA/专业进化计算库完全对等的成熟实现，本软件用数学上合理的近似方案并**明确标注**。

---

作者：（Li-Ding-PhDL）· 仅供学术研究使用。
