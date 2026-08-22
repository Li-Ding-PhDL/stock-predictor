# 一键式 A 股预测研究软件 · stock-predictor

一个**单文件**（`stock_predictor.py`）的 A 股走势预测与可视化研究平台：一份代码里同时提供
**图形界面（GUI）**、**命令行（CLI）** 和 **可编程 API** 三种用法，内置 34 种算法、5 种超参数
优化方式、12 种评价指标，支持真实行情数据与离线合成数据。

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

本软件为此内置了两个诚实性机制，默认开启：

1. **Naive(前值) 朴素基准**：每次运行都会自动在结果里加一行 `Naive(前值)`。你的模型只有在
   **MAE / RMSE 明显低于**它时，才算真的学到了东西。
2. **DA 方向准确率**（Directional Accuracy）：统计"预测的涨跌方向"与"真实涨跌方向"一致的比例。
   随机瞎猜期望约 50%，**只有显著 > 50%** 才说明模型对"明天涨还是跌"有判别力。这个指标比 R² 更贴近实战。

看结果表时，请永远先看这两项。

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

### 默认数据源：akshare（免费、免注册）
本软件默认用 [akshare](https://akshare.akfamily.xyz/) 拉取 A 股日线行情（底层是东方财富/新浪等公开接口）：

```bash
pip install akshare
```
装好后，GUI 里"数据源"选"真实数据 (akshare)"，或 CLI 去掉 `--synthetic` 即可。
第一次拉取会缓存到 `data_cache/*.parquet`，之后同一区间秒开。

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
| 五 | 评价指标（12 种，含 DA） | `Metrics` |
| 六 | 34 种算法 + 可插拔注册表 | `BaseModel`, `ALGO_REGISTRY` |
| 七 | 超参数优化（PSO/GA/FA/SOA/BO） | `run_hpo`, `HPO_REGISTRY` |
| 八 | 训练调度 + 朴素基准 + 向前预测 | `TrainingPipeline` |
| 九 | PySide6 图形界面 | `MainWindow` |
| 十 | 可编程 API | `run_experiment` |
| 十一 | GUI/CLI 双模式入口 | `main`, `run_cli` |

关于其中 4 个"近似实现"算法（Kstar / M5Rules / GEP / MEP）的说明，见对应类的注释——它们在 Python
生态里没有和 WEKA/专业进化计算库完全对等的成熟实现，本软件用数学上合理的近似方案并**明确标注**。

---

作者：（Li-Ding-PhDL）· 仅供学术研究使用。
