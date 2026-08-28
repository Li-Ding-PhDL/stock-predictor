# 工程总览 · stock-predictor

> 一句话：单文件 A 股预测研究平台，**真实数据 + 多周期涨跌预测 + 三分防泄露 + 收益回测 + 风险提示 +
> 实时监控 + 批量扫描 + 综合报告**，全程守住「**绝不造假**」。

⚠️ **免责声明**：仅供机器学习/时间序列建模的教学与科研，**不构成任何投资建议**。市场接近有效，
任何模型的历史表现都不代表未来收益。

---

## 一、核心原则（贯穿全项目）

1. **只用真实数据**：所有行情/估值/资金/新闻均来自公开网站，缺失就跳过、**绝不填假数据**。
2. **不泄露未来**：严格时间序列切分（训练/验证/测试），标准化只在训练集 fit，外部数据按日期向后对齐。
3. **诚实评估**：默认预测涨跌幅（非价格，避免 R² 虚高）；每次带 `Naive(前值)` 和 `总是涨` 两条基准；
   看方向准确率 DA / 上涨精确率 UP_P，必须明显赢过基准才算有用。
4. **不臆造、不归因、不保证**：不预测下一分钟、不断言涨跌是「机构/主力/散户」造成的、不承诺预测更准。

---

## 二、功能全景（GUI 11 个页签）

| 页签 | 作用 |
|---|---|
| 行情K线图 | 拉真实日线画蜡烛图(红涨绿跌)+MA5/10/20+成交量 |
| 预测结果对比图 | 各模型测试集预测 vs 真实 |
| 未来预测图 | 1日/1周/1月/3月**多周期直接预测**曲线(可缩放)+每股预测价 |
| 指标结果表格 | 每格「训练/验证/测试」——一眼看过拟合；含 DA/UP_P |
| 策略回测 | 按预测涨跌做多/空仓、扣手续费，与买入持有比净值(超额/回撤/胜率/夏普) |
| 运行日志 / 操作日志 | 训练细节 / 用户操作留痕(带时间戳) |
| 机器学习内部 | **数据透明**：真实趋势三分图 + 各来源分析速览 + **风险提示** + **数据溯源(可点链接)** + 数据集预览 |
| 综合报告 | 汇总所有图表(内嵌图片)+数据，可预览、**导出自包含 HTML** |
| 实时监控 | 交易时段定时刷新真实盘口(买卖5档)+**自动记录快照到 monitor_log/** + 预测参考 |
| 批量扫描 | 多股批量 取数→训练→预测→风险，出可排序/导出的结果表 |

命令行/可编程：`run_experiment()`(单股) / `batch_scan()`(多股) / `forecast_curve()` / `backtest_directional()` /
`assess_risks()`，均可被脚本或其它 AI 直接 import 调用；CLI 见 `python stock_predictor.py -h`。

---

## 三、数据来源与溯源（全部真实公开）

| 数据 | 来源 | akshare 接口 |
|---|---|---|
| 日线行情 | 东方财富 / baostock(自动容错) | `stock_zh_a_hist` / baostock |
| 财报估值(PE/PB/市值,日频) | 乐咕乐股 / 百度股市通(自动适配) | `stock_a_indicator_lg` / `stock_zh_valuation_baidu` |
| 大盘环境(沪深300) | 东方财富 | `stock_zh_index_daily` |
| 主力资金流向(主力/大单净额) | 东方财富(带浏览器头直连+akshare兜底) | `stock_individual_fund_flow` / 直连API |
| 个股新闻 | 东方财富(仅展示,不训练) | `stock_news_em` |
| 实时盘口(买卖5档) | 东方财富 | `stock_bid_ask_em` |
| 周/月/季趋势 | 由日线滚动计算(无泄露) | — |

「机器学习内部」页有**数据溯源表**：为当前股票列出以上来源的**真实网址,点击在浏览器打开**逐一核对。

---

## 四、35 个算法（统一接口 `BaseModel` + 注册表 `ALGO_REGISTRY`）

- 基础与传统(14)：BP/ANN、SVR、LSSVM、GPR、ElasticNet、Ridge、Lasso、PLSR、KNN、Kstar*、ELM、DTR、DT、M5Rules*
- 先进与前沿(17)：RF、Bagging、ExtraTrees、AdaBoost、GBRT、XGBoost、LightGBM、CatBoost、
  LSTM、GRU、Transformer、TabNet、CNN、DNN、ResNet、BPNet、RBFNet
- 公式拟合/演化(3)：SR、GEP*、MEP*
- 经典时序(1)：ARIMA
- 超参优化：PSO / GA / FA / SOA / 贝叶斯BO(默认)

（\*号为数学上合理的近似实现，见对应类注释。缺可选依赖的算法自动置灰，不崩。）

---

## 五、安装与运行

一次性装齐（国内清华镜像，避免超时）：
```bash
pip install numpy pandas scikit-learn matplotlib pyarrow requests PySide6 akshare baostock statsmodels optuna xgboost lightgbm catboost gplearn torch pytorch-tabnet -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 8
```
运行（注意用户名含 `&` 时先 cd 再跑，别用 VSCode ▶ 按钮）：
```bash
cd "路径/stock-predictor"; python stock_predictor.py
```

---

## 六、诚实边界（做不到/不做的，讲清楚）

- ❌ 不预测下一分钟涨跌（分钟级是噪声）。
- ❌ 不断言「这波是机构/主力/散户/某新闻造成的」——单一因果归因免费数据无法可靠判定。
- ❌ 不承诺「预测更准」——真实股票上多数模型 DA 接近 50%，这是市场接近有效的真相。
- ✅ 能做的：真实数据、真实风险筛查(退市/亏损/暴跌/负面新闻)、诚实的因素现状、可回测的收益检验。

---

## 七、路线图（下一步可做）

1. **美股隔夜影响**（用户提出）：美股夜间交易会影响同板块 A 股情绪。可接入**隔夜美股指数(纳指/道指)涨跌**
   作为上下文特征——A 股 D 日用美股 D-1 夜间收盘(按日期向后对齐，不泄露)。需接 akshare 美股指数接口并验证，
   属真实可做的增强项。
2. **盘口数据建模**：`monitor_log/` 长期积累的真实盘口快照(委比/大单挂撤)达到统计量后，做盘中短线特征分析。
3. **批量实时监控**：把实时监控扩展到一批股票同时轮询记录。
4. **精确算法**：Kstar/M5Rules 接 python-weka-wrapper3、GEP 接 geppy；ARIMA 改 walk-forward 一步预测。

详见 `AGENTS.md`（给 AI 协作者的架构地图与红线）。

---

作者：（Li-Ding-PhDL）· 私有仓库 github.com/Li-Ding-PhDL/stock-predictor · 仅供学术研究。
