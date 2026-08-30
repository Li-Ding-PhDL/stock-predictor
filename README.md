# stock-predictor · 一键式 A 股预测研究平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PySide6-41cd52)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **免责声明 / 红线（务必先读）**
> - 本项目**仅供交流与学习，不构成任何投资建议，也绝不荐股**；作者与本产品**不承担任何法律责任与风险**，据此交易盈亏自负。
> - 真实 A 股上，多数模型的方向准确率≈50%（和抛硬币差不多）。本软件的价值**不在"预测得准"，而在于诚实地告诉你到底准不准、并帮你避开明显的雷**。
> - 请仅用于个人研究：勿高频抓取公开接口、勿转售/再分发数据、勿无「证券投资咨询」牌照对外荐股或收费（在中国属违法）。
> - **任何"保证高准确率""稳赚不赔"的股票预测都是骗局。**

一个**单文件**（`stock_predictor.py`，约 6000 行）的沪深 A 股预测研究平台，用一套统一的
「取数 → 特征 → 训练 → 评估 → 可视化」流水线，把 **35 种算法**插拔在一起，
提供 **图形界面（GUI）/ 命令行（CLI）/ 可编程 API** 三种用法。

它是照着作者的 MATLAB「MIMO 多输出研究平台」做的 Python 版，最大的设计信条是 **诚实优先（radical honesty）**。

---

## ✨ 核心特性

### 🎯 诚实优先，绝不自欺
- **默认预测"涨跌幅"而非"价格"**：预测价格会让 R² 虚高到 0.9+（因为"明天价≈今天价"），是假象；预测涨跌幅更平稳、更诚实。
- **两条强制基准**：`Naive(前值)`（价格误差参照）+ `总是涨(方向基准)`（DA=历史上涨日占比）。模型必须明显超过它们才算有用。
- **方向准确率 DA + 上涨精确率 UP_P**：比 R² 更能反映"能不能真赚钱"。
- **DA 二项显著性检验**：DA=53% 到底是真本事还是运气？给出 p 值，样本太少直接标"可能只是运气"。
- **多重比较偏差警示**：比了一堆模型再挑最好的，本身就偏乐观——软件会提醒你去做样本外验证。
- **预测有置信区间**：未来预测给约 80% 置信带（点预测必错，区间才诚实）。

### 📊 全部真实数据、可溯源、绝不造假
| 数据 | 来源 | 接口 |
|---|---|---|
| 日线行情 | 东方财富 / baostock（多源自动容错） | `stock_zh_a_hist` / baostock |
| 财报估值 (PE/PB/市值) | 百度 / 乐咕乐股（自动适配） | `stock_zh_valuation_baidu` / `stock_a_indicator_lg` |
| 大盘环境 | 东方财富（沪深300） | `stock_zh_index_daily` |
| 主力资金流 | 东方财富（带浏览器头直连） | 东财资金流 API |
| 隔夜美股 / 北向资金 | 新浪 / 东方财富 | `index_us_stock_sina` / `stock_hsgt_north_net_flow_in_em` |
| 个股新闻 | 东方财富 | `stock_news_em`（仅展示 + 风险关键词筛查，不作训练特征） |
| 实时盘口 | 东方财富 | 盘口五档（监控用，自采集到 `monitor_log/`） |
| **交易日历** | 东方财富（新浪日历） | `tool_trade_date_hist_sina`（真实节假日，用于未来日期/到期日） |

> **任一来源取数失败 → 如实标注"未接入"、该特征缺省，绝不填假值、绝不编情绪分、绝不编因果。**

### 🧠 35 种算法 + 5 种超参优化
- 传统/集成：SVR、LSSVM、GPR、Lasso、Ridge、ElasticNet、PLSR、KNN、RF、ExtraTrees、Bagging、AdaBoost、GBRT、XGBoost、LightGBM、CatBoost、决策树…
- 深度学习：LSTM、GRU、Transformer、TabNet、CNN、DNN、ResNet、BPNet、RBFNet
- 公式演化：符号回归 SR、GEP、MEP
- 经典时序：ARIMA
- 超参优化：PSO / GA / FA / SOA / **BO(贝叶斯，默认)**
- 缺少可选依赖（xgboost/torch/gplearn 等）会**自动降级、不崩溃**，界面对应算法置灰并注明缺哪个库。

### 📈 未来一周·逐日预测
对**下 1~5 个交易日（约下周一~周五）各自独立训练一个"直接预测第 h 天"的模型**，
得到每天的预测价 + 涨跌 + 约 80% 置信区间。**不做递归预测**（拿预测喂预测会误差爆炸），**不编造未来的开高低量**。
日期取自**真实交易日历**，自动跳过周末与法定节假日。

### 💰 贴合 A 股制度的收益回测
「预测涨就满仓、预测跌就空仓」走一遍历史，扣手续费，和"买入持有"及"沪深300大盘"对比净值：
- **T+1**（持有≥1日天然满足）、**涨停买不进 / 跌停卖不出 / 停牌不可交易**（按板块自动涨跌停幅度：主板10%、创业板/科创板20%、北交所30%、ST 5%）
- **风控**：波动率目标仓位（高波动自动减仓）、单段止损
- 输出：年化、最大回撤、胜率、夏普（扣无风险利率）、相对大盘超额
> 只有**明显跑赢买入持有和大盘**，才说明预测涨跌真的能赚钱。

### 🏆 因子打分选股（横截面，5 因子）
价值（低 PE/PB）+ 动量（近 3 月）+ 资金（主力净流入）+ **质量（ROE/营收增速）** + **低波动**，
百分位排名加权、扣风险惩罚后排序。**这是研究筛查工具，不是荐股**；排在前面 ≠ 该买。

### 🔎 其它
- **预测跟踪**：把预测存下来，等目标日到了自动拉真实股价对比，算**真实**命中率（附记录时 DA，增加可信力）。
- **风险自动筛查**：ST/*ST、面值退市、亏损、近期暴跌、负面新闻关键词——每条都注明真实依据。
- **综合研判卡 + 综合报告**：把客观信号汇总成一页 HTML（可导出），含"模型对后市的机械倾向"（并列可信度，绝不当买卖信号）。
- **实时监控**：交易时段轮询盘口，记录快照供事后分析。
- **名词解释**：41+ 金融/模型名词 + 35 算法 + 14 指标的大白话解释（写给不懂金融的用户）。

---

## 🚀 快速开始

### 安装
```bash
pip install -r requirements.txt
# 国内可加清华镜像： -i https://pypi.tuna.tsinghua.edu.cn/simple
```
最小可跑（不含深度学习/符号回归）：`numpy pandas scikit-learn matplotlib pyarrow requests PySide6 akshare baostock`

### 图形界面
```bash
python stock_predictor.py
```

### 命令行（无界面，适合脚本/定时任务）
```bash
# 合成数据秒级自测
python stock_predictor.py --cli --synthetic --algos "RF,SVR,GBRT"

# 真实股票 + 未来一周逐日预测 + 带手续费回测
python stock_predictor.py --cli --code 600519 --week --backtest
```

> **Windows 用户注意**：若用户名文件夹含 `&`（例如 `name&co` 这类带 & 的用户名），VS Code 的 ▶ 运行按钮拼出的路径会让 PowerShell 报
> `AmpersandNotAllowed`。解法：别用 ▶，在终端里 `cd "带引号的路径"; python stock_predictor.py`。

---

## 📁 目录结构
```
stock-predictor/
├── stock_predictor.py   # 全部功能（单文件，约6000行）
├── README.md            # 本文件
├── PROJECT.md           # 5分钟工程总览
├── AGENTS.md            # 给 AI 协作者的改代码指南（含"不可触碰的红线"）
├── requirements.txt     # 依赖（分层，缺可选项自动降级）
└── .gitignore           # 行情缓存/快照/预测日志/报告 等本地产物不入库
```

---

## 🤝 给 AI / 开发者协作
本项目欢迎 AI 编码助手（Claude Code / Cursor / Copilot 等）参与。**改代码前请务必先读 [AGENTS.md](AGENTS.md)**，
尤其是"不可触碰的红线"（时序不能 shuffle、标准化只在训练集 fit、不删诚实基准、不泄漏未来、可选依赖降级不崩溃）。

---

## 🙏 借鉴的开源 Skills（致谢）

本项目的一部分「客观分析/风控/验真」功能，参考并用 Python 独立实现了以下开源 Claude Skills 里的**公开方法学**
（均为标准公开公式/思路，本项目未复制其代码，且全部按 A 股与本项目「诚实优先」原则重写；相关参考资料未纳入本仓库）：

| 借鉴来源(GitHub) | 借鉴到本项目的功能 |
|---|---|
| **staskh/trading_skills** | Piotroski F-Score(9 分基本面质量)、财报披露日历(临近财报事件提示) |
| **yennanliu/InvestSkill** | 基本面结构化分析思路(Piotroski/质量因子) |
| **agiprolabs/claude-trading-skills** | 凯利公式仓位、相关性/组合分散化、因子有效性 IC/ICIR、Walk-Forward 滚动前推验证、EWMA/GARCH 波动率建模、均值回归股性诊断(Hurst) |
| **tradermonty/claude-trading-skills** | 相对强弱 RS(CANSLIM「买领涨」)、大盘状态 regime |

> 说明：这些方法都是**客观统计/会计指标**，本项目只用于「陈列事实 + 验证真伪」，**不产出买卖信号、不荐股**。
> crypto/DeFi/美股期权/券商下单等与 A 股无关的 skill 未借鉴。

## ⚖️ 合规与许可
- 本项目采用 [MIT 许可证](LICENSE)。
- 数据来自公开接口，仅供个人学术研究。**不构成投资建议，不提供荐股服务。**
- 做成对外产品前请咨询证券合规律师；无「证券投资咨询」牌照对外荐股/收费在中国属违法。

## 👨‍💻 作者
- GitHub [@Li-Ding-PhDL](https://github.com/Li-Ding-PhDL)

---
*任何"保证高准确率""稳赚不赔"的股票预测都是骗局。本软件的立场是：市场接近有效，诚实比虚假的"准"更有价值。*
