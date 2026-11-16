---
name: stock-monitor
description: Connect an AI assistant to stock-predictor for real-time A-share 5-level orderbook monitoring, causal intraday direction calculation, market limit-up sentiment tracking, and out-of-sample accuracy evaluation.
---

# stock-monitor · A股量化看盘与实盘校验 Skill

让 AI 助手（Antigravity / WorkBuddy / Trae / Claude Code 等）直接调度本仓库的 `stock_predictor.py`，完成 **盘中五档买手卖手采集、实时涨跌倾向测算、市场涨停情绪统计、估值周期体检与严格的因果准确率回放**。

## 核心指令清单（在项目根目录执行）

1. **启动手机看板 + 同花顺悬浮窗 + 盘中自动落盘采集**
   ```bash
   python stock_predictor.py --web --web-port 8765 --web-interval 10
   ```
   - 完整看板: `http://127.0.0.1:8765/`
   - 同花顺小窗悬浮条: `http://127.0.0.1:8765/widget`
   - 盘口回放准确率页: `http://127.0.0.1:8765/orderbook_eval`

2. **单股场内实时涨跌测算（五档买手卖手微结构 + 分钟级 ML）**
   ```bash
   python stock_predictor.py --intraday-predict 600519 --bar-period 5 --bar-horizon 3 --json intraday_pred.json
   ```

3. **收盘复盘：因果回放验证买手/卖手盘口准确率（对比多数类基准）**
   ```bash
   python stock_predictor.py --eval-orderbook --eval-horizon 60 --json monitor_log/orderbook_eval.json
   ```

4. **A股当日情绪温度计（涨停家数、封板率、连板高度、昨涨停今表现、热点行业）**
   ```bash
   python stock_predictor.py --market-mood --json monitor_log/market_mood.json
   ```

5. **日线估值温度与股性诊断 / 暴跌抄反弹扫描**
   ```bash
   python stock_predictor.py --cli --value-check --global-scope mine
   python stock_predictor.py --cli --dip-daily --global-scope mine
   ```

## 使用规范与诚实红线

- **绝不删减基准对比**：向用户汇报任何预测准确率时，必须同时给出**多数类基准（总是涨/总是跌）**及显著性检验结果。
- **严谨区分交易日与休市**：休市时段行情静止，盘中实时测算仅反映收盘静默状态。
- **合规边界**：所有输出结果仅供个人量化研究，**绝不构成买卖信号或投资建议、绝不承诺盈利或涨停概率、盈亏自负**。
