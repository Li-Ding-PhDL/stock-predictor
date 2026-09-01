#!/usr/bin/env bash
# 每日自动验证到期预测（对应量化 JD 的 Bash/自动化要求）。
# 用法：把本脚本加入 crontab，收盘后自动用真实股价核对到期预测、更新真实准确率——无需手动开软件。
#
# 安装到 crontab（每个交易日 15:30 跑一次；周末照跑但脚本内会因无新数据而无操作）：
#   crontab -e
#   30 15 * * 1-5  /path/to/stock-predictor/scripts/daily_verify.sh >> /path/to/stock-predictor/verify.log 2>&1
#
# 说明：--verify 只核对已到目标日的预测；取不到真实行情就跳过、下次再试，绝不用假数据填充。
set -euo pipefail

# 定位到项目根目录（脚本所在目录的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 优先用项目虚拟环境的 python，否则用系统 python3
PY="python3"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始自动验证到期预测 ..."
"$PY" stock_predictor.py --verify
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 自动验证结束。"
