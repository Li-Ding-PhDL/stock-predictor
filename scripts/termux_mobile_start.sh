#!/data/data/com.termux/files/usr/bin/bash
# Android Termux 一键启动手机端实时分析与同花顺悬浮窗看板（无需电脑开机）
# 用法：在手机 Termux 中进入仓库目录后执行 bash scripts/termux_mobile_start.sh 600519
set -e

CODE="${1:-600519}"
PORT="${2:-8765}"

if ! python -c "import numpy, pandas, sklearn, requests" >/dev/null 2>&1; then
  echo "[Termux] 正在检查并安装轻量计算依赖（无需安装 PySide6）..."
  pkg update -y || true
  pkg install -y python python-numpy python-scipy || true
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas scikit-learn requests
fi

echo "============================================================"
echo "手机本地服务启动中..."
echo "1) 手机浏览器打开:       http://127.0.0.1:${PORT}/"
echo "2) 安卓小窗/分屏挂同花顺: http://127.0.0.1:${PORT}/widget"
echo "⚠ 仅供研究与学习，不构成投资建议、盈亏自负。"
echo "============================================================"

exec python stock_predictor.py --web --web-host 127.0.0.1 --web-port "${PORT}" --code "${CODE}"
