# stock-predictor —— 命令行/定时任务用 Docker 镜像（对应量化 JD 的 Linux/Docker 要求）
# 说明：GUI(PySide6) 需要显示环境，容器里主要用命令行(--cli / --verify / --snapshot / --week / --backtest)。
# 构建：  docker build -t stock-predictor .
# 冒烟：  docker run --rm stock-predictor --cli --synthetic --algos "RF,SVR,GBRT"
# 定时验证到期预测(挂载本地 predictions 目录持久化)：
#   docker run --rm -v "$PWD/predictions:/app/predictions" stock-predictor --verify
FROM python:3.12-slim

# 时区设为上海（A 股交易日历/日志对齐）
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

# 只装命令行跑通所需的核心依赖（不装 PySide6/torch 等重/GUI 依赖，缺失会自动降级不崩溃）
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn matplotlib pyarrow requests \
    akshare baostock statsmodels

COPY stock_predictor.py README.md AGENTS.md PROJECT.md requirements.txt ./

# 默认入口=命令行；docker run 追加的参数原样传给它
ENTRYPOINT ["python", "stock_predictor.py"]
CMD ["--cli", "--synthetic", "--algos", "RF,SVR,GBRT"]
