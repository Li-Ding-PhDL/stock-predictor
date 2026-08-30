#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
                    一键式 A 股预测研究软件  |  stock_predictor.py
========================================================================================

【软件说明】
    本软件是一个单文件（single-file）的 A 股股票走势预测与可视化平台，功能包括：
        1. 数据获取层   —— 东方财富(akshare)/baostock 多源自动容错拉取行情+本地缓存；可接入
                          真实财报估值/大盘环境/主力资金流向(缺失自动跳过、绝不造假)
        2. 特征工程层   —— 技术指标 + 周/月/季多周期趋势 + 量价情绪 + 主力意图代理 + 滑窗样本
        3. 模型层       —— 35 种机器学习/深度学习/公式拟合/经典时序(ARIMA)算法，统一接口可插拔
        4. 超参数优化层 —— PSO / GA / FA / SOA / 贝叶斯优化(BO，默认推荐) 五种寻优方式
        5. 训练评估层   —— 训练/验证/测试三分(验证集独立留出) + 多周期(1/3/5/20/60日)涨跌预测
                          + 方向准确率/上涨精确率 + 朴素/总是涨双基准 + 方向性收益回测
        6. 可视化 GUI 层 —— 基于 PySide6：K线/预测对比/未来走势/指标(训练验证测试)/策略回测/
                          运行日志/操作日志/机器学习内部(数据透明) 八个页签

【关于依赖库】
    本文件涉及的第三方库分为"核心依赖"和"可选依赖"两类：
        - 核心依赖（必须安装）：numpy, pandas, scikit-learn, matplotlib, PySide6
        - 可选依赖（缺失会自动降级，不影响整体运行，只是对应算法/功能不可用）：
              akshare      —— 真实 A 股数据获取（缺失则只能使用内置合成数据做演示/测试）
              xgboost      —— XGBoost 算法
              lightgbm     —— LightGBM 算法
              catboost     —— CatBoost 算法
              torch        —— LSTM/GRU/Transformer/CNN/DNN/ResNet/BPNet 等深度学习算法
              pytorch_tabnet —— TabNet 算法（若缺失则用等价 DNN 结构近似替代）
              gplearn      —— 符号回归 SR（以及作为 GEP/MEP 缺失时的近似替代）
              optuna       —— 贝叶斯超参数优化（若缺失则自动退化为随机搜索）

    安装建议（在你自己的电脑上执行，本沙盒环境仅用于开发调试）：
        pip install akshare xgboost lightgbm catboost torch pytorch-tabnet gplearn optuna PySide6

【关于"近似实现"的说明】
    35 个算法中，绝大多数（RF/XGBoost/LightGBM/SVR/GPR/LSTM/Transformer/ARIMA 等）都有
    公认的标准 Python 实现，本文件直接调用对应库，结果是"真实"的。
    但其中 4 个算法（KStar、M5Rules、GEP、MEP）在 Python 生态中没有和 WEKA / 专业
    进化计算库完全对等的成熟实现，本文件采用了数学上合理的近似方案，并在对应类的
    注释中明确标注"近似实现"字样和原因，方便你之后替换为更专业的库（如
    python-weka-wrapper3、geppy）。

========================================================================================
"""

# ==================== 第一部分：依赖导入与全局配置 ====================
# ------------------------------------------------------------------------------
# 1.1 标准库导入
# ------------------------------------------------------------------------------
import os                      # 文件路径处理
import sys                     # 系统相关（GUI 程序入口需要）
import io                      # 综合报告：把图表存成内存字节流再转 base64
import base64                  # 综合报告：图表转 base64 内嵌进 HTML（导出自包含网页）
import json                    # 配置/缓存的序列化
import argparse                # 命令行/无界面模式参数解析（方便脚本或其它 AI 程序化调用）
import time                    # 计时、生成随机种子
import warnings                # 屏蔽第三方库的冗余警告信息
import contextlib              # 临时忽略系统代理时用到的上下文管理器
import datetime as dt          # 日期处理（股票行情按日期索引）
from abc import ABC, abstractmethod            # 定义模型统一接口的抽象基类
from dataclasses import dataclass, field       # 简化配置类的书写
from typing import Optional, List, Dict, Tuple, Callable, Any

warnings.filterwarnings("ignore")  # 训练过程中大量的收敛警告/弃用警告不影响使用，统一屏蔽

# ------------------------------------------------------------------------------
# 1.2 核心第三方库导入（数值计算 / 数据处理 / 绘图 / 传统机器学习）
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# 让 matplotlib 正常显示中文标签与负号（否则图里的中文会变成方框、负号会缺失）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                          "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
# 注意：FigureCanvasQTAgg 依赖 PySide6/PyQt 才能导入，放到"可选依赖"区域（1.3.7）里按需导入，
# 这样在没装 GUI 库、只想用本文件做命令行训练/批量实验的场景下，导入本文件不会直接报错。

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------------------
# 1.3 可选第三方库导入（缺失时自动降级，不让整个程序崩溃）
# ------------------------------------------------------------------------------
# ---- 1.3.1 真实股票数据源 ----
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

# baostock：国内稳定的免费行情源（免注册），作为 akshare(东方财富) 失败时的自动备用数据源
try:
    import baostock as bs
    HAS_BAOSTOCK = True
except ImportError:
    HAS_BAOSTOCK = False

# ---- 1.3.2 三大主流 Boosting 库 ----
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ---- 1.3.3 深度学习框架 ----
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---- 1.3.4 TabNet（若无则用 DNN 近似替代，见模型层注释）----
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

# ---- 1.3.5 符号回归 / 遗传编程（用于 SR，也用于 GEP/MEP 的近似替代）----
try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except ImportError:
    HAS_GPLEARN = False

# ---- 1.3.6 贝叶斯超参数优化 ----
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ---- 1.3.6b 经典统计时序模型 ARIMA ----
try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ---- 1.3.7 GUI 框架 ----
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QCheckBox, QRadioButton, QButtonGroup, QPushButton, QLabel,
        QLineEdit, QDateEdit, QComboBox, QTableWidget, QTableWidgetItem,
        QTabWidget, QProgressBar, QMessageBox, QScrollArea, QSplitter, QTextEdit, QTextBrowser,
        QFileDialog, QDialog, QProgressDialog
    )
    from PySide6.QtCore import Qt, QThread, Signal, QDate, QTimer
    from PySide6.QtGui import QFont
    matplotlib.use("QtAgg")     # 让 matplotlib 使用 Qt 后端，方便嵌入 PySide6 界面
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# ==================== 第二部分：全局常量与配置 ====================
# ------------------------------------------------------------------------------
# 2.1 目录与缓存配置
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")      # 行情数据本地缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)
MONITOR_DIR = os.path.join(BASE_DIR, "monitor_log")   # 实时监控的盘口快照记录目录（自采集真实数据）
os.makedirs(MONITOR_DIR, exist_ok=True)
PRED_DIR = os.path.join(BASE_DIR, "predictions")      # 预测跟踪：保存每次预测，等日期到了对比真实股价
os.makedirs(PRED_DIR, exist_ok=True)
PRED_LOG = os.path.join(PRED_DIR, "pred_log.csv")

RANDOM_SEED = 42            # 全局随机种子，保证实验可复现
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------------
# 2.2 训练配置数据类
# ------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """
    训练总配置，对应界面上"B-1.2 功能选择"与"B-1.4 训练测试比例选择"等控件的取值。
    """
    window_size: int = 20            # 用过去多少天的数据作为一个输入样本（滑动窗口长度）
    horizon: int = 1                 # 预测未来第几天的收盘价（1 = 预测下一天）
    target_mode: str = "return"      # "price"=预测收盘价水平 / "return"=预测涨跌幅（默认，更诚实）
    #  ↑ 默认改为预测"涨跌幅"：股价水平自相关极强，预测价格会得到虚高的 R²；预测涨跌幅更平稳、
    #    也更贴近"判断涨跌"的真实目标。无论哪种模式，评估与画图都会统一还原成"价格"来比较。
    train_ratio: float = 0.7         # 训练集比例，默认 7:3（对应界面比例控件）
    val_ratio: float = 0.0           # 验证集比例(占全体)。0=自动取"剩余部分的一半"作独立验证集
    #  ↑ 三分：训练集在前(拟合模型) / 验证集居中(独立留出，只报误差，不参与训练/测试/寻优) / 测试集在后(最终留出)
    use_cv: bool = False             # 是否启用 5 折交叉验证
    cv_folds: int = 5
    hpo_method: str = "BO"           # 超参数优化方式："PSO"/"GA"/"FA"/"SOA"/"BO"/"关闭"
    hpo_trials: int = 20             # 寻优迭代/试验次数
    metrics: List[str] = field(default_factory=lambda: ["R2", "MAE", "RMSE", "DA", "UP_P"])
    add_naive_baseline: bool = True  # 是否自动加入"前值持有"朴素基准（强烈建议开启，见下方说明）
    #  ↑↑↑【股票预测最重要的诚实性开关】↑↑↑
    #  股价是强自相关序列，"用今天收盘价当作明天预测值"这种什么都不学的朴素基准，
    #  在"预测价格水平"任务上就能轻松拿到 R²>0.95。因此单看某个模型 R²=0.98 毫无意义，
    #  必须和朴素基准对比：只有当模型 MAE/RMSE 明显低于朴素基准、且方向准确率(DA)显著高于
    #  50%，才说明它真的学到了东西。开启本项后，结果表里会自动出现一行 "Naive(前值)" 作参照。


# ==================== 第三部分：数据获取层 ====================
@contextlib.contextmanager
def _no_proxy():
    """
    临时"忽略系统代理"的上下文管理器。

    背景：东方财富/新浪等 A 股行情接口都是**国内**站点，直连即可访问。但如果电脑上开着
    VPN / 科学上网 / 代理软件，它们往往会设置 HTTP_PROXY/HTTPS_PROXY 环境变量，导致
    akshare(底层 requests) 把对国内站点的请求也硬走代理，从而报
    "ProxyError: Unable to connect to proxy"。
    进入本上下文时清空这些代理变量并把 NO_PROXY 设为通配 "*"，退出时原样恢复，
    这样拉行情时自动直连、绕过代理，用户开着 VPN 也不受影响。
    """
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy"]
    saved = {k: os.environ.pop(k, None) for k in proxy_keys}
    saved_no = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        for k, v in saved_no.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StockDataFetcher:
    """
    A 股历史行情数据获取器。
    数据源：akshare（免费、无需注册、基于东方财富/新浪等公开接口）。

    使用方式：
        fetcher = StockDataFetcher()
        df = fetcher.fetch("600519", "20200101", "20241231")   # 贵州茅台
    """

    # ---------- 3.1 拉取真实历史行情（多源自动容错）----------
    def fetch(self, code: str, start_date: str, end_date: str,
              adjust: str = "qfq", use_cache: bool = True,
              bypass_proxy: bool = True, source: str = "auto",
              retries: int = 2) -> pd.DataFrame:
        """
        拉取指定股票代码在 [start_date, end_date] 区间的日线行情。

        参数：
            code       : 股票代码，如 "600519"（贵州茅台，不带市场前缀）
            start_date : 起始日期，格式 "YYYYMMDD"
            end_date   : 结束日期，格式 "YYYYMMDD"
            adjust     : 复权方式，"qfq"=前复权（推荐，训练用），"hfq"=后复权，""=不复权
            use_cache  : 是否优先读取本地缓存（避免重复请求，加快调试速度）
            bypass_proxy: 拉取时是否临时绕过系统代理（默认开，解决 VPN 导致的 ProxyError）
            source     : "auto"(先东方财富后 baostock) / "akshare" / "baostock"
            retries    : 单个数据源的重试次数（应对 RemoteDisconnected 等瞬时网络抖动）

        返回：
            统一格式的 DataFrame，至少包含列 [date, open, close, high, low, volume]。
        """
        cache_path = self._cache_path(code, start_date, end_date, adjust)

        # ---- 3.1.1 优先读取本地缓存 ----
        # 但：若结束日期是"今天或未来"，当天日线还在变(报告价格会滞后)，则强制重新拉取最新数据，不吃旧缓存。
        today = dt.date.today().strftime("%Y%m%d")
        if use_cache and os.path.exists(cache_path) and end_date < today:
            return pd.read_parquet(cache_path)

        # ---- 3.1.2 按顺序尝试各数据源，任一成功即返回；全失败则汇总错误 ----
        if source == "akshare":
            providers = [("akshare(东方财富)", self._fetch_akshare)]
        elif source == "baostock":
            providers = [("baostock", self._fetch_baostock)]
        else:                       # auto：东方财富优先，失败自动切 baostock
            providers = [("akshare(东方财富)", self._fetch_akshare),
                         ("baostock", self._fetch_baostock)]

        errors = []
        for name, fn in providers:
            available = (name.startswith("akshare") and HAS_AKSHARE) or \
                        (name == "baostock" and HAS_BAOSTOCK)
            if not available:
                errors.append(f"{name}: 未安装对应库")
                continue
            for attempt in range(1, retries + 1):
                try:
                    proxy_ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
                    with proxy_ctx:
                        df = fn(code, start_date, end_date, adjust)
                    if df is None or len(df) == 0:
                        raise RuntimeError("返回空数据（代码/日期可能有误）")
                    df.to_parquet(cache_path, index=False)     # 成功即缓存
                    return df
                except Exception as e:
                    errors.append(f"{name}(第{attempt}次): {e}")
                    if attempt < retries:
                        time.sleep(1.0 * attempt)     # 递增退避，缓解服务器瞬时限流/掐连

        # ---- 3.1.3 全部数据源失败：给出清晰、可操作的报错 ----
        hint = ""
        if not HAS_BAOSTOCK:
            hint = ("\n建议加装更稳定的国内备用源:  pip install baostock"
                    "\n（免费免注册，东方财富拉不到时会自动切换到它）")
        raise RuntimeError(
            "获取真实行情失败，已尝试以下数据源：\n  - " + "\n  - ".join(errors) +
            hint +
            "\n若网络必须走代理才能出网，可在代码里以 bypass_proxy=False 调用；"
            "\n或先用「合成数据」跑通流程。"
        )

    # ---------- 3.1a 数据源实现：akshare（东方财富）----------
    @staticmethod
    def _fetch_akshare(code, start_date, end_date, adjust):
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start_date, end_date=end_date, adjust=adjust)
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change",
            "涨跌额": "change", "换手率": "turnover"
        })
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ---------- 3.1b 数据源实现：baostock（国内稳定备用源）----------
    @staticmethod
    def _fetch_baostock(code, start_date, end_date, adjust):
        # 代码转 baostock 格式：sh.600519 / sz.000001 / bj.830799
        c = code.strip()
        if c.startswith("6"):
            bs_code = f"sh.{c}"
        elif c.startswith(("4", "8")):
            bs_code = f"bj.{c}"
        else:
            bs_code = f"sz.{c}"
        # 日期转 YYYY-MM-DD；复权标志：qfq=2 前复权 / hfq=1 后复权 / ""=3 不复权
        def _dash(d): return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
        adjustflag = {"qfq": "2", "hfq": "1", "": "3"}.get(adjust, "2")

        lg = bs.login()
        try:
            if getattr(lg, "error_code", "0") != "0":
                raise RuntimeError(f"baostock 登录失败: {getattr(lg, 'error_msg', '')}")
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume,amount",
                start_date=_dash(start_date), end_date=_dash(end_date),
                frequency="d", adjustflag=adjustflag)
            if rs.error_code != "0":
                raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            bs.logout()

        if len(df) == 0:
            return df
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    # ---------- 3.1c 外部数据接入：财报估值 / 大盘环境 / 主力资金 / 新闻（全部真实来源，缺失自动跳过不臆造）----------
    @staticmethod
    def enrich(df: pd.DataFrame, code: str, want_valuation: bool = True,
               want_index: bool = True, want_fundflow: bool = True,
               want_us: bool = True, want_northbound: bool = True,
               bypass_proxy: bool = True) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        把日线行情 df 用"按日期对齐(merge_asof 向后取最近已知值，无未来泄露)"的方式，
        并入真实的**财报估值**(日频 PE/PB/PS/股息率/总市值) 与**大盘环境**(沪深300 涨跌/均线偏离)。
        任一来源取数失败时**自动跳过该来源**(对应特征就是没有)，绝不填充假数据。
        返回 (增强后的 df, 各来源状态字典)。
        """
        # merge_asof 要求两侧 date 列 dtype 完全一致；统一成 datetime64[ns]，避免 us/s 分辨率不一致报错
        def _nsdate(d):
            d = d.copy()
            d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
            return d.sort_values("date")

        df = _nsdate(df)
        status: Dict[str, str] = {}
        if want_valuation and HAS_AKSHARE:
            try:
                v = _nsdate(StockDataFetcher._fetch_valuation(code, bypass_proxy))
                df = pd.merge_asof(df, v, on="date", direction="backward")
                vcols = [c for c in v.columns if c != "date"]
                status["valuation"] = f"已接入({len(v)}行: {'/'.join(vcols)})"
            except Exception as e:
                status["valuation"] = f"跳过(取数失败: {e})"
        elif want_valuation:
            status["valuation"] = "跳过(未安装 akshare)"
        if want_index and HAS_AKSHARE:
            try:
                ix = _nsdate(StockDataFetcher._fetch_index(bypass_proxy))
                df = pd.merge_asof(df, ix, on="date", direction="backward")
                status["index"] = "已接入(沪深300)"
            except Exception as e:
                status["index"] = f"跳过(取数失败: {e})"
        elif want_index:
            status["index"] = "跳过(未安装 akshare)"
        if want_fundflow and HAS_AKSHARE:
            try:
                mf = _nsdate(StockDataFetcher._fetch_fundflow(code, bypass_proxy))
                df = pd.merge_asof(df, mf, on="date", direction="backward")
                status["fundflow"] = f"已接入(东方财富主力资金, {len(mf)}行)"
            except Exception as e:
                status["fundflow"] = f"跳过(取数失败: {e})"
        elif want_fundflow:
            status["fundflow"] = "跳过(未安装 akshare)"
        if want_us and HAS_AKSHARE:
            try:
                us = _nsdate(StockDataFetcher._fetch_us_index(bypass_proxy))
                df = pd.merge_asof(df, us, on="date", direction="backward")
                status["us"] = "已接入(隔夜纳斯达克, 日期+1对齐无泄露)"
            except Exception as e:
                status["us"] = f"跳过(取数失败: {e})"
        elif want_us:
            status["us"] = "跳过(未安装 akshare)"
        if want_northbound and HAS_AKSHARE:
            try:
                nb = _nsdate(StockDataFetcher._fetch_northbound(bypass_proxy))
                df = pd.merge_asof(df, nb, on="date", direction="backward")
                status["northbound"] = f"已接入(北向资金, {len(nb)}行)"
            except Exception as e:
                status["northbound"] = f"跳过(取数失败: {e})"
        elif want_northbound:
            status["northbound"] = "跳过(未安装 akshare)"
        return df.reset_index(drop=True), status

    @staticmethod
    def _retry(fn, tries: int = 3, delay: float = 1.0):
        """对外部网络接口做几次重试，缓解 RemoteDisconnected 等瞬时抖动；最后一次失败则抛出。"""
        last = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                if i < tries - 1:
                    time.sleep(delay * (i + 1))
        raise last

    @staticmethod
    def _fetch_fundflow(code: str, bypass_proxy: bool = True) -> pd.DataFrame:
        """
        真实主力资金流向(日频)，数据源=东方财富(公开资金流 API)。
        这是"主力/大单是买是卖"的**历史可回测**版本，也是"买卖手力量"的历史等价物
        （实时盘口委买委卖是快照、无免费历史，不能用于回测）。
        取：主力净流入净额/净占比、超大单/大单/中单/小单净额。

        策略：先用"带浏览器请求头的直连"调东财 API(往往能绕过 akshare 遇到的反爬掐连 RemoteDisconnected)，
        直连失败再回退到 akshare。两条路都是同一份真实公开数据，只是取法不同。
        """
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        errors = []
        # ---- 路径 A：带浏览器 UA 的直连东财资金流 API ----
        try:
            with ctx:
                return StockDataFetcher._fetch_fundflow_direct(code)
        except Exception as e:
            errors.append(f"直连: {e}")
        # ---- 路径 B：回退到 akshare(更耐心的重试) ----
        market = "sh" if code.startswith("6") else ("bj" if code.startswith(("4", "8")) else "sz")
        try:
            ctx2 = _no_proxy() if bypass_proxy else contextlib.nullcontext()
            with ctx2:
                f = StockDataFetcher._retry(
                    lambda: ak.stock_individual_fund_flow(stock=code, market=market),
                    tries=4, delay=1.5)
            ren = {"日期": "date",
                   "主力净流入-净额": "mf_main_net", "主力净流入-净占比": "mf_main_pct",
                   "超大单净流入-净额": "mf_xl_net", "大单净流入-净额": "mf_l_net",
                   "中单净流入-净额": "mf_m_net", "小单净流入-净额": "mf_s_net"}
            f = f.rename(columns=ren)
            keep = ["date"] + [c for c in ren.values() if c != "date" and c in f.columns]
            f = f[keep]
            f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
            for c in keep:
                if c != "date":
                    f[c] = pd.to_numeric(f[c], errors="coerce")
            return f.dropna(subset=["date"]).sort_values("date")
        except Exception as e:
            errors.append(f"akshare: {e}")
        raise RuntimeError("；".join(errors))

    @staticmethod
    def _fetch_fundflow_direct(code: str) -> pd.DataFrame:
        """带浏览器请求头直连东财公开资金流 API(push2his.eastmoney.com)。真实数据，非爬虫破解。"""
        import requests
        secid = f"1.{code}" if code.startswith("6") else f"0.{code}"   # 1=沪 0=深/北
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0", "klt": "101", "secid": secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
        }
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Referer": "https://data.eastmoney.com/",
            "Accept": "application/json, text/plain, */*",
        }
        r = StockDataFetcher._retry(
            lambda: requests.get(url, params=params, headers=headers, timeout=12), tries=4, delay=1.2)
        klines = (r.json().get("data") or {}).get("klines") or []
        if not klines:
            raise RuntimeError("直连返回空(可能代码/市场不对)")
        rows = [k.split(",") for k in klines]
        df = pd.DataFrame([row[:7] for row in rows],
                          columns=["date", "mf_main_net", "mf_s_net", "mf_m_net",
                                   "mf_l_net", "mf_xl_net", "mf_main_pct"])
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _fetch_valuation(code: str, bypass_proxy: bool = True) -> pd.DataFrame:
        """真实财报估值(日频)。优先旧接口 stock_a_indicator_lg(乐咕乐股)；新版 akshare 已移除它，
        自动改用 stock_zh_valuation_baidu(百度)——逐指标(总市值/市盈率TTM/市净率/市盈率静)取全历史再按日期合并。
        返回列含 date + val_*(具体哪些取决于哪个接口/指标成功，缺失就没有，绝不造假)。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        fn_lg = getattr(ak, "stock_a_indicator_lg", None) or getattr(ak, "stock_a_lg_indicator", None)
        # ---- 路径 A：旧乐咕乐股接口(若此 akshare 版本仍有) ----
        if fn_lg is not None:
            with ctx:
                v = StockDataFetcher._retry(lambda: fn_lg(symbol=code))
            v = v.rename(columns={"trade_date": "date"})
            v["date"] = pd.to_datetime(v["date"]).astype("datetime64[ns]")
            keep = {"date": "date", "pe_ttm": "val_pe_ttm", "pb": "val_pb", "ps_ttm": "val_ps_ttm",
                    "dv_ttm": "val_dv_ttm", "total_mv": "val_total_mv"}
            cols = [c for c in keep if c in v.columns]
            v = v[cols].rename(columns=keep)
            for c in v.columns:
                if c != "date":
                    v[c] = pd.to_numeric(v[c], errors="coerce")
            return v.dropna(subset=["date"]).sort_values("date")
        # ---- 路径 B：百度个股估值 stock_zh_valuation_baidu(新版 akshare) ----
        if getattr(ak, "stock_zh_valuation_baidu", None) is None:
            raise RuntimeError("当前 akshare 无个股估值接口(stock_a_indicator_lg / stock_zh_valuation_baidu)")
        want = {"总市值": "val_total_mv", "市盈率(TTM)": "val_pe_ttm",
                "市净率": "val_pb", "市盈率(静)": "val_pe_static"}
        merged = None
        # 注意：ctx(_no_proxy) 是"一次性"上下文管理器，必须把整个循环包在一个 with 里，
        # 不能在循环内反复 with ctx（第二次进入会报错，导致后续指标被误跳过）。
        with ctx:
            for ind, col in want.items():
                try:
                    d = StockDataFetcher._retry(
                        lambda ind=ind: ak.stock_zh_valuation_baidu(symbol=code, indicator=ind, period="全部"))
                except Exception:
                    continue                     # 单个指标失败就跳过，不影响其它指标
                if d is None or len(d) == 0:
                    continue
                date_col = "date" if "date" in d.columns else d.columns[0]
                val_col = "value" if "value" in d.columns else d.columns[-1]
                d = d[[date_col, val_col]].rename(columns={date_col: "date", val_col: col})
                d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
                d[col] = pd.to_numeric(d[col], errors="coerce")
                merged = d if merged is None else pd.merge(merged, d, on="date", how="outer")
        if merged is None or len(merged) == 0:
            raise RuntimeError("百度估值接口未取到任何指标数据")
        return merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _fetch_index(bypass_proxy: bool = True, symbol: str = "sh000300") -> pd.DataFrame:
        """真实大盘环境：akshare stock_zh_index_daily(沪深300)。产出当日涨跌与相对20日均线偏离。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            ix = StockDataFetcher._retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
        ix["date"] = pd.to_datetime(ix["date"]).astype("datetime64[ns]")
        ix = ix.sort_values("date")
        ix["idx_ret_1d"] = ix["close"].pct_change()
        ix["idx_madev20"] = ix["close"] / (ix["close"].rolling(20).mean() + 1e-9) - 1
        return ix[["date", "idx_ret_1d", "idx_madev20"]].dropna()

    @staticmethod
    def fetch_quality(code: str, bypass_proxy: bool = True) -> Dict[str, Optional[float]]:
        """优雅获取基本面『质量/成长』因子：ROE(净资产收益率) 与 营收同比增长率。
        取数失败(无网络/接口变动/该股无数据)时返回空 → 因子缺失、绝不编造。"""
        out: Dict[str, Any] = {"roe": None, "rev_growth": None,
                               "f_score": None, "f_available": 0, "f_detail": []}
        if not HAS_AKSHARE:
            return out
        try:
            ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
            with ctx:
                fi = StockDataFetcher._retry(
                    lambda: ak.stock_financial_analysis_indicator(symbol=code), tries=2)
            if fi is None or len(fi) == 0:
                return out
            fi = fi.sort_index()
            row = fi.iloc[-1]                                # 最近一期财报
            for key in fi.columns:
                k = str(key)
                if out["roe"] is None and ("净资产收益率" in k) and ("加权" in k or "摊薄" in k or k.endswith("(%)")):
                    out["roe"] = pd.to_numeric(row[key], errors="coerce")
                if out["rev_growth"] is None and ("主营业务收入增长率" in k or "营业收入增长率" in k):
                    out["rev_growth"] = pd.to_numeric(row[key], errors="coerce")
            # 兜底：若上面没匹配到 ROE，取第一列含"净资产收益率"的
            if out["roe"] is None:
                for key in fi.columns:
                    if "净资产收益率" in str(key):
                        out["roe"] = pd.to_numeric(row[key], errors="coerce"); break
            for k in ("roe", "rev_growth"):
                if out[k] is not None and (pd.isna(out[k]) or np.isinf(out[k])):
                    out[k] = None
            # 复用同一份财报指标算 Piotroski F-Score(客观质量分，借鉴 trading_skills)
            pf = StockDataFetcher._piotroski_from_fi(fi)
            out.update(pf)
        except Exception:
            pass
        return out

    @staticmethod
    def _piotroski_from_fi(fi: pd.DataFrame) -> Dict[str, Any]:
        """从新浪财务指标表算 Piotroski 式 F-Score(0~9，越高质量越好；Joseph Piotroski 经典 9 项)。
        用最近两期年报做同比。此表拿不到"是否增发"，故该项跳过，只评可评的、如实报告 available 数——绝不编。
        返回 {f_score, f_available, f_detail(list)}。全部失败则返回空分。"""
        try:
            cols = list(fi.columns)
            cur, prev = fi.iloc[-1], (fi.iloc[-2] if len(fi) >= 2 else None)

            def num(row, *kws, avoid=None):
                if row is None:
                    return None
                for c in cols:
                    cs = str(c)
                    if all(k in cs for k in kws) and (avoid is None or avoid not in cs):
                        v = pd.to_numeric(row.get(c), errors="coerce")
                        if v is not None and not pd.isna(v) and not np.isinf(v):
                            return float(v)
                return None

            eps = num(cur, "每股收益")
            roa = num(cur, "总资产利润率") or num(cur, "总资产净利润率")
            cfo = num(cur, "每股经营现金流")
            debt_c, debt_p = num(cur, "资产负债率"), num(prev, "资产负债率")
            cr_c, cr_p = num(cur, "流动比率"), num(prev, "流动比率")
            gm_c, gm_p = num(cur, "销售毛利率"), num(prev, "销售毛利率")
            at_c, at_p = num(cur, "总资产周转率"), num(prev, "总资产周转率")

            crit = [
                ("盈利:净利润为正", None if eps is None else eps > 0),
                ("盈利:总资产收益率ROA>0", None if roa is None else roa > 0),
                ("盈利:经营现金流为正", None if cfo is None else cfo > 0),
                ("质量:现金流>净利(应计低)", None if (cfo is None or eps is None) else cfo > eps),
                ("杠杆:资产负债率下降", None if (debt_c is None or debt_p is None) else debt_c < debt_p),
                ("流动:流动比率上升", None if (cr_c is None or cr_p is None) else cr_c > cr_p),
                ("效率:毛利率上升", None if (gm_c is None or gm_p is None) else gm_c > gm_p),
                ("效率:资产周转率上升", None if (at_c is None or at_p is None) else at_c > at_p),
            ]
            avail = [(n, bool(v)) for n, v in crit if v is not None]
            f_score = sum(1 for _, v in avail if v)
            return {"f_score": f_score, "f_available": len(avail),
                    "f_detail": avail}
        except Exception:
            return {"f_score": None, "f_available": 0, "f_detail": []}

    @staticmethod
    def fetch_index_close(bypass_proxy: bool = True, symbol: str = "sh000300") -> pd.DataFrame:
        """拉大盘指数(默认沪深300)的日期+收盘价，供回测做『相对大盘超额』基准。失败抛异常由调用方跳过。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            ix = StockDataFetcher._retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
        ix["date"] = pd.to_datetime(ix["date"]).astype("datetime64[ns]")
        return ix.sort_values("date")[["date", "close"]].rename(columns={"close": "idx_px"})

    @staticmethod
    def _fetch_us_index(bypass_proxy: bool = True) -> pd.DataFrame:
        """隔夜美股(纳斯达克)涨跌：akshare index_us_stock_sina(.IXIC)。
        A股当日只能用"昨夜美股"(美股某日收盘在中国次日凌晨已知)，故把美股日期+1天再对齐，杜绝未来泄露。"""
        fn = getattr(ak, "index_us_stock_sina", None)
        if fn is None:
            raise RuntimeError("当前 akshare 无美股指数接口 index_us_stock_sina")
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            us = StockDataFetcher._retry(lambda: fn(symbol=".IXIC"))
        us = us.rename(columns={c: c.lower() for c in us.columns})
        dcol = "date" if "date" in us.columns else us.columns[0]
        us["date"] = pd.to_datetime(us[dcol]).astype("datetime64[ns]")
        us = us.sort_values("date")
        us["us_ret"] = pd.to_numeric(us["close"], errors="coerce").pct_change()
        # 关键：美股 D 日收盘 → A股 D+1 日开盘前才可用，故日期+1天再对齐(无泄露)
        us["date"] = us["date"] + pd.Timedelta(days=1)
        return us[["date", "us_ret"]].dropna()

    @staticmethod
    def _fetch_northbound(bypass_proxy: bool = True) -> pd.DataFrame:
        """北向资金(沪股通+深股通)每日净流入：兼容多个 akshare 函数名。产出 nb_net(亿元)。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        fn = (getattr(ak, "stock_hsgt_north_net_flow_in_em", None)
              or getattr(ak, "stock_hsgt_hist_em", None))
        if fn is None:
            raise RuntimeError("当前 akshare 无北向资金接口(stock_hsgt_*)")
        with ctx:
            try:
                nb = StockDataFetcher._retry(lambda: fn(symbol="北向"))
            except TypeError:
                nb = StockDataFetcher._retry(lambda: fn())
        cols = {c: str(c) for c in nb.columns}
        nb = nb.rename(columns=cols)
        dcol = next((c for c in nb.columns if "日期" in c or c.lower() == "date"), nb.columns[0])
        vcol = next((c for c in nb.columns if ("净" in c and ("流入" in c or "额" in c)) or "北向" in c),
                    nb.columns[-1])
        nb = nb[[dcol, vcol]].rename(columns={dcol: "date", vcol: "nb_net"})
        nb["date"] = pd.to_datetime(nb["date"], errors="coerce").astype("datetime64[ns]")
        nb["nb_net"] = pd.to_numeric(nb["nb_net"], errors="coerce")
        return nb.dropna(subset=["date"]).sort_values("date")

    @staticmethod
    def fetch_news(code: str, limit: int = 15, bypass_proxy: bool = True) -> pd.DataFrame:
        """真实个股新闻(仅供展示/近期参考)：akshare stock_news_em，数据源=东方财富。
        注意：新闻接口只覆盖近期，且把文本转成可靠的历史情绪分数需要 NLP 模型；为避免臆造，
        本软件**不**把新闻情绪当作历史训练特征，只在界面展示真实标题与链接。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            nd = StockDataFetcher._retry(lambda: ak.stock_news_em(symbol=code))
        nd = nd.rename(columns={"新闻标题": "title", "发布时间": "time",
                                "文章来源": "source", "新闻链接": "url"})
        cols = [c for c in ["time", "title", "source", "url"] if c in nd.columns]
        return nd[cols].head(limit)

    @staticmethod
    def fetch_stock_name(code: str, bypass_proxy: bool = True) -> str:
        """获取股票简称(用于识别 ST/*ST 退市风险)。真实来源=东方财富 stock_individual_info_em。失败返回空串。"""
        try:
            ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
            with ctx:
                info = StockDataFetcher._retry(lambda: ak.stock_individual_info_em(symbol=code), tries=2)
            # info 为两列(item/value)的表；找"股票简称"
            m = info.set_index(info.columns[0])[info.columns[1]].to_dict()
            for k, v in m.items():
                if "简称" in str(k):
                    return str(v)
        except Exception:
            pass
        return ""

    @staticmethod
    def fetch_realtime(code: str, bypass_proxy: bool = True) -> Dict[str, Any]:
        """实时盘口快照(用于盘中监控)：akshare stock_bid_ask_em，数据源=东方财富。
        返回字典(现价/涨跌幅/今开最高最低/量比/换手/成交量额 + 买卖5档)。仅监控展示，不作历史训练。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            d = StockDataFetcher._retry(lambda: ak.stock_bid_ask_em(symbol=code), tries=2, delay=0.8)
        # d 为两列(item/value)
        m = {}
        try:
            m = d.set_index(d.columns[0])[d.columns[1]].to_dict()
        except Exception:
            pass
        def g(*keys):
            for k in keys:
                if k in m and m[k] not in ("", "-", None):
                    return m[k]
            return None
        # 买卖5档
        bids = [(g(f"buy_{i}"), g(f"buy_{i}_vol")) for i in range(1, 6)]
        asks = [(g(f"sell_{i}"), g(f"sell_{i}_vol")) for i in range(1, 6)]
        return {
            "price": g("最新"), "pct": g("涨幅"), "chg": g("涨跌"),
            "open": g("今开"), "high": g("最高"), "low": g("最低"), "prev": g("昨收"),
            "avg": g("均价"), "vol": g("总手"), "amount": g("金额"),
            "turnover": g("换手"), "vol_ratio": g("量比"),
            "limit_up": g("涨停"), "limit_down": g("跌停"),
            "bids": bids, "asks": asks, "raw": m,
        }

    # ---------- 3.2 缓存文件路径 ----------
    def _cache_path(self, code: str, start_date: str, end_date: str, adjust: str) -> str:
        fname = f"{code}_{start_date}_{end_date}_{adjust}.parquet"
        return os.path.join(CACHE_DIR, fname)

    # ---------- 3.3 离线合成数据（无网络/未安装 akshare 时，用于快速跑通全流程测试）----------
    @staticmethod
    def generate_synthetic_data(n_days: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
        """
        生成一份带有趋势 + 周期 + 噪声的模拟股价数据，字段格式与 fetch() 的返回完全一致，
        方便在没有网络 / 没装 akshare 的情况下，先把特征工程、模型训练、GUI 流程跑通。
        """
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2019-01-01", periods=n_days, freq="B")   # 工作日频率
        trend = np.linspace(20, 60, n_days)
        season = 3 * np.sin(np.linspace(0, 40 * np.pi, n_days))
        noise = rng.normal(0, 1.2, n_days)
        close = trend + season + noise
        close = np.clip(close, 1, None)

        daily_range = np.abs(rng.normal(0.8, 0.3, n_days))
        high = close + daily_range
        low = close - daily_range
        open_ = close + rng.normal(0, 0.3, n_days)
        volume = rng.integers(1_000_00, 5_000_00, n_days).astype(float)

        df = pd.DataFrame({
            "date": dates, "open": open_, "close": close,
            "high": high, "low": low, "volume": volume,
        })
        return df


# ==================== 第四部分：特征工程层 ====================
class FeatureEngineer:
    """
    负责把原始行情 DataFrame 转换成机器学习模型可以直接使用的 (X, y) 样本。
    整体流程： 原始行情 -> 计算技术指标 -> 标准化 -> 滑动窗口切片 -> 训练/测试切分
    """

    def __init__(self, window_size: int = 20, horizon: int = 1, target_mode: str = "price"):
        self.window_size = window_size      # 滑动窗口长度（用过去多少天预测未来）
        self.horizon = horizon              # 预测未来第几天
        self.target_mode = target_mode      # "price"=预测收盘价水平 / "return"=预测涨跌幅(更诚实)
        self.scaler_x: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        # build_supervised_samples 会顺便填充下面两个数组（与 X 对齐），供还原价格/朴素基准/方向评估使用
        self.prev_close_: Optional[np.ndarray] = None    # 每个样本的"基准日"(窗口最后一天)收盘价
        self.close_target_: Optional[np.ndarray] = None  # 每个样本的"真实未来"收盘价(价格空间)

    # ---------- 4.1 技术指标计算 ----------
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在原始行情基础上追加常用技术指标列，全部为"当天可获得"的信息，不引入未来数据泄漏。
        """
        df = df.copy()

        # ---- 4.1.1 均线类：MA / EMA ----
        for w in (5, 10, 20, 60):
            df[f"ma{w}"] = df["close"].rolling(w).mean()
            df[f"ema{w}"] = df["close"].ewm(span=w, adjust=False).mean()

        # ---- 4.1.2 MACD ----
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_dif"] = ema12 - ema26
        df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

        # ---- 4.1.3 RSI（相对强弱指标）----
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi14"] = 100 - 100 / (1 + rs)

        # ---- 4.1.4 布林带（Bollinger Bands）----
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["boll_mid"] = mid
        df["boll_upper"] = mid + 2 * std
        df["boll_lower"] = mid - 2 * std

        # ---- 4.1.5 KDJ ----
        low9 = df["low"].rolling(9).min()
        high9 = df["high"].rolling(9).max()
        rsv = (df["close"] - low9) / (high9 - low9 + 1e-9) * 100
        df["kdj_k"] = rsv.ewm(com=2).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=2).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # ---- 4.1.6 成交量类指标 ----
        df["volume_ma5"] = df["volume"].rolling(5).mean()
        df["volume_change"] = df["volume"].pct_change()

        # ---- 4.1.7 价格动量 / 波动率 ----
        df["return_1d"] = df["close"].pct_change()
        df["volatility_10d"] = df["return_1d"].rolling(10).std()

        # ---- 4.1.8 市场情绪代理特征 ----
        # 说明：这些都是从"行情本身"就能算出、当天即可获得的**情绪代理**指标（不引入未来信息）。
        # 真正的"舆情情绪"(新闻/股吧/研报文本) 需要接外部数据源做 NLP，属于扩展项(见 README/AGENTS)，
        # 这里不臆造，只用可得的量价情绪。
        # 量比：当日成交量 / 过去5日均量（>1 放量、<1 缩量，反映资金热度）
        df["vol_ratio"] = df["volume"] / (df["volume"].rolling(5).mean() + 1e-9)
        # 换手率：若数据源自带则直接用（baostock/东方财富的 turnover）
        if "turnover" in df.columns:
            df["turnover_feat"] = pd.to_numeric(df["turnover"], errors="coerce")
        # 日内振幅：(最高-最低)/前收盘，反映多空博弈激烈程度
        df["amplitude_feat"] = (df["high"] - df["low"]) / (df["close"].shift(1) + 1e-9)
        # 连涨/连跌天数（带符号：+3=连涨3天，-2=连跌2天），刻画情绪惯性
        sign = np.sign(df["close"].diff().fillna(0.0))
        streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
        df["updown_streak"] = sign * streak
        # 收盘价在近20日高低区间中的相对位置（0=贴近区间低点，1=贴近高点/情绪亢奋）
        roll_hi = df["high"].rolling(20).max()
        roll_lo = df["low"].rolling(20).min()
        df["pos_in_range20"] = (df["close"] - roll_lo) / (roll_hi - roll_lo + 1e-9)

        # ---- 4.1.9 周/月/季 多周期趋势（由日线滚动计算，只回看历史，无未来泄露）----
        # 这是把"周K线/月K线"信息编码进特征的标准做法：各周期的涨跌幅、区间位置、相对均线偏离。
        for w, tag in [(5, "w1"), (20, "m1"), (60, "q1")]:      # 5日≈1周, 20日≈1月, 60日≈1季
            df[f"ret_{tag}"] = df["close"] / (df["close"].shift(w) + 1e-9) - 1        # 周期涨跌幅
            hi_w = df["high"].rolling(w).max(); lo_w = df["low"].rolling(w).min()
            df[f"pos_{tag}"] = (df["close"] - lo_w) / (hi_w - lo_w + 1e-9)            # 周期区间位置
            df[f"madev_{tag}"] = df["close"] / (df["close"].rolling(w).mean() + 1e-9) - 1  # 相对周期均线偏离

        # ---- 4.1.10 主力意图代理特征（仅当已接入真实主力资金流向 mf_* 时才计算，全部只回看历史，无泄露）----
        # 用真实的"每日主力净流入"派生"进场/退场/洗盘"的**可学习代理**，而不是臆造一个"主力预言"。
        if "mf_main_net" in df.columns:
            mf = pd.to_numeric(df["mf_main_net"], errors="coerce").fillna(0.0)
            sgn = np.sign(mf)
            # 连续净流入(+)/净流出(-)天数：刻画"主力持续进场/持续撤退"
            grp = (sgn != sgn.shift()).cumsum()
            df["mf_streak"] = sgn * (df.groupby(grp).cumcount() + 1)
            df["mf_cum5"] = mf.rolling(5).sum()            # 近一周主力累计净流入
            df["mf_cum20"] = mf.rolling(20).sum()          # 近一月主力累计净流入
            # 资金-价格背离：主力净流入方向 与 当日涨跌方向 不一致 → 吸筹/洗盘 或 派发 的代理信号
            price_sgn = np.sign(df["close"].pct_change().fillna(0.0))
            df["mf_price_div"] = sgn - price_sgn           # +2≈价跌但主力流入(疑吸筹/洗盘); -2≈价涨但主力流出(疑派发)
            if "mf_main_pct" in df.columns:
                df["mf_pct5"] = pd.to_numeric(df["mf_main_pct"], errors="coerce").rolling(5).mean()

        # ---- 4.1.10 外部数据列缺失处理 ----
        # 外部来源(估值/大盘/资金流)可能在"上市前/数据未覆盖期/亏损股无市盈率"等处为 NaN。
        # 这些列的 NaN 一律填 0(中性、无未来泄露)——否则后面按行 dropna 时，只要某外部列局部/整列为 NaN，
        # 就会把大段甚至全部历史行删光(空数据集报错 array=[])。填 0 表示"该处无此信息"，不影响时序纪律。
        for c in df.columns:
            if c != "date" and str(c).startswith(("val_", "idx_", "mf_", "us_", "nb_")):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        return df

    # ---------- 4.2 构造监督学习样本（滑动窗口）----------
    def build_supervised_samples(self, df: pd.DataFrame,
                                  feature_cols: Optional[List[str]] = None
                                  ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        把时间序列切成 (X, y) 监督学习样本：
            X[i] = 第 i 天往前数 window_size 天的全部特征（展平成一维向量）
            y[i] = 第 (i + horizon) 天的收盘价

        返回：
            X : shape = (样本数, window_size * 特征数)
            y : shape = (样本数,)
            sample_dates : 每个样本对应的"预测目标日期"，用于后续画图对齐横轴
        """
        df = self.add_technical_indicators(df)
        # 停牌日成交量可能为 0，使得"环比/量比"等特征出现 inf；dropna 删不掉 inf，需先转成 NaN 再删。
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1, how="all")            # 安全网：先丢弃"完全为空"的特征列，避免整表被行dropna删空
        df = df.dropna().reset_index(drop=True)      # 技术指标开头会有 NaN（如 MA60 需要60天），一并丢弃

        if feature_cols is None:
            # 默认使用除日期外的全部数值列作为特征
            feature_cols = [c for c in df.columns if c != "date"]
        self.feature_cols = feature_cols

        values = df[feature_cols].values
        close = df["close"].values
        openv = df["open"].values if "open" in df.columns else close
        highv = df["high"].values if "high" in df.columns else close
        lowv = df["low"].values if "low" in df.columns else close
        volv = df["volume"].values if "volume" in df.columns else np.ones_like(close)
        dates = df["date"]

        X, y, sample_dates, prev_close, close_target = [], [], [], [], []
        # 回测真实化所需的逐样本日线细节：基准日(买入日)与目标日(卖出日)的当日涨跌、是否一字板、成交量
        base_ret, base_locked, base_vol = [], [], []
        exit_ret, exit_locked, exit_vol = [], [], []
        n = len(df)
        for i in range(self.window_size, n - self.horizon + 1):
            window = values[i - self.window_size: i]          # 过去 window_size 天
            target = close[i + self.horizon - 1]              # 未来第 horizon 天的真实收盘价
            base = close[i - 1]                                # 基准日(窗口最后一天)的收盘价
            X.append(window.flatten())
            prev_close.append(base)
            close_target.append(target)
            # 目标：预测"价格水平" 或 "相对基准日的涨跌幅"（后者更平稳、更诚实）
            if self.target_mode == "return":
                y.append(target / (base + 1e-12) - 1.0)
            else:
                y.append(target)
            sample_dates.append(dates.iloc[i + self.horizon - 1])
            # --- 基准日(准备买入的那天，df 行 i-1) ---
            bi = i - 1
            base_ret.append(close[bi] / (close[bi - 1] + 1e-12) - 1.0 if bi >= 1 else 0.0)
            base_locked.append(1.0 if (highv[bi] - lowv[bi]) <= 1e-9 else 0.0)   # 最高==最低≈一字板
            base_vol.append(volv[bi])
            # --- 目标日(准备卖出的那天，df 行 i+horizon-1) ---
            ti = i + self.horizon - 1
            exit_ret.append(close[ti] / (close[ti - 1] + 1e-12) - 1.0 if ti >= 1 else 0.0)
            exit_locked.append(1.0 if (highv[ti] - lowv[ti]) <= 1e-9 else 0.0)
            exit_vol.append(volv[ti])

        self.prev_close_ = np.array(prev_close)
        self.close_target_ = np.array(close_target)
        # 逐样本日线细节(供回测判断涨跌停买不进/卖不出、停牌)
        self.bar_info_ = {
            "base_ret": np.array(base_ret), "base_locked": np.array(base_locked),
            "base_vol": np.array(base_vol), "exit_ret": np.array(exit_ret),
            "exit_locked": np.array(exit_locked), "exit_vol": np.array(exit_vol),
        }
        return np.array(X), np.array(y), pd.Series(sample_dates).reset_index(drop=True)

    # ---------- 4.3 标准化（在训练集上 fit，训练集/测试集上 transform，避免数据泄漏）----------
    def fit_scale(self, X_train: np.ndarray, y_train: np.ndarray):
        self.scaler_x = StandardScaler().fit(X_train)
        self.scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_scaled = self.scaler_x.transform(X)
        if y is None:
            return X_scaled
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
        return X_scaled, y_scaled

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    # ---------- 4.4 时间序列训练/测试切分 ----------
    @staticmethod
    def time_series_split(X: np.ndarray, y: np.ndarray, dates: pd.Series,
                           train_ratio: float = 0.7):
        """
        !!! 重要：股票数据是时间序列，绝对不能用 sklearn 默认的随机 shuffle 切分，
        否则会用"未来"的数据去预测"过去"，造成严重的数据泄漏、评估结果虚高。
        必须按时间顺序，前 train_ratio 部分做训练集，后面部分做测试集。
        """
        n = len(X)
        split = int(n * train_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        dates_train, dates_test = dates[:split], dates[split:]
        return X_train, X_test, y_train, y_test, dates_train, dates_test


# ==================== 第五部分：评价指标层 ====================
class Metrics:
    """
    对应界面上"B-1.3 评价指标核心配置"里的全部 10 个指标。
    统一入口：Metrics.calc_all(y_true, y_pred) -> dict
    """

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        denom = np.where(denom == 0, 1e-9, denom)
        return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

    @staticmethod
    def r_corr(y_true, y_pred):
        """皮尔逊相关系数 R（注意与决定系数 R² 不同，R 可以为负）"""
        if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
            return 0.0
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    @staticmethod
    def nse(y_true, y_pred):
        """
        纳什效率系数 Nash-Sutcliffe Efficiency。
        公式与 CE（效率系数）完全相同，水文/环境领域常用 NSE 这个名字，
        工程/机器学习领域常称为 CE，两者在数学上是同一个指标。
        """
        y_true = np.asarray(y_true)
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom < 1e-12:
            return 0.0
        return float(1 - np.sum((y_true - y_pred) ** 2) / denom)

    @staticmethod
    def ce(y_true, y_pred):
        """效率系数 CE，与 NSE 公式相同，此处直接复用"""
        return Metrics.nse(y_true, y_pred)

    @staticmethod
    def kge(y_true, y_pred):
        """
        Kling-Gupta 效率系数，综合考虑相关性(r)、变异性比(alpha)、均值比(beta) 三个分量。
        KGE = 1 - sqrt[ (r-1)^2 + (alpha-1)^2 + (beta-1)^2 ]
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        r = Metrics.r_corr(y_true, y_pred)
        std_t, std_p = np.std(y_true), np.std(y_pred)
        mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
        alpha = std_p / (std_t + 1e-9)
        beta = mean_p / (mean_t + 1e-9)
        return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    @staticmethod
    def wi(y_true, y_pred):
        """Willmott's 一致性指数 (Index of Agreement)"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mean_t = np.mean(y_true)
        numerator = np.sum((y_true - y_pred) ** 2)
        denom = np.sum((np.abs(y_pred - mean_t) + np.abs(y_true - mean_t)) ** 2)
        if denom < 1e-12:
            return 0.0
        return float(1 - numerator / denom)

    @staticmethod
    def si(y_true, y_pred):
        """散布指数 Scatter Index = RMSE / 观测均值"""
        mean_t = np.mean(y_true)
        if abs(mean_t) < 1e-9:
            return 0.0
        return float(Metrics.rmse(y_true, y_pred) / mean_t)

    @staticmethod
    def da(y_true, y_pred):
        """
        方向准确率 Directional Accuracy（单位：%）—— 股票预测里比 R² 更有实战意义的指标。

        做法：把测试集按时间排好序后，对每一天 t，以"前一天的真实收盘价"作为参照点，
        比较【模型预测的涨跌方向】与【真实的涨跌方向】是否一致，统计一致的比例。
            真实方向 = sign(真实收盘[t] - 真实收盘[t-1])
            预测方向 = sign(预测收盘[t] - 真实收盘[t-1])
        随机瞎猜的期望约为 50%。只有明显 > 50% 才说明模型对"明天涨还是跌"真的有判别力。

        注意：本实现假设 horizon=1（预测下一天），且传入的 y 已按时间升序排列，
        这在本软件的时间序列切分下天然成立。
        """
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if len(yt) < 2:
            return 0.0
        prev = yt[:-1]                                   # 以真实的前一天收盘价为参照
        true_dir = np.sign(yt[1:] - prev)
        pred_dir = np.sign(yp[1:] - prev)
        return float(np.mean(true_dir == pred_dir) * 100)

    @staticmethod
    def up_p(y_true, y_pred):
        """
        上涨精确率 Up-Precision（单位：%）：在模型**预测会涨**的那些天里，实际真的涨了的比例。
        这比总体方向准确率更贴近实盘——你只会在"模型说涨"时买入，所以最该关心"它说涨时到底准不准"。
        若模型从不预测上涨，则返回 0（无从谈起）。同样假设 horizon=1、y 按时间升序。
        """
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if len(yt) < 2:
            return 0.0
        prev = yt[:-1]
        pred_up = yp[1:] > prev
        true_up = yt[1:] > prev
        if pred_up.sum() == 0:
            return 0.0
        return float(np.mean(true_up[pred_up]) * 100)

    # ---------- 5.1 统一入口：按指标名称列表批量计算 ----------
    _FUNC_MAP = {
        "R2": r2.__func__, "MAE": mae.__func__, "MSE": mse.__func__, "RMSE": rmse.__func__,
        "MAPE": mape.__func__, "SMAPE": smape.__func__, "R": r_corr.__func__,
        "NSE": nse.__func__, "CE": ce.__func__, "KGE": kge.__func__,
        "WI": wi.__func__, "SI": si.__func__, "DA": da.__func__, "UP_P": up_p.__func__,
    }

    @classmethod
    def calc_all(cls, y_true, y_pred, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        names = metric_names or list(cls._FUNC_MAP.keys())
        return {name: round(cls._FUNC_MAP[name](y_true, y_pred), 6) for name in names if name in cls._FUNC_MAP}


# ==================== 第六部分：模型层（34 种算法，统一接口 + 可插拔注册表） ====================
# ------------------------------------------------------------------------------
# 6.0 统一接口：所有算法都必须实现 BaseModel 定义的 fit / predict 方法
# ------------------------------------------------------------------------------
class BaseModel(ABC):
    """
    所有 34 个算法的统一父类。不管底层是 sklearn / xgboost / torch 还是自定义 numpy 实现，
    外部训练调度层（TrainingPipeline）只通过这三个方法与模型交互，做到"可插拔"。
    """
    name: str = "BaseModel"
    category: str = ""          # "基础与传统模型" / "先进与前沿模型" / "公式拟合/演化计算"

    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_hpo_space(self) -> Dict[str, Tuple[float, float]]:
        """
        返回该模型可供超参数优化搜索的空间，格式 {参数名: (下界, 上界)}。
        默认空字典表示"不支持/不需要超参数寻优"，各子类按需覆盖。
        """
        return {}

    def set_hpo_params(self, param_dict: Dict[str, float]):
        """超参数优化时，用寻优得到的参数重新构建内部模型"""
        self.params.update(param_dict)
        self._build()

    def _build(self):
        """由子类实现：根据 self.params 构建/重建内部模型对象"""
        pass


# ------------------------------------------------------------------------------
# 6.1 基础与传统模型（对应截图 14 个：BP/ANN, SVR, LSSVM, GPR, ElasticNet, RidgeReg,
#                                  Lasso, PLSR, KNN, Kstar, ELM, DTR, DT, M5Rules）
# ------------------------------------------------------------------------------
class ANNModel(BaseModel):
    """BP/ANN —— 多层感知机，等价于经典的 BP 神经网络（误差反向传播训练）"""
    name, category = "BP/ANN", "基础与传统模型"

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, **kwargs)
        self._build()

    def _build(self):
        self.model = MLPRegressor(hidden_layer_sizes=self.params.get("hidden_layer_sizes", (64, 32)),
                                   alpha=self.params.get("alpha", 1e-4),
                                   max_iter=2000, random_state=RANDOM_SEED)

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-6, 1e-1)}


class SVRModel(BaseModel):
    """SVR —— 支持向量回归"""
    name, category = "SVR", "基础与传统模型"

    def __init__(self, C=10.0, epsilon=0.05, gamma="scale", **kwargs):
        super().__init__(C=C, epsilon=epsilon, gamma=gamma, **kwargs); self._build()

    def _build(self):
        self.model = SVR(C=self.params.get("C", 10.0), epsilon=self.params.get("epsilon", 0.05),
                          gamma=self.params.get("gamma", "scale"))

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"C": (0.1, 100.0), "epsilon": (0.001, 0.5)}


class LSSVMModel(BaseModel):
    """
    LSSVM（最小二乘支持向量机）—— 近似实现说明：
    数学上可以证明，LSSVM 在等式约束下的解等价于"核岭回归"(Kernel Ridge Regression)，
    两者只是求解方式不同（LSSVM 解线性方程组，标准 SVR 解二次规划），最终的回归函数
    形式完全一致。因此这里直接使用 sklearn 的 KernelRidge(kernel="rbf") 实现，
    是数学上严格等价的做法，而非粗糙近似。
    """
    name, category = "LSSVM", "基础与传统模型"

    def __init__(self, alpha=1.0, gamma=0.1, **kwargs):
        super().__init__(alpha=alpha, gamma=gamma, **kwargs); self._build()

    def _build(self):
        self.model = KernelRidge(alpha=self.params.get("alpha", 1.0),
                                  kernel="rbf", gamma=self.params.get("gamma", 0.1))

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-3, 10.0), "gamma": (1e-3, 5.0)}


class GPRModel(BaseModel):
    """GPR —— 高斯过程回归"""
    name, category = "GPR", "基础与传统模型"

    def __init__(self, alpha=1e-2, **kwargs):
        super().__init__(alpha=alpha, **kwargs); self._build()

    def _build(self):
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=self.params.get("alpha", 1e-2),
                                               normalize_y=True, random_state=RANDOM_SEED)

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-4, 1.0)}


class ElasticNetModel(BaseModel):
    name, category = "ElasticNet", "基础与传统模型"

    def __init__(self, alpha=0.01, l1_ratio=0.5, **kwargs):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **kwargs); self._build()

    def _build(self):
        self.model = ElasticNet(alpha=self.params.get("alpha", 0.01),
                                 l1_ratio=self.params.get("l1_ratio", 0.5),
                                 random_state=RANDOM_SEED, max_iter=5000)

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-4, 1.0), "l1_ratio": (0.0, 1.0)}


class RidgeRegModel(BaseModel):
    name, category = "RidgeReg", "基础与传统模型"

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(alpha=alpha, **kwargs); self._build()

    def _build(self): self.model = Ridge(alpha=self.params.get("alpha", 1.0), random_state=RANDOM_SEED)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-3, 100.0)}


class LassoModel(BaseModel):
    name, category = "Lasso", "基础与传统模型"

    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(alpha=alpha, **kwargs); self._build()

    def _build(self): self.model = Lasso(alpha=self.params.get("alpha", 0.01),
                                          random_state=RANDOM_SEED, max_iter=5000)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"alpha": (1e-4, 1.0)}


class PLSRModel(BaseModel):
    """PLSR —— 偏最小二乘回归"""
    name, category = "PLSR", "基础与传统模型"

    def __init__(self, n_components=5, **kwargs):
        super().__init__(n_components=n_components, **kwargs); self._build()

    def _build(self): self.model = PLSRegression(n_components=int(self.params.get("n_components", 5)))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X).ravel()
    def get_hpo_space(self): return {"n_components": (2, 20)}


class KNNModel(BaseModel):
    name, category = "KNN", "基础与传统模型"

    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs); self._build()

    def _build(self): self.model = KNeighborsRegressor(n_neighbors=int(self.params.get("n_neighbors", 5)))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_neighbors": (2, 30)}


class KStarModel(BaseModel):
    """
    K* —— 近似实现说明：
    真正的 K* 算法（Cleary & Trigg, 1995）使用基于"熵/信息压缩长度"的相似度而非
    欧式距离，是 WEKA 中的经典算法，Python 生态中没有成熟对等实现。
    此处用"距离倒数加权的 KNN"（相当于 K* 核心思想里"距离越近权重越大"的简化版）
    作为近似替代，效果接近但不完全等价。如需精确复现 K*，建议使用
    python-weka-wrapper3 调用 WEKA 原生实现。
    """
    name, category = "Kstar", "基础与传统模型"

    def __init__(self, n_neighbors=8, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs); self._build()

    def _build(self):
        self.model = KNeighborsRegressor(n_neighbors=int(self.params.get("n_neighbors", 8)),
                                          weights="distance")

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_neighbors": (3, 30)}


class ELMModel(BaseModel):
    """
    ELM —— 极限学习机（Extreme Learning Machine），自实现（numpy）：
    隐藏层权重/偏置随机生成并"冻结不训练"，只用最小二乘（Moore-Penrose 伪逆）
    求解输出层权重，这是 ELM 算法本身的标准定义，训练速度极快。
    """
    name, category = "ELM", "基础与传统模型"

    def __init__(self, n_hidden=200, **kwargs):
        super().__init__(n_hidden=n_hidden, **kwargs)
        self.W = None; self.b = None; self.beta = None

    def _build(self): pass  # ELM 的"结构"在 fit() 中按输入维度动态生成

    def fit(self, X, y):
        n_hidden = int(self.params.get("n_hidden", 200))
        rng = np.random.default_rng(RANDOM_SEED)
        n_features = X.shape[1]
        self.W = rng.normal(0, 1, size=(n_features, n_hidden))     # 随机输入权重（不训练）
        self.b = rng.normal(0, 1, size=(n_hidden,))                # 随机偏置（不训练）
        H = np.tanh(X @ self.W + self.b)                            # 隐藏层激活输出
        self.beta = np.linalg.pinv(H) @ y                           # 最小二乘求输出权重
        return self

    def predict(self, X):
        H = np.tanh(X @ self.W + self.b)
        return H @ self.beta

    def get_hpo_space(self): return {"n_hidden": (50, 500)}


class DTRModel(BaseModel):
    """DTR —— 决策树回归"""
    name, category = "DTR", "基础与传统模型"

    def __init__(self, max_depth=8, **kwargs):
        super().__init__(max_depth=max_depth, **kwargs); self._build()

    def _build(self): self.model = DecisionTreeRegressor(max_depth=int(self.params.get("max_depth", 8)),
                                                           random_state=RANDOM_SEED)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"max_depth": (2, 20)}


class DTModel(DTRModel):
    """
    DT —— 决策树。在回归任务场景下与 DTR 本质相同（sklearn 中分类树/回归树是两个类，
    但截图里 DT 与 DTR 并列，通常 DT 特指"稍浅、更强调可解释规则"的树），
    这里用较浅的最大深度与 DTR 区分开，避免两个算法结果完全一样。
    """
    name = "DT"

    def __init__(self, max_depth=4, **kwargs):
        super().__init__(max_depth=max_depth, **kwargs)


class M5RulesModel(BaseModel):
    """
    M5Rules —— 近似实现说明：
    真正的 M5Rules（Model Tree + Rules，Quinlan 1992 / Holmes et al. 1999）会先长出一棵
    模型树，每个叶子节点拟合一个局部线性回归，再从树中抽取规则集，是 WEKA 的经典算法。
    Python 生态没有成熟对等实现，这里采用等价思路的简化版"模型树"（Model Tree）近似：
        1) 用一棵较浅的决策树对特征空间做分区（相当于生成"规则"的条件部分）
        2) 每个叶子节点内部单独拟合一个线性回归（相当于规则的"结论部分"）
    如需精确复现 M5Rules，建议使用 python-weka-wrapper3。
    """
    name, category = "M5Rules", "基础与传统模型"

    def __init__(self, max_depth=4, **kwargs):
        super().__init__(max_depth=max_depth, **kwargs)
        self.tree = None
        self.leaf_models: Dict[int, Ridge] = {}

    def _build(self): pass

    def fit(self, X, y):
        max_depth = int(self.params.get("max_depth", 4))
        self.tree = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_SEED)
        self.tree.fit(X, y)
        leaf_ids = self.tree.apply(X)
        self.leaf_models = {}
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            lr = Ridge(alpha=1e-2)
            if mask.sum() >= 2:
                lr.fit(X[mask], y[mask])
            else:
                lr.fit(X, y)      # 样本太少时退化为全局线性回归，避免过拟合报错
            self.leaf_models[leaf] = lr
        return self

    def predict(self, X):
        leaf_ids = self.tree.apply(X)
        preds = np.zeros(len(X))
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            preds[mask] = self.leaf_models[leaf].predict(X[mask])
        return preds

    def get_hpo_space(self): return {"max_depth": (2, 10)}


# ------------------------------------------------------------------------------
# 6.2 先进与前沿模型（对应截图 17 个：RF, Bagging, ExtraTrees, AdaBoost, GBRT, XGBoost,
#     LightGBM, CatBoost, LSTM, GRU, Transformer, TabNet, CNN, DNN, ResNet, BPNet, RBFNet）
# ------------------------------------------------------------------------------

# ---------- 6.2.1 树模型 / Boosting 集成家族 ----------
class RFModel(BaseModel):
    name, category = "RF", "先进与前沿模型"

    def __init__(self, n_estimators=300, max_depth=None, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, **kwargs); self._build()

    def _build(self):
        self.model = RandomForestRegressor(n_estimators=int(self.params.get("n_estimators", 300)),
                                            max_depth=self.params.get("max_depth", None),
                                            random_state=RANDOM_SEED, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (50, 500)}


class BaggingModel(BaseModel):
    name, category = "Bagging", "先进与前沿模型"

    def __init__(self, n_estimators=100, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs); self._build()

    def _build(self):
        self.model = BaggingRegressor(n_estimators=int(self.params.get("n_estimators", 100)),
                                       random_state=RANDOM_SEED, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (10, 300)}


class ExtraTreesModel(BaseModel):
    name, category = "ExtraTrees", "先进与前沿模型"

    def __init__(self, n_estimators=300, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs); self._build()

    def _build(self):
        self.model = ExtraTreesRegressor(n_estimators=int(self.params.get("n_estimators", 300)),
                                          random_state=RANDOM_SEED, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (50, 500)}


class AdaBoostModel(BaseModel):
    name, category = "AdaBoost", "先进与前沿模型"

    def __init__(self, n_estimators=150, learning_rate=0.1, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs); self._build()

    def _build(self):
        self.model = AdaBoostRegressor(n_estimators=int(self.params.get("n_estimators", 150)),
                                        learning_rate=self.params.get("learning_rate", 0.1),
                                        random_state=RANDOM_SEED)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (30, 300), "learning_rate": (0.01, 1.0)}


class GBRTModel(BaseModel):
    """GBRT —— 梯度提升回归树（sklearn 原生实现）"""
    name, category = "GBRT", "先进与前沿模型"

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, **kwargs)
        self._build()

    def _build(self):
        self.model = GradientBoostingRegressor(n_estimators=int(self.params.get("n_estimators", 200)),
                                                 learning_rate=self.params.get("learning_rate", 0.05),
                                                 max_depth=int(self.params.get("max_depth", 3)),
                                                 random_state=RANDOM_SEED)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (50, 500), "learning_rate": (0.001, 0.3)}


class XGBoostModel(BaseModel):
    name, category = "XGBoost", "先进与前沿模型"

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=5, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, **kwargs)
        if not HAS_XGBOOST:
            raise ImportError("未安装 xgboost，请执行: pip install xgboost")
        self._build()

    def _build(self):
        self.model = xgb.XGBRegressor(n_estimators=int(self.params.get("n_estimators", 300)),
                                       learning_rate=self.params.get("learning_rate", 0.05),
                                       max_depth=int(self.params.get("max_depth", 5)),
                                       random_state=RANDOM_SEED, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (50, 600), "learning_rate": (0.001, 0.3), "max_depth": (2, 12)}


class LightGBMModel(BaseModel):
    name, category = "LightGBM", "先进与前沿模型"

    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, **kwargs)
        if not HAS_LIGHTGBM:
            raise ImportError("未安装 lightgbm，请执行: pip install lightgbm")
        self._build()

    def _build(self):
        self.model = lgb.LGBMRegressor(n_estimators=int(self.params.get("n_estimators", 300)),
                                        learning_rate=self.params.get("learning_rate", 0.05),
                                        num_leaves=int(self.params.get("num_leaves", 31)),
                                        random_state=RANDOM_SEED, verbose=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"n_estimators": (50, 600), "learning_rate": (0.001, 0.3)}


class CatBoostModel(BaseModel):
    name, category = "CatBoost", "先进与前沿模型"

    def __init__(self, iterations=300, learning_rate=0.05, depth=6, **kwargs):
        super().__init__(iterations=iterations, learning_rate=learning_rate, depth=depth, **kwargs)
        if not HAS_CATBOOST:
            raise ImportError("未安装 catboost，请执行: pip install catboost")
        self._build()

    def _build(self):
        self.model = cb.CatBoostRegressor(iterations=int(self.params.get("iterations", 300)),
                                           learning_rate=self.params.get("learning_rate", 0.05),
                                           depth=int(self.params.get("depth", 6)),
                                           random_state=RANDOM_SEED, verbose=False)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def get_hpo_space(self): return {"iterations": (50, 600), "learning_rate": (0.001, 0.3)}


# ---------- 6.2.2 RBFNet（径向基函数网络，自实现，不依赖 torch）----------
class RBFNetModel(BaseModel):
    """
    RBFNet —— 径向基函数网络，标准三步走实现（不依赖深度学习框架，纯 numpy/sklearn）：
        1) 用 KMeans 在训练集特征空间中找若干个"中心点"
        2) 计算每个样本到各中心点的高斯径向基响应，作为隐藏层输出
        3) 对隐藏层输出做线性回归，得到输出层权重
    """
    name, category = "RBFNet", "先进与前沿模型"

    def __init__(self, n_centers=30, gamma=0.5, **kwargs):
        super().__init__(n_centers=n_centers, gamma=gamma, **kwargs)
        self.kmeans = None; self.linear = None

    def _build(self): pass

    def _rbf_features(self, X):
        d2 = np.sum((X[:, None, :] - self.kmeans.cluster_centers_[None, :, :]) ** 2, axis=2)
        return np.exp(-self.params.get("gamma", 0.5) * d2)

    def fit(self, X, y):
        n_centers = min(int(self.params.get("n_centers", 30)), len(X))
        self.kmeans = KMeans(n_clusters=n_centers, random_state=RANDOM_SEED, n_init=10).fit(X)
        H = self._rbf_features(X)
        self.linear = Ridge(alpha=1e-2).fit(H, y)
        return self

    def predict(self, X):
        H = self._rbf_features(X)
        return self.linear.predict(H)

    def get_hpo_space(self): return {"n_centers": (5, 100), "gamma": (0.01, 2.0)}


# ---------- 6.2.3 深度学习模型家族（LSTM/GRU/Transformer/CNN/DNN/ResNet/BPNet/TabNet） ----------
if HAS_TORCH:

    class _SeqDataset(TensorDataset):
        """内部小工具：把 (窗口数, 时间步, 特征) 的三维数组包装成 torch 数据集"""
        pass

    class TorchModelBase(BaseModel):
        """
        所有基于 PyTorch 的算法共用的训练/预测框架，避免每个网络都重复写一遍训练循环。
        子类只需要实现 _build_network(n_features) -> nn.Module 即可。
        """
        epochs = 60
        batch_size = 64
        lr = 1e-3

        def __init__(self, window_size: int = 20, **kwargs):
            super().__init__(window_size=window_size, **kwargs)
            self.window_size = window_size
            self.net: Optional[nn.Module] = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _build(self):
            pass  # 网络需要知道输入维度，延迟到 fit() 里再真正构建

        def _build_network(self, n_features: int) -> "nn.Module":
            raise NotImplementedError

        def _reshape_for_sequence(self, X: np.ndarray) -> np.ndarray:
            """
            FeatureEngineer.build_supervised_samples() 输出的 X 是展平的
            (样本数, window_size * 特征数)，这里还原成 (样本数, window_size, 特征数)
            供 LSTM/GRU/Transformer/CNN 等需要时间步结构的网络使用。
            """
            n_feat_total = X.shape[1]
            n_features = n_feat_total // self.window_size
            return X.reshape(-1, self.window_size, n_features)

        def fit(self, X, y):
            torch.manual_seed(RANDOM_SEED)
            X_seq = self._reshape_for_sequence(X)
            n_features = X_seq.shape[2]
            self.net = self._build_network(n_features).to(self.device)

            X_t = torch.tensor(X_seq, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.get("lr", self.lr))
            loss_fn = nn.MSELoss()

            self.net.train()
            for _ in range(int(self.params.get("epochs", self.epochs))):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = self.net(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
            return self

        def predict(self, X):
            X_seq = self._reshape_for_sequence(X)
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            self.net.eval()
            with torch.no_grad():
                pred = self.net(X_t).cpu().numpy().ravel()
            return pred

        def get_hpo_space(self):
            return {"lr": (1e-4, 1e-2)}

    class LSTMModel(TorchModelBase):
        name, category = "LSTM", "先进与前沿模型"

        def _build_network(self, n_features):
            hidden = int(self.params.get("hidden_size", 64))
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
                    self.fc = nn.Linear(hidden, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            return Net()

    class GRUModel(TorchModelBase):
        name, category = "GRU", "先进与前沿模型"

        def _build_network(self, n_features):
            hidden = int(self.params.get("hidden_size", 64))
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = nn.GRU(n_features, hidden, batch_first=True)
                    self.fc = nn.Linear(hidden, 1)
                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.fc(out[:, -1, :])
            return Net()

    class TransformerModel(TorchModelBase):
        """Transformer Encoder，对时间步序列做自注意力编码后回归"""
        name, category = "Transformer", "先进与前沿模型"

        def _build_network(self, n_features):
            d_model = int(self.params.get("d_model", 32))
            nhead = int(self.params.get("nhead", 4))
            window_size = self.window_size

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_proj = nn.Linear(n_features, d_model)
                    self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
                    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=64, batch_first=True)
                    self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                    self.fc = nn.Linear(d_model, 1)
                def forward(self, x):
                    x = self.input_proj(x) + self.pos_embed
                    out = self.encoder(x)
                    return self.fc(out[:, -1, :])
            return Net()

        def get_hpo_space(self):
            return {"lr": (1e-4, 1e-2)}

    class CNNModel(TorchModelBase):
        """1D 卷积网络，把时间步维度当作"信号长度"做卷积"""
        name, category = "CNN", "先进与前沿模型"

        def _build_network(self, n_features):
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.fc = nn.Linear(16, 1)
                    self.act = nn.ReLU()
                def forward(self, x):
                    x = x.transpose(1, 2)                 # (batch, features, time)
                    x = self.act(self.conv1(x))
                    x = self.act(self.conv2(x))
                    x = self.pool(x).squeeze(-1)
                    return self.fc(x)
            return Net()

    class DNNModel(TorchModelBase):
        """普通全连接深度神经网络（把窗口展平后直接过多层 MLP）"""
        name, category = "DNN", "先进与前沿模型"

        def _build_network(self, n_features):
            in_dim = self.window_size * n_features
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Flatten(), nn.Linear(in_dim, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
                    )
                def forward(self, x): return self.net(x)
            return Net()

    class ResNetModel(TorchModelBase):
        """带残差连接的全连接网络（1D ResNet 风格，缓解深层网络训练退化问题）"""
        name, category = "ResNet", "先进与前沿模型"

        def _build_network(self, n_features):
            in_dim = self.window_size * n_features
            class ResBlock(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.fc1 = nn.Linear(dim, dim); self.fc2 = nn.Linear(dim, dim)
                    self.act = nn.ReLU()
                def forward(self, x):
                    out = self.act(self.fc1(x)); out = self.fc2(out)
                    return self.act(x + out)                   # 残差连接
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.input_fc = nn.Linear(in_dim, 64)
                    self.blocks = nn.Sequential(ResBlock(64), ResBlock(64))
                    self.out_fc = nn.Linear(64, 1)
                def forward(self, x):
                    x = self.flatten(x)
                    x = torch.relu(self.input_fc(x))
                    x = self.blocks(x)
                    return self.out_fc(x)
            return Net()

    class BPNetModel(TorchModelBase):
        """
        BPNet —— 基于 PyTorch 手动实现的 BP 神经网络（区别于 6.1 中基于 sklearn 的 BP/ANN，
        便于和其它深度学习模型放在同一训练框架/同一设备下对比）
        """
        name, category = "BPNet", "先进与前沿模型"

        def _build_network(self, n_features):
            in_dim = self.window_size * n_features
            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Flatten(), nn.Linear(in_dim, 32), nn.Sigmoid(), nn.Linear(32, 1)
                    )
                def forward(self, x): return self.net(x)
            return Net()

    class TabNetModelWrapper(BaseModel):
        """
        TabNet —— 若安装了 pytorch_tabnet 则调用其官方实现（真实的注意力特征选择机制）；
        若未安装，则自动降级为一个等价复杂度的 DNN 近似，并在日志中明确提示，
        保证在没有装 pytorch-tabnet 时程序依然能跑通，而不是直接报错崩溃。
        """
        name, category = "TabNet", "先进与前沿模型"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._fallback = not HAS_TABNET
            if self._fallback:
                print("[提示] 未安装 pytorch_tabnet，TabNet 自动降级为 DNN 近似实现。"
                      "如需真正的 TabNet，请执行: pip install pytorch-tabnet")
                self._fallback_model = DNNModel(**kwargs)
            else:
                self.model = TabNetRegressor(seed=RANDOM_SEED, verbose=0)

        def fit(self, X, y):
            if self._fallback:
                self._fallback_model.fit(X, y)
            else:
                self.model.fit(X, y.reshape(-1, 1), max_epochs=80, patience=15)
            return self

        def predict(self, X):
            if self._fallback:
                return self._fallback_model.predict(X)
            return self.model.predict(X).ravel()

else:
    # 若未安装 torch，以下几个类在被实例化时直接抛出清晰的错误提示，而不是让程序在导入阶段崩溃
    class _TorchUnavailableModel(BaseModel):
        category = "先进与前沿模型"
        def __init__(self, **kwargs):
            raise ImportError(f"算法 {self.name} 依赖 torch，请先执行: pip install torch")
        def fit(self, X, y): ...
        def predict(self, X): ...

    class LSTMModel(_TorchUnavailableModel): name = "LSTM"
    class GRUModel(_TorchUnavailableModel): name = "GRU"
    class TransformerModel(_TorchUnavailableModel): name = "Transformer"
    class CNNModel(_TorchUnavailableModel): name = "CNN"
    class DNNModel(_TorchUnavailableModel): name = "DNN"
    class ResNetModel(_TorchUnavailableModel): name = "ResNet"
    class BPNetModel(_TorchUnavailableModel): name = "BPNet"
    class TabNetModelWrapper(_TorchUnavailableModel): name = "TabNet"


# ------------------------------------------------------------------------------
# 6.3 公式拟合 / 演化计算模型（对应截图 3 个：SR 符号回归, GEP, MEP）
# ------------------------------------------------------------------------------
class SRModel(BaseModel):
    """SR —— 符号回归 (Symbolic Regression)，用遗传编程自动搜索"数学公式"来拟合数据，
    优点是能得到可解释的显式公式（如 y = 0.3*x1 + sin(x2) - 1.2），而不是黑箱模型。
    """
    name, category = "SR", "公式拟合/演化计算"

    def __init__(self, population_size=800, generations=15, **kwargs):
        super().__init__(population_size=population_size, generations=generations, **kwargs)
        if not HAS_GPLEARN:
            raise ImportError("未安装 gplearn，请执行: pip install gplearn")
        self._build()

    def _build(self):
        self.model = SymbolicRegressor(
            population_size=int(self.params.get("population_size", 800)),
            generations=int(self.params.get("generations", 15)),
            function_set=("add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"),
            random_state=RANDOM_SEED, n_jobs=-1, verbose=0
        )

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)

    def get_formula(self) -> str:
        """返回拟合出的可解释数学公式（SR 相比其它算法的最大卖点）"""
        return str(self.model._program) if self.model is not None else ""

    def get_hpo_space(self): return {"generations": (5, 40)}


class GEPModel(SRModel):
    """
    GEP —— 基因表达式编程 (Gene Expression Programming)，近似实现说明：
    GEP 是 SR 的一个变种，核心区别在于"基因型-表现型分离"的编码方式（线性染色体
    编码 + 树状表达），Python 生态中的成熟实现是 geppy 库（需要额外配置遗传算子）。
    为了保证在未安装 geppy 时该功能依然"能跑通、不报错"，此处临时复用 gplearn 的
    符号回归引擎（同属演化计算搜索数学公式的思路）作为近似替代。
    如需严格意义上的 GEP，请安装 geppy 库并替换本类的 fit()/predict() 实现：
        pip install geppy
    """
    name = "GEP"


class MEPModel(SRModel):
    """
    MEP —— 多表达式编程 (Multi Expression Programming)，近似实现说明：
    MEP 同样是演化计算搜索公式家族的一员，特点是单个染色体可以同时编码多个候选表达式，
    Python 生态没有维护活跃的成熟库。此处同样临时复用 gplearn 符号回归引擎作为近似替代，
    保证整体训练流程可以跑通。如需严格意义上的 MEP，需要自行实现"多表达式染色体解码"
    逻辑，或参考 C++ 版 MEPX 项目做二次开发。
    """
    name = "MEP"


# ------------------------------------------------------------------------------
# 6.3b 经典统计时序模型：ARIMA
# ------------------------------------------------------------------------------
class ARIMAModel(BaseModel):
    """
    ARIMA —— 差分自回归移动平均，统计意义上"真正的时序模型"（不依赖滑窗特征）。

    在本框架里的定位与做法：
      - fit()   : 只用训练段的**目标序列** y 本身来拟合 ARIMA(p,d,q)，忽略窗口特征 X。
      - predict(): 对测试段做**多步向前预测**(forecast)，一次性给出整段测试期的预测。
    这是"纯时序模型"的经典对照。要提醒的是：在股票这种近乎随机游走的序列上，ARIMA 的多步预测
    往往会迅速收敛到均值（价格模式下≈一条平线，收益率模式下≈常数），因此它常常**跑不赢朴素基准**——
    这恰恰印证了"复杂/经典时序模型≠更准"。把它放进来，是为了让对照更诚实、更完整。

    依赖 statsmodels：  pip install statsmodels
    """
    name, category = "ARIMA", "经典时序模型"

    def __init__(self, p: int = 5, d: int = 1, q: int = 0, **kwargs):
        super().__init__(p=p, d=d, q=q, **kwargs)
        if not HAS_STATSMODELS:
            raise ImportError("未安装 statsmodels，请执行: pip install statsmodels")
        self._fitted = None

    def fit(self, X, y):
        order = (int(self.params.get("p", 5)), int(self.params.get("d", 1)),
                 int(self.params.get("q", 0)))
        self._fitted = _ARIMA(np.asarray(y, dtype=float), order=order).fit()
        return self

    def predict(self, X):
        n = len(X)
        fc = np.asarray(self._fitted.forecast(steps=n), dtype=float)
        # 极端参数下 statsmodels 偶尔返回 NaN，兜底为最后一次拟合的均值，避免整条结果变 NaN
        if np.any(~np.isfinite(fc)):
            fc = np.nan_to_num(fc, nan=float(np.nanmean(self._fitted.fittedvalues)))
        return fc


# ------------------------------------------------------------------------------
# 6.4 模型注册表 —— 算法的统一入口
# 新增算法时只需要：1) 实现一个 BaseModel 子类  2) 在这里加一行注册，GUI 会自动出现对应勾选框
# ------------------------------------------------------------------------------
ALGO_REGISTRY: Dict[str, Callable[..., BaseModel]] = {
    # ---- 基础与传统模型 (14) ----
    "BP/ANN": ANNModel, "SVR": SVRModel, "LSSVM": LSSVMModel, "GPR": GPRModel,
    "ElasticNet": ElasticNetModel, "RidgeReg": RidgeRegModel, "Lasso": LassoModel,
    "PLSR": PLSRModel, "KNN": KNNModel, "Kstar": KStarModel, "ELM": ELMModel,
    "DTR": DTRModel, "DT": DTModel, "M5Rules": M5RulesModel,
    # ---- 先进与前沿模型 (17) ----
    "RF": RFModel, "Bagging": BaggingModel, "ExtraTrees": ExtraTreesModel,
    "AdaBoost": AdaBoostModel, "GBRT": GBRTModel, "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel, "CatBoost": CatBoostModel,
    "LSTM": LSTMModel, "GRU": GRUModel, "Transformer": TransformerModel,
    "TabNet": TabNetModelWrapper, "CNN": CNNModel, "DNN": DNNModel,
    "ResNet": ResNetModel, "BPNet": BPNetModel, "RBFNet": RBFNetModel,
    # ---- 公式拟合/演化计算 (3) ----
    "SR": SRModel, "GEP": GEPModel, "MEP": MEPModel,
    # ---- 经典时序模型 (1) ----
    "ARIMA": ARIMAModel,
}

# 各算法所需的可选依赖是否满足，GUI 里据此把不可用的算法勾选框置灰
ALGO_AVAILABILITY: Dict[str, bool] = {
    "XGBoost": HAS_XGBOOST, "LightGBM": HAS_LIGHTGBM, "CatBoost": HAS_CATBOOST,
    "LSTM": HAS_TORCH, "GRU": HAS_TORCH, "Transformer": HAS_TORCH, "CNN": HAS_TORCH,
    "DNN": HAS_TORCH, "ResNet": HAS_TORCH, "BPNet": HAS_TORCH, "TabNet": True,  # TabNet 有 DNN 降级方案，永远可用
    "SR": HAS_GPLEARN, "GEP": HAS_GPLEARN, "MEP": HAS_GPLEARN,
    "ARIMA": HAS_STATSMODELS,
}
for _name in ALGO_REGISTRY:
    ALGO_AVAILABILITY.setdefault(_name, True)     # 其余算法只依赖 sklearn，始终可用


# ==================== 第七部分：超参数优化层（PSO / GA / FA / SOA / 贝叶斯优化 BO） ====================
# 对应截图 "①启动超参数优化 / Enable HPO" 里的五个单选项，BO 是系统默认推荐("Best")选项。
#
# 设计说明：
#   前四种（PSO/GA/FA/SOA）都属于"种群类元启发式算法"，思路高度相似——
#   一群候选解在参数空间里按各自的更新规则移动，每一代评估适应度（这里用验证集 RMSE，
#   越小越好），逐步收敛到较优的超参数组合。因此抽象出一个公共基类 PopulationOptimizer，
#   四个子类只需实现"每一代个体如何移动"这一条差异化规则，避免重复代码。
#   第五种（BO，贝叶斯优化）原理不同，用高斯过程对"参数->效果"关系建模，每次采样
#   "最有可能带来改进"的参数点，通常比随机/元启发式搜索更高效，因此是系统默认推荐项。

class PopulationOptimizer:
    """种群类元启发式优化器的公共基类"""

    def __init__(self, n_particles: int = 15, n_iter: int = 20, seed: int = RANDOM_SEED):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def optimize(self, objective: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """
        子类需实现具体更新规则。约定：
            objective(x) : 输入一个参数向量 x（每维在 bounds 对应范围内），返回"越小越好"的损失
            返回 (最优参数向量, 最优损失值)
        """
        raise NotImplementedError

    def _init_population(self, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pop = lo + self.rng.random((self.n_particles, len(bounds))) * (hi - lo)
        return pop, lo, hi


class PSOOptimizer(PopulationOptimizer):
    """粒子群优化 Particle Swarm Optimization"""

    def optimize(self, objective, bounds):
        pop, lo, hi = self._init_population(bounds)
        vel = np.zeros_like(pop)
        fitness = np.array([objective(p) for p in pop])
        pbest, pbest_fit = pop.copy(), fitness.copy()
        gbest = pbest[np.argmin(pbest_fit)].copy()
        gbest_fit = pbest_fit.min()

        w, c1, c2 = 0.6, 1.5, 1.5
        for _ in range(self.n_iter):
            r1, r2 = self.rng.random(pop.shape), self.rng.random(pop.shape)
            vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
            pop = np.clip(pop + vel, lo, hi)
            fitness = np.array([objective(p) for p in pop])
            improve = fitness < pbest_fit
            pbest[improve] = pop[improve]; pbest_fit[improve] = fitness[improve]
            if pbest_fit.min() < gbest_fit:
                gbest_fit = pbest_fit.min(); gbest = pbest[np.argmin(pbest_fit)].copy()
        return gbest, gbest_fit


class GAOptimizer(PopulationOptimizer):
    """遗传算法 Genetic Algorithm（实数编码：选择 + 算术交叉 + 高斯变异）"""

    def optimize(self, objective, bounds):
        pop, lo, hi = self._init_population(bounds)
        fitness = np.array([objective(p) for p in pop])
        best_idx = np.argmin(fitness); best, best_fit = pop[best_idx].copy(), fitness[best_idx]

        for _ in range(self.n_iter):
            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            elite = pop[:max(2, self.n_particles // 5)]           # 精英保留

            children = [elite[i % len(elite)] for i in range(self.n_particles)]
            children = np.array(children, dtype=float)
            for i in range(len(children)):
                p1, p2 = elite[self.rng.integers(len(elite))], elite[self.rng.integers(len(elite))]
                alpha = self.rng.random()
                child = alpha * p1 + (1 - alpha) * p2                       # 算术交叉
                child += self.rng.normal(0, 0.1, size=child.shape) * (hi - lo)  # 高斯变异
                children[i] = np.clip(child, lo, hi)
            pop = children
            fitness = np.array([objective(p) for p in pop])
            if fitness.min() < best_fit:
                best_fit = fitness.min(); best = pop[np.argmin(fitness)].copy()
        return best, best_fit


class FireflyOptimizer(PopulationOptimizer):
    """萤火虫算法 Firefly Algorithm（较暗的萤火虫向较亮的萤火虫移动，亮度对应适应度）"""

    def optimize(self, objective, bounds):
        pop, lo, hi = self._init_population(bounds)
        fitness = np.array([objective(p) for p in pop])
        beta0, gamma, alpha = 1.0, 1.0, 0.2

        best_idx = np.argmin(fitness); best, best_fit = pop[best_idx].copy(), fitness[best_idx]
        for _ in range(self.n_iter):
            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    if fitness[j] < fitness[i]:                     # j 比 i 更亮，i 向 j 移动
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = beta0 * np.exp(-gamma * r ** 2)
                        pop[i] = pop[i] + beta * (pop[j] - pop[i]) + alpha * self.rng.normal(0, 1, len(bounds)) * (hi - lo)
                        pop[i] = np.clip(pop[i], lo, hi)
                        fitness[i] = objective(pop[i])
            if fitness.min() < best_fit:
                best_fit = fitness.min(); best = pop[np.argmin(fitness)].copy()
        return best, best_fit


class SeagullOptimizer(PopulationOptimizer):
    """海鸥优化算法 Seagull Optimization Algorithm（迁徙 + 螺旋攻击两阶段行为）"""

    def optimize(self, objective, bounds):
        pop, lo, hi = self._init_population(bounds)
        fitness = np.array([objective(p) for p in pop])
        best_idx = np.argmin(fitness); best, best_fit = pop[best_idx].copy(), fitness[best_idx]

        for t in range(self.n_iter):
            Fc = 2 - t * (2 / self.n_iter)          # 控制参数，随迭代线性递减（迁徙->攻击的过渡）
            for i in range(self.n_particles):
                A = Fc * (2 * self.rng.random() - 1)
                C = A * pop[i]
                Dp = np.abs(best + C - pop[i])
                u, v = 1.0, 1.0
                theta = self.rng.random() * 2 * np.pi
                spiral = Dp * np.exp(u * theta) * np.cos(theta) * v
                pop[i] = np.clip(best + spiral, lo, hi)
            fitness = np.array([objective(p) for p in pop])
            if fitness.min() < best_fit:
                best_fit = fitness.min(); best = pop[np.argmin(fitness)].copy()
        return best, best_fit


class BayesianOptimizer:
    """
    贝叶斯优化 BO（系统默认推荐项）。
    优先使用 optuna（基于 TPE 的高效贝叶斯优化）；若未安装 optuna，
    自动退化为"拉丁超立方式"随机搜索，保证功能始终可用。
    """

    def __init__(self, n_trials: int = 20, seed: int = RANDOM_SEED):
        self.n_trials = n_trials
        self.seed = seed

    def optimize(self, objective: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        if HAS_OPTUNA:
            def optuna_objective(trial):
                x = np.array([trial.suggest_float(f"x{i}", b[0], b[1]) for i, b in enumerate(bounds)])
                return objective(x)
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(optuna_objective, n_trials=self.n_trials, show_progress_bar=False)
            best = np.array([study.best_params[f"x{i}"] for i in range(len(bounds))])
            return best, study.best_value
        else:
            # ---- 退化方案：随机搜索 ----
            rng = np.random.default_rng(self.seed)
            lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
            best, best_fit = None, np.inf
            for _ in range(self.n_trials):
                x = lo + rng.random(len(bounds)) * (hi - lo)
                fit = objective(x)
                if fit < best_fit:
                    best_fit, best = fit, x
            return best, best_fit


HPO_REGISTRY: Dict[str, Callable] = {
    "PSO": PSOOptimizer, "GA": GAOptimizer, "FA": FireflyOptimizer,
    "SOA": SeagullOptimizer, "BO": BayesianOptimizer,
}


def run_hpo(method: str, model_cls: Callable[..., BaseModel],
            X_train, y_train, X_val, y_val, n_trials: int = 20,
            model_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    对某个模型类执行超参数寻优，返回最优超参数字典。
    若该模型的 get_hpo_space() 返回空字典（不支持寻优），直接返回空字典，调用方使用默认参数即可。

    model_kwargs：与超参数无关、但构造模型时必须传入的固定参数（如深度学习模型的 window_size），
                  会在每次试验里和寻优参数合并后一起传给 model_cls。
    """
    model_kwargs = model_kwargs or {}
    probe = model_cls(**model_kwargs)
    space = probe.get_hpo_space()
    if not space:
        return {}

    param_names = list(space.keys())
    bounds = [space[k] for k in param_names]

    def objective(x: np.ndarray) -> float:
        params = dict(zip(param_names, x))
        try:
            m = model_cls(**{**model_kwargs, **params})
            m.fit(X_train, y_train)
            pred = m.predict(X_val)
            return float(np.sqrt(np.mean((y_val - pred) ** 2)))     # 用验证集 RMSE 作为寻优目标
        except Exception:
            return 1e9    # 参数不合法/训练失败时给一个很大的惩罚值，让优化器避开这个区域

    optimizer_cls = HPO_REGISTRY.get(method, BayesianOptimizer)
    optimizer = optimizer_cls(n_trials=n_trials) if method == "BO" else optimizer_cls(n_iter=n_trials)
    best_x, _ = optimizer.optimize(objective, bounds)
    return dict(zip(param_names, best_x))


# ==================== 第八部分：训练调度层 ====================
@dataclass
class ModelResult:
    """单个模型的训练/评估结果，供 GUI 展示与绘图使用"""
    algo_name: str
    metrics: Dict[str, float]
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    test_dates: pd.Series
    cv_metrics: Optional[Dict[str, float]] = None
    formula: Optional[str] = None          # 仅 SR/GEP/MEP 有意义
    error: Optional[str] = None            # 若训练失败，记录错误信息，不让一个算法出错影响其它算法
    prev_close: Optional[np.ndarray] = None  # 每个测试样本的"基准日收盘价"，供方向性收益回测使用
    metrics_train: Optional[Dict[str, float]] = None  # 训练集误差（看是否过拟合）
    metrics_val: Optional[Dict[str, float]] = None    # 独立验证集误差（不参与训练/测试/寻优）
    feature_importance: Optional[List[Tuple[str, float]]] = None  # 模型最看重的输入(按重要性排序)
    bar_info: Optional[Dict[str, np.ndarray]] = None  # 测试段逐样本日线细节(涨跌停/停牌判断，供真实化回测)


@dataclass
class PreparedData:
    """一次数据准备的产物。y 处于"目标空间"(price 或 return)，未缩放；
    评估/画图统一用价格空间：真实值=close_*，预测值由各模型输出还原得到。
    按时间顺序三分：训练集(train) / 验证集(val，独立留出) / 测试集(test，最终留出)。"""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray                # 目标空间(price 或 return)，未缩放
    y_val: np.ndarray
    y_test: np.ndarray
    d_train: pd.Series
    d_val: pd.Series
    d_test: pd.Series
    prev_close_train: np.ndarray       # 每个样本"基准日"收盘价
    prev_close_val: np.ndarray
    prev_close_test: np.ndarray
    close_train: np.ndarray            # 每个样本"真实未来"收盘价(价格空间)
    close_val: np.ndarray
    close_test: np.ndarray
    bar_info_test: Optional[Dict[str, np.ndarray]] = None  # 测试段逐样本日线细节(供真实化回测)


class TrainingPipeline:
    """
    训练总调度器：给定一批算法名字 + 一份数据 + 一份配置，依次训练、评估、（可选）交叉验证，
    每个模型互相独立，单个模型失败不影响其它模型继续跑。
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.fe = FeatureEngineer(window_size=config.window_size, horizon=config.horizon,
                                  target_mode=config.target_mode)

    def _to_price(self, pred_target: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        """把模型在"目标空间"的预测统一还原成"价格"：
        return 模式: 预测价 = 基准日收盘 × (1 + 预测涨跌幅)；price 模式: 预测价 = 预测值本身。"""
        if self.config.target_mode == "return":
            return prev_close * (1.0 + pred_target)
        return pred_target

    @staticmethod
    def _dir_metrics(close_true, price_pred, prev_close) -> Dict[str, float]:
        """
        对**任意预测周期(horizon)**都正确的方向指标：以每个样本自己的"基准日收盘价"prev_close
        为参照，比较"未来是否上涨"的真实与预测。这样第二天/3日/一周/一月/三月的涨跌都能正确评估
        （通用版 Metrics.da/up_p 用相邻真值近似，仅 horizon=1 才准，故这里在管道内用真实基准价覆盖）。
        """
        ct = np.asarray(close_true, float); pp = np.asarray(price_pred, float)
        pc = np.asarray(prev_close, float)
        true_up = ct > pc
        pred_up = pp > pc
        da = float(np.mean(true_up == pred_up) * 100) if len(ct) else 0.0
        up_p = float(np.mean(true_up[pred_up]) * 100) if pred_up.sum() > 0 else 0.0
        return {"DA": round(da, 6), "UP_P": round(up_p, 6)}

    def _apply_dir(self, metrics: Dict[str, float], close_true, price_pred, prev_close):
        """把管道内算出的、对任意周期都正确的 DA/UP_P 覆盖进 metrics（仅当用户勾选了它们）。"""
        dm = self._dir_metrics(close_true, price_pred, prev_close)
        for k in ("DA", "UP_P"):
            if k in metrics:
                metrics[k] = dm[k]
        return metrics

    # ---------- 8.1 数据准备：技术指标 -> 样本 -> 标准化 -> 训练/验证/测试三分 ----------
    def prepare_data(self, raw_df: pd.DataFrame) -> PreparedData:
        X, y, dates = self.fe.build_supervised_samples(raw_df)
        pc, ct = self.fe.prev_close_, self.fe.close_target_
        n = len(X)
        # 三分切点（严格按时间顺序，绝不打乱）：
        #   训练 [0, a) / 验证 [a, b) / 测试 [b, n)
        a = int(n * self.config.train_ratio)
        rest = n - a
        n_val = int(n * self.config.val_ratio) if self.config.val_ratio > 0 else rest // 2
        n_val = max(0, min(n_val, rest))
        b = a + n_val
        self.fe.fit_scale(X[:a], y[:a])                          # 只在训练集上 fit，防止数据泄漏
        d = dates.reset_index(drop=True)
        return PreparedData(
            X_train=self.fe.transform(X[:a]), X_val=self.fe.transform(X[a:b]),
            X_test=self.fe.transform(X[b:]),
            y_train=y[:a], y_val=y[a:b], y_test=y[b:],
            d_train=d[:a].reset_index(drop=True), d_val=d[a:b].reset_index(drop=True),
            d_test=d[b:].reset_index(drop=True),
            prev_close_train=pc[:a], prev_close_val=pc[a:b], prev_close_test=pc[b:],
            close_train=ct[:a], close_val=ct[a:b], close_test=ct[b:],
            bar_info_test={k: v[b:] for k, v in getattr(self.fe, "bar_info_", {}).items()},
        )

    def _feature_importance(self, model, topn: int = 8):
        """取模型对各输入特征的重要性(按原始特征聚合窗口内各天)。树模型用 feature_importances_、
        线性模型用 |系数|；其余(SVR-rbf/深度学习)无标准重要性则返回 None。返回 [(特征名, 占比%)]。"""
        est = getattr(model, "model", None)
        imp = None
        if est is not None:
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_, dtype=float).ravel()
            elif hasattr(est, "coef_"):
                imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        feats = list(getattr(self.fe, "feature_cols", []))
        n = len(feats)
        if imp is None or n == 0 or len(imp) != n * self.config.window_size:
            return None
        agg = imp.reshape(self.config.window_size, n).sum(axis=0)   # 按原始特征聚合窗口内各天
        total = float(agg.sum()) + 1e-12
        order = np.argsort(agg)[::-1][:topn]
        return [(feats[i], round(float(agg[i] / total * 100), 2)) for i in order if agg[i] > 0]

    def _eval(self, model, X, prev_close, close_true) -> Dict[str, float]:
        """在给定数据集上评估模型：预测→还原价格→算价格类+方向类指标(统一价格口径)。"""
        if X is None or len(X) == 0:
            return {}
        pred_price = self._to_price(self.fe.inverse_y(model.predict(X)), prev_close)
        m = Metrics.calc_all(close_true, pred_price, self.config.metrics)
        self._apply_dir(m, close_true, pred_price, prev_close)
        return m

    # ---------- 8.2 单个模型：训练 + 测试评估（+可选HPO +可选CV）----------
    def run_single(self, algo_name: str, pdata: PreparedData,
                    progress_cb: Optional[Callable[[str], None]] = None) -> ModelResult:
        log = progress_cb or (lambda msg: None)
        model_cls = ALGO_REGISTRY[algo_name]
        # 深度学习模型需要知道窗口长度才能把展平的 X 还原成 (样本, 时间步, 特征)；
        # 其余模型的 __init__ 都带 **kwargs，多传一个 window_size 会被安全忽略。
        mk = {"window_size": self.config.window_size}
        X_train, X_test = pdata.X_train, pdata.X_test
        y_train = pdata.y_train
        close_test = pdata.close_test                # 价格空间的真实值（统一评估口径）

        try:
            # ---- 8.2.1 目标值缩放（训练更稳定），预测后再逆缩放回目标空间 ----
            y_train_s = self.fe.scaler_y.transform(y_train.reshape(-1, 1)).ravel()

            best_params = {}
            # ---- 8.2.2 超参数优化（若启用） ----
            if self.config.hpo_method and self.config.hpo_method != "关闭":
                log(f"[{algo_name}] 正在用 {self.config.hpo_method} 方法做超参数寻优 ...")
                n_val = max(1, int(len(X_train) * 0.2))
                X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
                y_tr, y_val = y_train_s[:-n_val], y_train_s[-n_val:]
                best_params = run_hpo(self.config.hpo_method, model_cls, X_tr, y_tr, X_val, y_val,
                                       n_trials=self.config.hpo_trials, model_kwargs=mk)

            # ---- 8.2.3 用（寻优得到的/默认的）参数训练最终模型 ----
            log(f"[{algo_name}] 训练中 ...")
            model = model_cls(**{**mk, **best_params})
            model.fit(X_train, y_train_s)

            # ---- 8.2.4 训练/验证/测试三集分别评估（统一还原成价格口径）----
            pred_price = self._to_price(self.fe.inverse_y(model.predict(X_test)),
                                        pdata.prev_close_test)
            metrics = Metrics.calc_all(close_test, pred_price, self.config.metrics)
            self._apply_dir(metrics, close_test, pred_price, pdata.prev_close_test)
            metrics_train = self._eval(model, X_train, pdata.prev_close_train, pdata.close_train)
            metrics_val = self._eval(model, pdata.X_val, pdata.prev_close_val, pdata.close_val)
            m_te = metrics.get("RMSE", "?"); m_tr = metrics_train.get("RMSE", "?")
            log(f"[{algo_name}] 完成：训练RMSE={m_tr} / 验证RMSE={metrics_val.get('RMSE','-')} "
                f"/ 测试RMSE={m_te}，测试DA={metrics.get('DA','-')}%")

            # ---- 8.2.5 可选：5折交叉验证（衡量稳定性；仅算非方向类指标，方向类需价格还原故略）----
            cv_metrics = None
            if self.config.use_cv:
                log(f"[{algo_name}] 正在做 {self.config.cv_folds} 折交叉验证 ...")
                cv_metrics = self._cross_validate(model_cls, best_params, X_train, y_train_s,
                                                  y_train, mk)

            formula = model.get_formula() if hasattr(model, "get_formula") else None

            return ModelResult(algo_name=algo_name, metrics=metrics, y_test_true=close_test,
                                y_test_pred=pred_price, test_dates=pdata.d_test,
                                cv_metrics=cv_metrics, formula=formula,
                                prev_close=pdata.prev_close_test,
                                metrics_train=metrics_train, metrics_val=metrics_val,
                                feature_importance=self._feature_importance(model),
                                bar_info=pdata.bar_info_test)
        except Exception as e:
            log(f"[{algo_name}] 训练失败：{e}")
            return ModelResult(algo_name=algo_name, metrics={}, y_test_true=close_test,
                                y_test_pred=np.full_like(close_test, np.nan, dtype=float),
                                test_dates=pdata.d_test, error=str(e))

    # ---------- 8.3 K 折交叉验证（时间序列专用：前扩窗口 walk-forward，只用过去预测未来，绝不泄漏）----------
    def _cross_validate(self, model_cls, best_params, X, y_scaled, y_raw,
                        model_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """时间序列交叉验证必须用『前扩窗口』：第 k 折只用它**之前**的数据训练、预测紧邻的下一折。
        绝不能像普通 K 折那样把验证折之后的『未来』也拿去训练——那是用未来预测过去的数据泄漏。
        (评估口径说明：DA/UP_P 需价格还原故此处略去；其余指标在目标空间 price/return 内计算，
         与主结果表的价格空间口径不同，仅用于横向比较各折的稳定性。)"""
        model_kwargs = model_kwargs or {}
        cv_metric_names = [m for m in self.config.metrics if m not in ("DA", "UP_P")]
        n = len(X)
        folds = max(2, self.config.cv_folds)
        # 把序列切成 folds+1 段：第 1 段作最初训练集，其后每段依次作一折验证(训练集不断向前扩张)
        block = n // (folds + 1)
        fold_metrics = []
        if block < 1:
            return {}
        for k in range(1, folds + 1):
            start, end = k * block, (k + 1) * block if k < folds else n
            if start >= end - 1 or start < 1:
                continue
            X_tr = X[:start]                 # 只用验证折之前的数据(前扩窗口)，杜绝未来泄漏
            y_tr = y_scaled[:start]
            X_va, y_va_raw = X[start:end], y_raw[start:end]
            try:
                m = model_cls(**{**model_kwargs, **best_params})
                m.fit(X_tr, y_tr)
                pred = self.fe.inverse_y(m.predict(X_va))
                fold_metrics.append(Metrics.calc_all(y_va_raw, pred, cv_metric_names))
            except Exception:
                continue
        if not fold_metrics:
            return {}
        # 对各折结果取平均，得到交叉验证的综合评价指标
        avg = {}
        for key in fold_metrics[0]:
            avg[key] = round(float(np.mean([f[key] for f in fold_metrics if key in f])), 6)
        return avg

    # ---------- 8.4 诚实性参照基准（价格空间）----------
    def _naive_baseline(self, pdata: PreparedData) -> ModelResult:
        """
        "前值持有"朴素基准：预测明天收盘价 = 今天(基准日)收盘价（即预测涨跌幅为 0）。
        它在"价格水平"上往往 R² 很高，却毫无方向判别力(DA≈0)——用来揭穿"R² 虚高"。
        """
        def _naive_m(prev, close):
            if prev is None or len(prev) == 0:
                return {}
            p = prev.astype(float)
            m = Metrics.calc_all(close, p, self.config.metrics)
            self._apply_dir(m, close, p, prev)
            return m
        metrics = _naive_m(pdata.prev_close_test, pdata.close_test)
        return ModelResult(algo_name="Naive(前值)", metrics=metrics, y_test_true=pdata.close_test,
                           y_test_pred=pdata.prev_close_test.astype(float), test_dates=pdata.d_test,
                           prev_close=pdata.prev_close_test,
                           metrics_train=_naive_m(pdata.prev_close_train, pdata.close_train),
                           metrics_val=_naive_m(pdata.prev_close_val, pdata.close_val))

    def _alwaysup_baseline(self, pdata: PreparedData) -> ModelResult:
        """
        "总是猜涨"方向基准：每天都预测上涨。它的方向准确率(DA)与上涨精确率(UP_P) 就等于
        测试集里"上涨日占比"。**任何模型的 DA/UP_P 只有明显高于这一行，才算真的有涨跌判别力。**
        （价格类指标对本行无意义，请只看 DA / UP_P。）
        """
        def _up_m(prev, close):
            if prev is None or len(prev) == 0:
                return {}
            p = prev.astype(float) * 1.001
            m = Metrics.calc_all(close, p, self.config.metrics)
            self._apply_dir(m, close, p, prev)
            return m
        up_price = pdata.prev_close_test.astype(float) * 1.001
        return ModelResult(algo_name="总是涨(方向基准)", metrics=_up_m(pdata.prev_close_test, pdata.close_test),
                           y_test_true=pdata.close_test, y_test_pred=up_price,
                           test_dates=pdata.d_test, prev_close=pdata.prev_close_test,
                           metrics_train=_up_m(pdata.prev_close_train, pdata.close_train),
                           metrics_val=_up_m(pdata.prev_close_val, pdata.close_val))

    # ---------- 8.5 批量运行多个算法 ----------
    def run_batch(self, algo_names: List[str], raw_df: pd.DataFrame,
                  progress_cb: Optional[Callable[[str], None]] = None) -> List[ModelResult]:
        log = progress_cb or (lambda msg: None)
        pdata = self.prepare_data(raw_df)
        def _span(dseries):
            if dseries is None or len(dseries) == 0:
                return "空"
            return (f"{pd.to_datetime(dseries.iloc[0]).strftime('%Y-%m-%d')}~"
                    f"{pd.to_datetime(dseries.iloc[-1]).strftime('%Y-%m-%d')}")
        log(f"[数据集三分] 训练 {len(pdata.y_train)}条({_span(pdata.d_train)}) | "
            f"验证 {len(pdata.y_val)}条({_span(pdata.d_val)}) | 测试 {len(pdata.y_test)}条({_span(pdata.d_test)})")
        log(f"[无泄露] 标准化器只在训练集上 fit；验证集为独立留出(不参与训练/测试/寻优)；测试集最终留出。")
        results = []
        # 两个诚实性基准放最前面，一眼对照：模型须在"价格误差"上赢过 Naive、在"方向"上赢过"总是涨"。
        # 对任意预测周期都成立：用每个样本的"基准日收盘价"判断这段周期到底涨没涨。
        if self.config.add_naive_baseline and len(pdata.close_test) >= 2:
            results.append(self._naive_baseline(pdata))
            results.append(self._alwaysup_baseline(pdata))
            up_rate = float(np.mean(pdata.close_test > pdata.prev_close_test) * 100)
            log(f"[诚实性提示] 预测周期={self.config.horizon}个交易日；测试集共 {len(pdata.close_test)} 段，"
                f"其中上涨占比 {up_rate:.1f}%。模型方向准确率(DA) 必须明显 > {up_rate:.1f}% 才算真有判别力。")
        for algo_name in algo_names:
            results.append(self.run_single(algo_name, pdata, progress_cb))
        return results

    # ---------- 8.6 向前预测：用最新的窗口预测"数据之后"的收盘价（真正的"预测未来"）----------
    def predict_next(self, algo_name: str, raw_df: pd.DataFrame,
                     progress_cb: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        与 run_single 的"测试集回测"不同，本方法用【全部历史数据】训练一个模型，
        再用【最新的 window_size 天】作为输入，预测最后一个交易日之后第 horizon 天的收盘价。
        这才是日常使用中"预测明天"的场景。

        返回一个字典：
            {
              "algo": 算法名,
              "last_date": 历史数据里最后一天的日期(字符串),
              "last_close": 最后一天的真实收盘价,
              "pred_close": 模型预测的未来收盘价,
              "pred_change_pct": 相对最后收盘价的预测涨跌幅(%),
            }
        """
        log = progress_cb or (lambda msg: None)
        mk = {"window_size": self.config.window_size}

        # 1) 用全部历史构造样本并在全量数据上拟合缩放器与模型
        X, y, _ = self.fe.build_supervised_samples(raw_df)
        self.fe.fit_scale(X, y)
        X_s, y_s = self.fe.transform(X, y)
        log(f"[{algo_name}] 用全部历史数据训练用于向前预测的模型 ...")
        model = ALGO_REGISTRY[algo_name](**mk)
        model.fit(X_s, y_s)

        # 2) 取"最新的一整个窗口"（含最后一个交易日），构造一条待预测样本
        df_ind = self.fe.add_technical_indicators(raw_df)
        df_ind = df_ind.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        last_window = df_ind[self.fe.feature_cols].values[-self.config.window_size:]
        x_new = self.fe.scaler_x.transform(last_window.flatten().reshape(1, -1))
        pred_target = float(self.fe.inverse_y(model.predict(x_new))[0])

        last_close = float(df_ind["close"].iloc[-1])
        # 按目标模式统一还原成"预测收盘价"
        pred_close = float(last_close * (1.0 + pred_target)) \
            if self.config.target_mode == "return" else pred_target
        last_date = pd.to_datetime(df_ind["date"].iloc[-1]).strftime("%Y-%m-%d") \
            if "date" in df_ind.columns else ""
        return {
            "algo": algo_name,
            "last_date": last_date,
            "last_close": round(last_close, 4),
            "pred_close": round(pred_close, 4),
            "pred_change_pct": round((pred_close - last_close) / (last_close + 1e-12) * 100, 3),
        }


# ==================== 第八部分补充：方向性收益回测（带手续费的诚实检验） ====================
def board_limit_pct(code: Optional[str], is_st: bool = False) -> float:
    """按 A 股板块返回单日涨跌停幅度(小数)：主板±10%，创业板(30x)/科创板(688)±20%，
    北交所(8/4 开头)±30%，ST/*ST ±5%。无法判断时按主板 10%。"""
    c = (code or "").strip()
    if is_st:
        return 0.05
    if c.startswith(("300", "301", "688")):
        return 0.20
    if c.startswith(("8", "4", "920")):        # 北交所
        return 0.30
    return 0.10


def backtest_directional(prev_close, close_target, pred_price, dates,
                         horizon: int = 1, cost_bps: float = 0.2,
                         ann_days: int = 252, bar_info: Optional[Dict[str, np.ndarray]] = None,
                         code: Optional[str] = None, is_st: bool = False,
                         bench_close=None, rf_annual: float = 0.0,
                         vol_target_annual: float = 0.0, vol_lookback: int = 20,
                         stop_loss_pct: float = 0.0) -> Dict[str, Any]:
    """
    "跟着预测做多/空仓"策略回测（贴合 A 股真实交易制度）：每一段(周期=horizon)开始时，若模型预测
    **上涨**(pred>基准价)就买入满仓，持有到该段结束(≥1 日，天然满足 T+1)，否则空仓。
    每次真正持有(买入→卖出)扣一次**往返成本** cost_bps(%)(佣金+印花税+滑点)。非重叠切段避免交易重叠。

    真实化(需传入 bar_info)：
      · 涨停买不进：预测该买、但基准日是**一字涨停**(最高==最低且当日≈+涨停幅)→ 无法买入，放弃该段。
      · 跌停卖不出：持有中、但目标日是**一字跌停**→ 无法在收盘卖出，计为“卡跌停”段并提示(收益仍按目标价计，
        属乐观口径，实盘可能更差)。
      · 停牌：基准日成交量为 0 → 无法买入，放弃该段。
    另可传入 bench_close(与各段对齐的沪深300收盘)得到**相对大盘超额**；rf_annual 为年化无风险利率(算夏普用)。

    这是检验"预测涨跌到底能不能赚钱"的终极诚实标准：策略净值须明显跑赢"买入持有"与"大盘"，才算真有用。
    返回统计 + 净值曲线，可直接用于画图与 JSON。
    """
    prev_close = np.asarray(prev_close, float); close_target = np.asarray(close_target, float)
    pred_price = np.asarray(pred_price, float)
    n = len(prev_close)
    if n == 0:
        return {"error": "无测试样本"}
    idx = np.arange(0, n, max(1, horizon))           # 非重叠取段
    pc, ct, pp = prev_close[idx], close_target[idx], pred_price[idx]
    seg_ret = ct / (pc + 1e-12) - 1.0                # 每段真实收益率
    pos = (pp > pc).astype(float)                    # 期望仓位：预测涨=1(满仓)，否则=0(空仓)

    # ---- A 股制度真实化：涨停买不进 / 停牌不可买 / 跌停卖不出 ----
    limit = board_limit_pct(code, is_st)
    n_block_buy = n_suspend = n_stuck_sell = 0
    if bar_info:
        b_ret = np.asarray(bar_info.get("base_ret"), float)[idx]
        b_lock = np.asarray(bar_info.get("base_locked"), float)[idx]
        b_vol = np.asarray(bar_info.get("base_vol"), float)[idx]
        x_ret = np.asarray(bar_info.get("exit_ret"), float)[idx]
        x_lock = np.asarray(bar_info.get("exit_locked"), float)[idx]
        eps = 0.005
        buy_block = (pos > 0) & (b_lock > 0) & (b_ret >= limit - eps)   # 一字涨停买不进
        suspend = (pos > 0) & (b_vol <= 0)                             # 停牌买不进
        n_block_buy = int(buy_block.sum()); n_suspend = int(suspend.sum())
        pos = np.where(buy_block | suspend, 0.0, pos)                  # 这些段无法建仓 → 空仓
        stuck_sell = (pos > 0) & (x_lock > 0) & (x_ret <= -limit + eps)  # 一字跌停卖不出(仅统计提示)
        n_stuck_sell = int(stuck_sell.sum())

    # ---- 风控①：波动率目标仓位(高波动少下注/低波动多下注；只用过去已实现收益，无泄漏) ----
    size = np.ones_like(pos)
    if vol_target_annual and vol_target_annual > 0:
        tgt_seg = vol_target_annual * np.sqrt(max(1, horizon) / ann_days)   # 折算到每段的目标波动
        for k in range(len(pos)):
            past = seg_ret[max(0, k - vol_lookback):k]                      # 只看该段之前的已实现波动
            tv = float(np.std(past)) if len(past) >= 3 else 0.0
            size[k] = min(1.0, tgt_seg / (tv + 1e-9)) if tv > 0 else 1.0    # 只减仓不加杠杆(上限1)
    # ---- 风控②：单段止损(该段实际跌幅超过阈值则这段收益按止损位截断) ----
    seg_eff = seg_ret.copy()
    n_stop = 0
    if stop_loss_pct and stop_loss_pct > 0:
        sl = -abs(stop_loss_pct) / 100.0
        hit = (pos > 0) & (seg_ret < sl)
        n_stop = int(hit.sum())
        seg_eff = np.where(hit, sl, seg_ret)          # 触发止损：这段亏损截断在止损位(近似,未计跳空穿透)

    expo = pos * size                                # 有效敞口(0~1)，含仓位缩放
    gross = np.where(pos > 0, seg_eff * size, 0.0)   # 毛收益(未扣成本)，含仓位缩放
    oneway = (cost_bps / 100.0) / 2.0
    prev_expo = np.concatenate([[0.0], expo[:-1]])
    cost_arr = np.abs(expo - prev_expo) * oneway     # 按有效敞口变化扣单边成本(调仓也计费)
    cost_arr[-1] += expo[-1] * oneway                # 末段若仍持有，结束时清仓再扣一次单边
    strat_ret = gross - cost_arr

    eq_strat = np.cumprod(1.0 + strat_ret)
    eq_bh = np.cumprod(1.0 + seg_ret)                # 买入持有(全程持有)
    years = max(n / ann_days, 1e-9)
    def _ann(total): return (1.0 + total) ** (1.0 / years) - 1.0
    total = float(eq_strat[-1] - 1.0); total_bh = float(eq_bh[-1] - 1.0)
    peak = np.maximum.accumulate(eq_strat); mdd = float((eq_strat / peak - 1.0).min())
    n_trades = int((pos > np.concatenate([[0.0], pos[:-1]])).sum())   # 新建仓次数
    held = pos > 0
    win_rate = float(np.mean(strat_ret[held] > 0) * 100) if held.sum() > 0 else 0.0
    # 夏普：扣每段无风险利率后再年化(rf_annual 折算到每段)
    rf_seg = (1.0 + rf_annual) ** (max(1, horizon) / ann_days) - 1.0
    excess_ret = strat_ret - rf_seg
    sharpe = float(np.mean(excess_ret) / (np.std(strat_ret) + 1e-12) *
                   np.sqrt(ann_days / max(1, horizon))) if len(strat_ret) > 1 else 0.0

    out = {
        "n_segments": int(len(idx)), "n_trades": n_trades,
        "total_return_pct": round(total * 100, 2), "annual_return_pct": round(_ann(total) * 100, 2),
        "buyhold_return_pct": round(total_bh * 100, 2), "buyhold_annual_pct": round(_ann(total_bh) * 100, 2),
        "excess_vs_buyhold_pct": round((total - total_bh) * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2), "win_rate_pct": round(win_rate, 2),
        "sharpe": round(sharpe, 3), "cost_bps": cost_bps, "limit_pct": round(limit * 100, 1),
        "n_block_buy": n_block_buy, "n_suspend": n_suspend, "n_stuck_sell": n_stuck_sell,
        "n_stop": n_stop, "avg_position": round(float(np.mean(expo[pos > 0])) if (pos > 0).any() else 0.0, 3),
        "eq_strat": eq_strat, "eq_bh": eq_bh,
        "seg_dates": pd.Series(dates).reset_index(drop=True).iloc[idx].reset_index(drop=True),
    }
    # ---- 相对大盘(沪深300)超额 ----
    if bench_close is not None:
        bench = np.asarray(bench_close, float)
        if len(bench) >= 2 and np.all(bench[:1] > 0):
            bench_total = float(bench[-1] / bench[0] - 1.0)
            eq_bench = bench / bench[0]
            out["bench_return_pct"] = round(bench_total * 100, 2)
            out["bench_annual_pct"] = round(_ann(bench_total) * 100, 2)
            out["excess_vs_bench_pct"] = round((total - bench_total) * 100, 2)
            out["eq_bench"] = eq_bench
    return out


# ==================== 第八部分补充2：未来走势预测（多周期直接预测，不递归、不造假） ====================
def is_trading_now(now: Optional[dt.datetime] = None) -> Tuple[bool, str]:
    """判断当前是否 A 股交易时段(9:30-11:30 / 13:00-15:00，且为真实交易日)。返回(是否交易中, 状态文字)。
    优先用真实交易日历排除法定节假日；日历取不到时退回『仅排除周末』。"""
    now = now or dt.datetime.now()
    if now.weekday() >= 5:
        return False, "周末休市"
    # 用真实交易日历排除法定节假日(清明/五一/国庆/春节等)；取不到则仅按周末判断
    cal = _load_trade_calendar()
    if cal is not None:
        today = pd.Timestamp(now).normalize()
        if today < cal[0] or today > cal[-1]:
            pass                                  # 超出日历覆盖范围，无法判定节假日，按工作日继续
        elif today not in cal:
            return False, "节假日休市"
    t = now.time()
    if dt.time(9, 30) <= t <= dt.time(11, 30) or dt.time(13, 0) <= t <= dt.time(15, 0):
        return True, "交易中"
    if t < dt.time(9, 30):
        return False, "未开盘(集合竞价9:15-9:25)"
    if dt.time(11, 30) < t < dt.time(13, 0):
        return False, "午间休市"
    return False, "已收盘"


# ==================== 第八部分补充3：风险提示（全部基于真实数据，绝不臆造） ====================
# 风险关键词表：命中真实新闻标题时提示。只做"关键词命中提醒"，不做情绪打分、不下结论。
RISK_KEYWORDS = {
    "退市风险": ["退市", "面值退", "终止上市", "*ST", "ST股", "强制退市"],
    "监管/立案风险": ["立案", "问询", "被查", "违规", "处罚", "风险警示", "关注函", "警示函", "调查", "诉讼", "仲裁"],
    "财务风险": ["亏损", "预亏", "商誉减值", "资不抵债", "债务逾期", "计提", "债务危机", "爆雷"],
    "股东/资金风险": ["减持", "质押", "爆仓", "冻结", "资金占用", "违规担保", "清仓"],
    "停牌": ["停牌", "停复牌"],
}


def assess_risks(code: str, name: str, df: pd.DataFrame,
                 news: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
    """
    基于真实数据的风险自动筛查(非投资建议、非预测)。每条提示都注明**真实依据来源**。
    返回 [{level, category, msg, source, url}]，level ∈ {高,中,低}。
    """
    warns: List[Dict[str, str]] = []
    close = pd.to_numeric(df["close"], errors="coerce").dropna() if "close" in df.columns else pd.Series([], dtype=float)
    lc = float(close.iloc[-1]) if len(close) else None

    # 1) ST/*ST 退市风险（依据：真实股票简称）
    up = (name or "").upper()
    if name and ("ST" in up):
        warns.append({"level": "高", "category": "退市风险",
                      "msg": f"股票简称为「{name}」，含 ST/*ST——已被交易所实施风险警示，存在退市风险，请高度谨慎。",
                      "source": "东方财富·股票简称", "url": ""})
    # 2) 面值/低价退市风险（依据：真实收盘价）
    if lc is not None:
        if lc < 1.0:
            warns.append({"level": "高", "category": "面值退市风险",
                          "msg": f"最新收盘 {lc:.2f} 元 < 1 元，若连续 20 个交易日低于面值将触及面值退市。",
                          "source": "东方财富·行情", "url": ""})
        elif lc < 2.0:
            warns.append({"level": "中", "category": "低价股风险",
                          "msg": f"最新收盘 {lc:.2f} 元，属低价股，波动与不确定性较大。",
                          "source": "东方财富·行情", "url": ""})
    # 3) 亏损（依据：真实估值 PE(TTM)）
    if "val_pe_ttm" in df.columns:
        pe_s = pd.to_numeric(df["val_pe_ttm"], errors="coerce").replace(0, np.nan).dropna()
        if len(pe_s) and pe_s.iloc[-1] <= 0:
            warns.append({"level": "中", "category": "盈利风险",
                          "msg": f"最新市盈率(TTM) {pe_s.iloc[-1]:.1f} ≤ 0，公司当前处于亏损状态。",
                          "source": "乐咕乐股/百度·估值", "url": ""})
    # 4) 近期暴跌（依据：真实价格）
    if len(close) > 21:
        r20 = lc / float(close.iloc[-21]) - 1
        if r20 < -0.25:
            warns.append({"level": "中", "category": "近期大跌",
                          "msg": f"近 1 个月跌幅 {r20*100:.1f}%，短期跌幅较大，注意风险。",
                          "source": "东方财富·行情", "url": ""})
    # 5) 真实新闻命中风险关键词（展示原文标题+链接，不下结论）
    if news is not None and len(news) > 0:
        for _, r in news.iterrows():
            title = str(r.get("title", "")); url = str(r.get("url", ""))
            for cat, kws in RISK_KEYWORDS.items():
                hit = next((k for k in kws if k in title), None)
                if hit:
                    warns.append({"level": "中", "category": f"新闻·{cat}",
                                  "msg": f"近期新闻命中「{hit}」：{title}",
                                  "source": "东方财富·新闻", "url": url})
                    break
    return warns


_TRADE_CAL_CACHE = {"dates": None, "source": None}   # 全会话缓存 A 股交易日历，避免重复请求


def _load_trade_calendar(progress_cb: Optional[Callable[[str], None]] = None):
    """拉取并缓存 A 股真实交易日历(含交易所已公布的、当年未来交易日；含节假日安排)。
    成功返回升序的 DatetimeIndex；失败返回 None(调用方自行退回工作日推算，绝不编造)。"""
    if _TRADE_CAL_CACHE["dates"] is not None:
        return _TRADE_CAL_CACHE["dates"]
    log = progress_cb or (lambda m: None)
    if not HAS_AKSHARE:
        return None
    try:
        with _no_proxy():
            cal = ak.tool_trade_date_hist_sina()          # 列 'trade_date'，覆盖到当年年底
        dates = pd.to_datetime(cal["trade_date"]).sort_values().reset_index(drop=True)
        idx = pd.DatetimeIndex(dates)
        _TRADE_CAL_CACHE["dates"] = idx
        _TRADE_CAL_CACHE["source"] = "akshare·tool_trade_date_hist_sina"
        log(f"[交易日历] 已加载 A 股真实交易日历，共 {len(idx)} 个交易日(覆盖到 {idx[-1].date()})。")
        return idx
    except Exception as e:
        log(f"[交易日历] 加载失败，退回工作日推算(节假日可能不准)：{e}")
        return None


def next_trading_days(last_date, n: int,
                      progress_cb: Optional[Callable[[str], None]] = None):
    """返回 last_date 之后的 n 个『真实交易日』(pd.Timestamp 列表)。
    优先用 akshare 真实日历(自动跳过周末+法定节假日)；日历缺失/不够时对超出部分退回工作日(BDay)推算。"""
    last = pd.Timestamp(last_date).normalize()
    cal = _load_trade_calendar(progress_cb)
    out = []
    if cal is not None:
        future = cal[cal > last]
        out = list(future[:n])
    # 日历没有(或未来交易日不足 n 个，多见于临近年底)时，其余用工作日近似补齐
    if len(out) < n:
        anchor = out[-1] if out else last
        h = 1
        while len(out) < n:
            out.append(anchor + pd.tseries.offsets.BDay(h)); h += 1
    return out[:n]


def forecast_curve(algo_name: str, raw_df: pd.DataFrame, target_mode: str = "return",
                   horizons=(1, 5, 10, 20, 40, 60), window: int = 20,
                   progress_cb: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    预测"未来走势"的诚实做法：对每个周期 h（1日/1周/2周/1月/2月/3月）**分别**训练一个
    "直接预测第 h 天"的模型(用全部历史)，各得到一个未来价格锚点。各锚点都是真实的直接预测，
    中间可插值成曲线——**不做递归预测**(拿预测喂预测会误差爆炸)，**不编造**未来的开高低量。

    返回：{algo, last_close, last_date, points:[{horizon, pred_close, change_pct, lo, hi, band_pct} ...]}
    其中 lo/hi 是约 80% 置信区间(基于历史日收益波动率 σ·√h)，提醒"点预测必错、区间才诚实"。
    """
    log = progress_cb or (lambda m: None)
    points, last_close, last_date = [], None, None
    # 历史日收益率波动率 σ(用于给预测加不确定区间；不是精确分布，只作量级参考)
    try:
        rr = pd.to_numeric(raw_df["close"], errors="coerce").pct_change().dropna()
        sigma_d = float(rr.std())
    except Exception:
        sigma_d = 0.0
    Z80 = 1.2816                                    # 80% 双侧置信的正态分位数
    for h in horizons:
        cfg = TrainConfig(window_size=window, horizon=h, target_mode=target_mode,
                          hpo_method="关闭", add_naive_baseline=False)
        pipe = TrainingPipeline(cfg)
        try:
            log(f"[未来预测] 训练「第 {h} 天」直接预测模型 ...")
            r = pipe.predict_next(algo_name, raw_df)
            band = Z80 * sigma_d * (h ** 0.5)       # 区间半宽(相对比例)，随周期 √h 放大
            pc = r["pred_close"]
            points.append({"horizon": h, "pred_close": pc, "change_pct": r["pred_change_pct"],
                           "lo": round(pc * (1 - band), 4), "hi": round(pc * (1 + band), 4),
                           "band_pct": round(band * 100, 2)})
            last_close, last_date = r["last_close"], r["last_date"]
        except Exception as e:
            points.append({"horizon": h, "error": str(e)})
    return {"algo": algo_name, "last_close": last_close, "last_date": last_date,
            "points": points, "sigma_daily_pct": round(sigma_d * 100, 3)}


# ==================== 第八部分补充4：批量扫描（多股批量 取数→训练→预测→风险） ====================
def batch_scan(codes: List[str], algo: str = "Lasso", start: str = "20200101",
               end: Optional[str] = None, target_mode: str = "return", horizon: int = 5,
               use_valuation: bool = True, use_index: bool = True, use_fundflow: bool = True,
               progress_cb: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """
    对一批股票逐只：取真实数据(+外部)→训练一个模型→测试集方向准确率(DA)→向前预测→风险筛查。
    返回每只一行的字典列表(可排序/导出)。**这是研究筛查工具，不是荐股**：请用风险列避开雷，
    别把"预测涨幅高/DA高"当买入信号——市场接近有效，单模型点预测不可靠。
    """
    log = progress_cb or (lambda m: None)
    end = end or dt.date.today().strftime("%Y%m%d")
    rows = []
    for i, code in enumerate(codes, 1):
        code = code.strip()
        if not code:
            continue
        log(f"[批量 {i}/{len(codes)}] {code} 取数+训练+预测 ...")
        try:
            df = StockDataFetcher().fetch(code, start, end)
            if use_valuation or use_index or use_fundflow:
                df, _ = StockDataFetcher.enrich(df, code, use_valuation, use_index, use_fundflow)
            name = StockDataFetcher.fetch_stock_name(code)
            cfg = TrainConfig(horizon=horizon, target_mode=target_mode, hpo_method="关闭",
                              metrics=["RMSE", "DA", "UP_P"])
            pipe = TrainingPipeline(cfg)
            res = pipe.run_batch([algo], df)
            mr = next((r for r in res if r.algo_name == algo and not r.error), None)
            da = mr.metrics.get("DA") if mr else None
            up_p = mr.metrics.get("UP_P") if mr else None
            fc = pipe.predict_next(algo, df)
            warns = assess_risks(code, name, df, news=None)
            hi_risk = [w for w in warns if w["level"] == "高"]
            rows.append({
                "code": code, "name": name, "last_close": fc["last_close"],
                "pred_close": fc["pred_close"], "pred_change_pct": fc["pred_change_pct"],
                "DA": da, "UP_P": up_p, "risk_n": len(warns),
                "risk_top": (hi_risk[0]["category"] if hi_risk else (warns[0]["category"] if warns else "")),
                "error": None,
            })
        except Exception as e:
            rows.append({"code": code, "name": "", "error": str(e)})
    return rows


def batch_factor_scan(codes: List[str], start: str = "20200101", end: Optional[str] = None,
                      w_value: float = 1.0, w_momentum: float = 1.0, w_money: float = 1.0,
                      w_quality: float = 1.0, w_lowvol: float = 1.0, use_quality: bool = True,
                      progress_cb: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """
    批量"因子打分选股"(横截面)：对一篮子股票，各算真实因子——
      价值(低PE/PB)、动量(近3月涨幅)、资金(近5日主力净流入)、
      质量/成长(ROE高、营收同比高更好)、低波动(近60日波动率低更好，低波动异象)，
      在这一篮子里做百分位排名后加权成综合分，并扣除风险(ST/亏损/暴跌)惩罚，最后排序。

    诚实说明：因子投资长期**统计上**有小优势(尤其价值+动量+质量)，但**不是保证、不是择时**，
    篮子越小排名越不稳；质量因子依赖财报接口，取不到就自动缺省不参与。本工具是研究筛查、**不是荐股**。
    """
    log = progress_cb or (lambda m: None)
    end = end or dt.date.today().strftime("%Y%m%d")
    recs = []
    for i, code in enumerate([c.strip() for c in codes if c.strip()], 1):
        log(f"[因子扫描 {i}/{len(codes)}] {code} 采集因子 ...")
        try:
            df = StockDataFetcher().fetch(code, start, end)
            df, _ = StockDataFetcher.enrich(df, code, True, False, True, False, False)  # 估值+资金
            name = StockDataFetcher.fetch_stock_name(code)
            close = pd.to_numeric(df["close"], errors="coerce")
            lc = float(close.iloc[-1])
            def last(col):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    return float(s.iloc[-1]) if len(s) else None
                return None
            mom1 = lc / float(close.iloc[-21]) - 1 if len(close) > 21 else None
            mom3 = lc / float(close.iloc[-63]) - 1 if len(close) > 63 else None
            mf5 = None
            if "mf_main_net" in df.columns:
                mf5 = float(pd.to_numeric(df["mf_main_net"], errors="coerce").fillna(0).tail(5).sum())
            # 低波动因子：近60日日收益年化波动率(越低越好)
            rr = close.pct_change().dropna()
            vol60 = float(rr.tail(60).std() * np.sqrt(252)) if len(rr) >= 20 else None
            # 质量/成长因子：ROE、营收同比(取不到即缺省，不编造)
            roe = rev_g = fscore = None
            if use_quality:
                q = StockDataFetcher.fetch_quality(code)
                roe, rev_g, fscore = q.get("roe"), q.get("rev_growth"), q.get("f_score")
            warns = assess_risks(code, name, df, news=None)
            recs.append({"code": code, "name": name, "last_close": round(lc, 2),
                         "pe": last("val_pe_ttm"), "pb": last("val_pb"),
                         "mom1m": mom1, "mom3m": mom3, "mf5": mf5,
                         "vol60": vol60, "roe": roe, "rev_growth": rev_g, "f_score": fscore,
                         "risk_n": len(warns), "risk_hi": sum(1 for w in warns if w["level"] == "高"),
                         "risk_top": (warns[0]["category"] if warns else ""), "error": None})
        except Exception as e:
            recs.append({"code": code, "name": "", "error": str(e)})

    ok = [r for r in recs if not r.get("error")]
    if ok:
        def pct(vals, higher_better=True):
            s = pd.Series(vals, dtype="float64")
            r = s.rank(pct=True, na_option="keep") * 100
            if not higher_better:
                r = 100 - r
            return r
        pe = [r["pe"] if (r["pe"] and r["pe"] > 0) else np.nan for r in ok]     # 亏损/无PE不参与价值分
        pb = [r["pb"] if (r["pb"] and r["pb"] > 0) else np.nan for r in ok]
        v_pe, v_pb = pct(pe, False), pct(pb, False)                            # PE/PB 越低越好
        s_val = pd.concat([v_pe, v_pb], axis=1).mean(axis=1)
        s_mom = pct([r["mom3m"] for r in ok], True)                            # 动量越高越好
        s_mon = pct([r["mf5"] for r in ok], True)                             # 资金流入越多越好
        # 质量：ROE + 营收增速 + Piotroski F-Score 三者百分位取平均(有几个算几个)；低波动：波动率越低越好
        s_roe = pct([r.get("roe") for r in ok], True)
        s_revg = pct([r.get("rev_growth") for r in ok], True)
        s_fsc = pct([r.get("f_score") for r in ok], True)
        s_qual = pd.concat([s_roe, s_revg, s_fsc], axis=1).mean(axis=1)
        s_lowvol = pct([r.get("vol60") for r in ok], False)                   # 波动越低分越高
        for k, r in enumerate(ok):
            comps, wts = [], []
            for sc, wt in [(s_val.iloc[k], w_value), (s_mom.iloc[k], w_momentum),
                           (s_mon.iloc[k], w_money), (s_qual.iloc[k], w_quality),
                           (s_lowvol.iloc[k], w_lowvol)]:
                if pd.notna(sc):
                    comps.append(sc * wt); wts.append(wt)
            base = (sum(comps) / sum(wts)) if wts else 0.0
            r["value_score"] = None if pd.isna(s_val.iloc[k]) else round(float(s_val.iloc[k]), 1)
            r["mom_score"] = None if pd.isna(s_mom.iloc[k]) else round(float(s_mom.iloc[k]), 1)
            r["money_score"] = None if pd.isna(s_mon.iloc[k]) else round(float(s_mon.iloc[k]), 1)
            r["qual_score"] = None if pd.isna(s_qual.iloc[k]) else round(float(s_qual.iloc[k]), 1)
            r["lowvol_score"] = None if pd.isna(s_lowvol.iloc[k]) else round(float(s_lowvol.iloc[k]), 1)
            r["score"] = round(float(base) - 12.0 * r["risk_hi"] - 3.0 * (r["risk_n"] - r["risk_hi"]), 1)
    recs.sort(key=lambda r: (r.get("score") if r.get("score") is not None else -1e9), reverse=True)
    return recs


# ==================== 第八部分补充5：预测跟踪（存预测→到期对比真实股价→算真实准确率） ====================
PRED_COLS = ["id", "made_at", "code", "name", "model", "horizon", "model_da", "base_date", "base_close",
             "pred_close", "pred_change_pct", "pred_dir", "target_date",
             "status", "actual_close", "actual_change_pct", "hit_dir", "verified_at"]


def load_pred_log() -> pd.DataFrame:
    if os.path.exists(PRED_LOG):
        try:
            return pd.read_csv(PRED_LOG, dtype=str).fillna("")
        except Exception:
            pass
    return pd.DataFrame(columns=PRED_COLS)


def save_predictions(rows: List[Dict[str, Any]]) -> int:
    """把若干条新预测追加进日志(去重：同 code+model+horizon+base_date 只留最新)。返回新增条数。"""
    df = load_pred_log()
    add = pd.DataFrame(rows)
    for c in PRED_COLS:
        if c not in add.columns:
            add[c] = ""
    df = pd.concat([df, add[PRED_COLS]], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "model", "horizon", "base_date"], keep="last")
    df.to_csv(PRED_LOG, index=False, encoding="utf-8-sig")
    return len(add)


def verify_predictions(progress_cb: Optional[Callable[[str], None]] = None) -> int:
    """对所有"到期(target_date<=今天)但未验证"的预测，拉真实股价对比，回填实际涨跌与是否命中方向。
    返回本次新验证的条数。全部基于真实历史行情，绝不臆造。"""
    log = progress_cb or (lambda m: None)
    df = load_pred_log()
    if len(df) == 0:
        return 0
    today = dt.date.today().strftime("%Y-%m-%d")
    pend = df[(df["status"] != "verified") & (df["target_date"] <= today) & (df["target_date"] != "")]
    if len(pend) == 0:
        return 0
    n_ok = 0
    for code in pend["code"].unique():
        try:
            end = dt.date.today().strftime("%Y%m%d")
            hist = StockDataFetcher().fetch(code, "20190101", end, use_cache=False)
            hist = hist[["date", "close"]].copy()
            hist["date"] = pd.to_datetime(hist["date"])
        except Exception as e:
            log(f"[验证] {code} 取真实行情失败: {e}")
            continue
        for idx in pend[pend["code"] == code].index:
            try:
                tgt = pd.to_datetime(df.at[idx, "target_date"])
                after = hist[hist["date"] >= tgt]
                if len(after) == 0:
                    continue                       # 目标日之后还没有真实数据
                actual = float(after["close"].iloc[0])
                base = float(df.at[idx, "base_close"])
                chg = (actual / base - 1) * 100
                pred_dir = df.at[idx, "pred_dir"]
                actual_dir = "涨" if actual >= base else "跌"
                # 注意：日志以 str 载入(pyarrow str 列拒绝写入 float)，回填一律转成字符串
                df.at[idx, "actual_close"] = str(round(actual, 4))
                df.at[idx, "actual_change_pct"] = str(round(chg, 3))
                df.at[idx, "hit_dir"] = "1" if pred_dir == actual_dir else "0"
                df.at[idx, "status"] = "verified"
                df.at[idx, "verified_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                n_ok += 1
            except Exception:
                continue
    df.to_csv(PRED_LOG, index=False, encoding="utf-8-sig")
    return n_ok


def prediction_accuracy() -> Dict[str, Dict[str, Any]]:
    """按预测周期统计**真实**方向准确率(只用已验证的记录)。返回 {周期标签: {n, dir_acc, mae_ret}}。"""
    df = load_pred_log()
    v = df[df["status"] == "verified"].copy()
    label = {"1": "每天(1日)", "3": "三天", "5": "每周(5日)", "20": "每月(20日)", "60": "每季(60日)"}
    out = {}
    if len(v) == 0:
        return out
    v["hit_dir"] = pd.to_numeric(v["hit_dir"], errors="coerce")
    v["ae"] = (pd.to_numeric(v["pred_change_pct"], errors="coerce") -
               pd.to_numeric(v["actual_change_pct"], errors="coerce")).abs()
    for hz, g in v.groupby("horizon"):
        out[label.get(str(hz), f"{hz}日")] = {
            "n": int(len(g)),
            "dir_acc": round(float(g["hit_dir"].mean() * 100), 1),
            "mae_ret": round(float(g["ae"].mean()), 2),
        }
    return out


# ==================== 第八部分补充6：盘口快照记录 + 挂单量变化分析 ====================
def write_snapshot(code: str, q: Dict[str, Any]) -> str:
    """把一次实时盘口快照追加写入当天 CSV(GUI 定时与 CLI --snapshot 共用)。返回文件路径。"""
    import csv
    day = dt.datetime.now().strftime("%Y%m%d")
    path = os.path.join(MONITOR_DIR, f"monitor_{code}_{day}.csv")
    is_new = not os.path.exists(path)
    row = {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "price": q.get("price"), "pct": q.get("pct"), "vol_ratio": q.get("vol_ratio"),
           "turnover": q.get("turnover")}
    for i, (p, v) in enumerate(q.get("bids", []), 1):
        row[f"buy{i}_p"], row[f"buy{i}_v"] = p, v
    for i, (p, v) in enumerate(q.get("asks", []), 1):
        row[f"sell{i}_p"], row[f"sell{i}_v"] = p, v
    with open(path, "a", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            wr.writeheader()
        wr.writerow(row)
    return path


def analyze_order_sincerity(code: str, day: Optional[str] = None) -> Dict[str, Any]:
    """
    挂单量变化分析(诚意粗判)：读当天记录的多次盘口快照，**按价位**对比首末两次的挂单量变化。
    大额挂单若显著减少→可能撤单/成交(免费数据无法区分)；变化不大→挂单意愿相对真实。

    诚实边界：只看得到"各价位聚合挂单量"，看不到逐笔委托，**无法区分撤单与成交**；要真正区分需付费
    Level-2 数据。故本分析仅是**粗略参考**，不是"主力真伪"的定论。
    """
    day = day or dt.datetime.now().strftime("%Y%m%d")
    path = os.path.join(MONITOR_DIR, f"monitor_{code}_{day}.csv")
    if not os.path.exists(path):
        return {"error": f"未找到当天快照文件 {os.path.basename(path)}；请先在交易时段记录几次盘口。"}
    df = pd.read_csv(path)
    if len(df) < 2:
        return {"error": f"当天只有 {len(df)} 条快照，至少需要 2 条才能作差对比。"}
    first, last = df.iloc[0], df.iloc[-1]

    def side_map(row, pref):
        m = {}
        for i in range(1, 6):
            p, v = row.get(f"{pref}{i}_p"), row.get(f"{pref}{i}_v")
            try:
                p = float(p); v = float(v)
                if p > 0:
                    m[round(p, 3)] = m.get(round(p, 3), 0.0) + v
            except (TypeError, ValueError):
                continue
        return m

    def compare(pref):
        a, b = side_map(first, pref), side_map(last, pref)
        rows, tot0, tot1 = [], 0.0, 0.0
        for price in sorted(set(a) | set(b), reverse=True):
            v0, v1 = a.get(price, 0.0), b.get(price, 0.0)
            tot0 += v0; tot1 += v1
            chg = (v1 - v0) / (v0 + 1e-9) * 100 if v0 > 0 else (100.0 if v1 > 0 else 0.0)
            rows.append({"price": price, "v0": v0, "v1": v1, "chg_pct": round(chg, 1)})
        keep = round(tot1 / (tot0 + 1e-9) * 100, 1) if tot0 > 0 else None
        return rows, round(tot0, 0), round(tot1, 0), keep

    buy_rows, b0, b1, buy_keep = compare("buy")
    sell_rows, s0, s1, sell_keep = compare("sell")
    return {
        "code": code, "day": day, "n_snap": int(len(df)),
        "t_first": str(first.get("time", "")), "t_last": str(last.get("time", "")),
        "buy": buy_rows, "sell": sell_rows,
        "buy_total0": b0, "buy_total1": b1, "buy_keep_pct": buy_keep,
        "sell_total0": s0, "sell_total1": s1, "sell_keep_pct": sell_keep,
    }


# ==================== 第八部分补充7：名词解释（大白话，给不懂金融/机器学习的用户） ====================
GLOSSARY = {
    "金融名词（看盘/选股）": [
        ("PE 市盈率(TTM)", "股价是公司「每股年利润」的多少倍。打比方：开个店，PE=20 就是按现在的赚钱速度约 20 年回本。一般越低越便宜；公司亏损时没有 PE。"),
        ("PB 市净率", "股价是公司「每股净资产(账面身家)」的多少倍。PB<1 表示比公司账面身家还便宜。"),
        ("PS 市销率", "股价对「每股营业收入」的倍数，常用于还没盈利的成长公司。"),
        ("总市值", "公司所有股票加起来值多少钱 = 股价 × 总股本。越大公司越大。"),
        ("换手率", "当天成交的股数占「流通股」的百分比。高=买卖很频繁/很活跃。"),
        ("量比", "当天成交量 ÷ 最近几天的平均成交量。>1 是放量(比平时活跃)，<1 是缩量。"),
        ("主力资金净流入", "当天「大单」(机构、大户)是净买入还是净卖出。正数=大资金在买，负数=在卖。"),
        ("北向资金", "通过沪深港通进来的外资(港/外资)，常被叫「聪明钱」，是外资对 A 股的态度。"),
        ("隔夜美股(纳斯达克)", "昨晚美股涨跌。美股和 A 股有些板块联动，昨夜美股大跌，A 股今天开盘可能承压。"),
        ("前复权", "把「除权除息」(送股/分红)造成的价格跳空抹平，让不同时间的股价能公平比较。"),
        ("ST / *ST", "被交易所「特别处理/风险警示」的股票，通常是连续亏损等，<b>有退市风险，很危险</b>。"),
        ("面值退市", "股价连续 20 个交易日低于 1 元，会被强制退市。"),
        ("涨跌幅 / 收益率", "(今天价格 − 昨天价格) ÷ 昨天价格，就是涨了或跌了百分之几。"),
        ("盘口 / 买卖五档", "当前排队等着买/卖的前 5 个价位和挂单量(买手/卖手)。"),
    ],
    "公司财务健康（财报名词·避雷用）": [
        ("亏损", "公司花的比赚的多，净利润是负的。<b>长期亏损很危险</b>，可能被 ST、甚至退市——软件的风险提示会标出来。"),
        ("盈利 / 净利润", "公司一年真正落袋的钱(收入减掉所有成本、费用、税)。正数=盈利，负数=亏损。"),
        ("营业收入(营收)", "公司卖产品/服务收到的总钱(还没减成本)。营收在增长通常是好事。"),
        ("扣非净利润", "剔除「卖房子/卖资产/政府补贴」等一次性收益后的<b>主业真实利润</b>。比净利润更实在。"),
        ("每股收益 EPS", "每一股股票分到多少利润。你手机上「中报每股收益 -0.89 元」就是每股亏 0.89 元。"),
        ("毛利率", "(营收 − 成本) ÷ 营收。反映产品本身赚不赚钱，越高越有竞争力。"),
        ("ROE 净资产收益率", "用股东的钱赚钱的效率，越高越好(巴菲特最看重的指标之一)。"),
        ("Piotroski F-Score", "把公司财务健康拆成 9 个客观判断(盈利/现金流/杠杆/效率)，每满足一条得 1 分。"
         "8~9 分很强、0~3 分弱。是**客观体检、不是预测**；本软件按可评项数如实给分(拿不到的项跳过)。"),
        ("相对强弱 RS", "个股近期涨幅 减去 大盘(沪深300)同期涨幅。为正=跑赢大盘(强)，为负=跑输(弱)。"
         "CANSLIM 选股讲究『买领涨、不买落后』，但强弱是历史事实、不保证未来。"),
        ("大盘状态(牛/熊/震荡)", "看沪深300 相对 200 日均线的位置与均线方向：上方且上行=偏多，下方且下行=偏空，其余=震荡。"
         "熊市里再好的个股信号也要打折——这是客观背景，不是买卖信号。"),
        ("资产负债率", "公司欠的钱占总资产的比例。太高(如 >70%)说明借钱多，有偿债/爆雷风险。"),
        ("现金流(经营)", "公司实际收进/付出的现金，比「利润」更难造假。经营现金流长期为负要警惕。"),
        ("商誉 / 商誉减值", "收购别人时多付的溢价叫商誉；若买来的公司变差，要「商誉减值」→ 可能一次性<b>巨亏</b>。"),
        ("分红 / 股息率", "公司把利润分给股东的钱 / 占股价的比例。常年稳定分红通常是好公司。"),
        ("净资产(账面价值)", "公司总资产减总负债，股东实际拥有的家底。PB 就是股价对每股净资产的倍数。"),
        ("中报/年报/季报", "半年/全年/每季度的财务成绩单。业绩公布前后股价常波动。"),
        ("业绩预亏 / 变脸 / 爆雷", "提前公告要亏 / 业绩突然大幅低于预期 / 突然暴露重大问题(财务造假、巨额计提等)，通常大跌。"),
    ],
    "模型与评估名词（判断准不准）": [
        ("DA 方向准确率", "预测「涨还是跌」猜对的比例。<b>50% = 抛硬币</b>，只有明显 >50% 才算真有本事。"),
        ("UP_P 上涨精确率", "模型说「会涨」的那些次里，真的涨了的比例。你只在它说涨时买，所以这个最实用。"),
        ("R² 决定系数", "模型解释了多少波动，越接近 1 越好。<b>但预测「价格」时会虚高到 0.9+，是假象，别被骗</b>——所以我们默认预测涨跌幅。"),
        ("MAE 平均绝对误差", "预测平均差多少(元或%)，越小越好。"),
        ("RMSE 均方根误差", "和 MAE 类似但对「大错」惩罚更重，越小越好。"),
        ("过拟合", "模型在「训练集」上很准、到「测试集」就拉胯——像死记硬背、不会举一反三。看训练误差远小于测试误差就是过拟合。"),
        ("训练 / 验证 / 测试集", "学习用 / 调参用 / 最终考试用，分开来防作弊(不能拿考题当练习)。"),
        ("策略回测", "用历史数据模拟「跟着模型预测买卖」到底能赚多少，还扣手续费。跑不赢「一直持有」就说明没用。"),
        ("夏普比率", "每承担一分风险换来多少收益，越高越好(>1 算不错)。"),
        ("最大回撤", "从最高点跌到最低点的最大跌幅，衡量「最坏能亏多少」。"),
        ("均线 MA5/10/20", "最近 5/10/20 天收盘价的平均线，看趋势。"),
        ("区间平均线(虚线)", "所选整段时间收盘价的平均，一条水平参考线。"),
        ("因子(价值/动量/资金/质量/低波动)", "选股打分的几个角度：价值=便不便宜(低PE/PB)，动量=近期涨势强不强，"
         "资金=大钱在不在买，质量=公司好不好(ROE高、营收增速快)，低波动=波动小(低波动异象，长期风险调整后常更优)。"
         "质量因子依赖财报接口，取不到就自动不参与、绝不编造。"),
    ],
}


# 算法简称 → 全称 + 大白话（用于勾选框悬停提示 & 名词解释）
ALGO_EXPLAIN = {
    "BP/ANN":     "BP 神经网络：最经典的多层神经网络，靠反向传播误差来拟合复杂关系。",
    "SVR":        "支持向量回归：找一条「容忍小误差」的最优曲线，样本少也稳，快，常用作基准。",
    "LSSVM":      "最小二乘支持向量机：SVR 的快速版(解线性方程组代替二次规划)。",
    "GPR":        "高斯过程回归：不但给预测值还给「不确定度」，小数据好用但较慢。",
    "ElasticNet": "弹性网：Lasso + Ridge 的结合，兼顾自动选特征和稳定。",
    "RidgeReg":   "岭回归：带 L2 惩罚的线性回归，防止系数过大、抗特征共线性。",
    "Lasso":      "套索回归：带 L1 惩罚的线性回归，能把没用的特征系数压成 0(自动选特征)，快且稳，常用作基准。",
    "PLSR":       "偏最小二乘回归：先把很多相关特征压缩成几个主成分再回归。",
    "KNN":        "K 近邻：找历史上最像的 K 天，取它们的平均来预测。",
    "Kstar":      "K*：基于熵距离的近邻法(本软件为近似实现)。",
    "ELM":        "极限学习机：单隐层神经网络，随机权重、训练极快。",
    "DTR":        "回归决策树：一层层 if-else 把数据切开做预测，直观但易过拟合。",
    "DT":         "决策树：同上一类的树模型，直观但单棵易过拟合。",
    "M5Rules":    "M5 规则：基于模型树的规则学习(本软件为近似实现)。",
    "RF":         "随机森林：很多棵决策树投票取平均，抗过拟合、稳，常用主力模型。",
    "Bagging":    "自助聚合：对数据反复抽样训练多个模型再平均，降低波动。",
    "ExtraTrees": "极端随机树：比随机森林切分更随机，更快、更抗过拟合。",
    "AdaBoost":   "自适应提升：一轮轮重点补做错的样本，串行叠加。",
    "GBRT":       "梯度提升回归树：一棵接一棵纠正上一棵的残差，精度高，常用主力模型。",
    "XGBoost":    "极致梯度提升：GBRT 的高效工程实现，比赛常胜(需装 xgboost)。",
    "LightGBM":   "微软高速梯度提升：大数据更快(需装 lightgbm)。",
    "CatBoost":   "Yandex 梯度提升：对类别特征友好、默认参数就不错(需装 catboost)。",
    "LSTM":       "长短期记忆网络：专门记时间序列的长期依赖，深度学习经典时序模型(需装 torch)。",
    "GRU":        "门控循环单元：LSTM 的精简版，更快(需装 torch)。",
    "Transformer":"自注意力网络：让每个时间点关注全序列的关键处(需装 torch)。",
    "TabNet":     "表格注意力网络：为表格数据设计(需装 pytorch-tabnet，缺则用 DNN 近似)。",
    "CNN":        "卷积神经网络：用滑动窗口捕捉近几日的价量局部形态(需装 torch)。",
    "DNN":        "深度全连接网络(多层感知机)：通用非线性拟合(需装 torch)。",
    "ResNet":     "残差网络：加跳跃连接让深层网络更好训练(需装 torch)。",
    "BPNet":      "BP 网络(神经网络的另一种实现，需装 torch)。",
    "RBFNet":     "径向基函数网络：用「离中心多远」做非线性映射(需装 torch)。",
    "SR":         "符号回归：直接进化出一个数学公式来拟合，可解释(需装 gplearn)。",
    "GEP":        "基因表达式编程：用进化算法搜数学公式(本软件为近似实现)。",
    "MEP":        "多表达式编程：同类的进化公式搜索(本软件为近似实现)。",
    "ARIMA":      "差分自回归移动平均：最经典的统计时序模型，只用价格自身的历史规律(需装 statsmodels)。",
}

# 评价指标简称 → 全称 + 大白话（用于勾选框悬停提示 & 名词解释）
METRIC_EXPLAIN = {
    "R2":   "R² 决定系数：模型解释了多少波动，越接近 1 越好。预测「价格」时会虚高到 0.9+ 是假象，故默认预测涨跌幅。",
    "MAE":  "平均绝对误差：预测平均差多少，越小越好。",
    "MSE":  "均方误差：误差平方后的平均，对「大错」惩罚更重，越小越好。",
    "RMSE": "均方根误差：MSE 开根号，单位与原数据一致，越小越好。",
    "MAPE": "平均绝对百分比误差：平均差百分之几，越小越好(真实值接近 0 时会失真)。",
    "SMAPE":"对称平均绝对百分比误差：MAPE 的改良版，越小越好。",
    "R":    "相关系数：预测与真实的线性相关程度，越接近 1 越好。",
    "NSE":  "纳什效率系数：1 为完美，0 相当于「只用均值预测」，越接近 1 越好。",
    "CE":   "效率系数(与 NSE 同类)：越接近 1 越好。",
    "KGE":  "Kling-Gupta 效率：综合相关性/偏差/变率的评分，越接近 1 越好。",
    "WI":   "Willmott 一致性指数：0~1，越接近 1 越好。",
    "SI":   "散度指数：RMSE ÷ 均值，越小越好。",
    "DA":   "方向准确率：预测「涨还是跌」猜对的比例。50% = 抛硬币，只有明显 >50% 才算真有本事。",
    "UP_P": "上涨精确率：模型说「会涨」的那些次里真的涨了的比例。你只在它说涨时买，所以这个最实用。",
}

# 把算法与指标简称并入名词解释总表（自动生成两个新板块）
GLOSSARY["算法简称（模型 = 35 种不同的预测方法）"] = [
    (name, ALGO_EXPLAIN[name]) for name in ALGO_EXPLAIN
]
GLOSSARY["评价指标简称（判断准不准的 14 个数）"] = [
    (name, METRIC_EXPLAIN[name]) for name in METRIC_EXPLAIN
]


def glossary_html() -> str:
    parts = ["<h2>名词解释（大白话）</h2>",
             "<p style='color:#888'>看不懂的名词在这里找。都是尽量通俗的解释，仅帮助理解，不构成投资建议。</p>"]
    for section, items in GLOSSARY.items():
        parts.append(f"<h3 style='color:#2c6fbb'>{section}</h3>"
                     "<table border=1 cellpadding=5 cellspacing=0 width=100%>")
        for term, expl in items:
            parts.append(f"<tr><td width=22% valign=top><b>{term}</b></td><td>{expl}</td></tr>")
        parts.append("</table>")
    return "".join(parts)


# ==================== 第八部分补充8：综合研判卡（只汇总客观事实，绝不下买卖结论/荐股） ====================
def market_regime(bypass_proxy: bool = True) -> Dict[str, Any]:
    """判断当前**大盘状态**(牛市/熊市/震荡)——借鉴 CANSLIM/宏观 regime 思路，只看客观价格结构：
    以沪深300 收盘 vs 200 日均线(MA200)及 MA200 斜率判断。取不到数据返回 unknown。
    这是给"择时"的客观背景参考，**不是买卖信号**：熊市里再好的个股信号也要打折。"""
    out = {"regime": "unknown", "text": "大盘状态未知(指数取数失败)", "idx_vs_ma200": None}
    if not HAS_AKSHARE:
        return out
    try:
        ix = StockDataFetcher.fetch_index_close(bypass_proxy)
        c = pd.to_numeric(ix["idx_px"], errors="coerce").dropna()
        if len(c) < 210:
            return out
        ma200 = c.rolling(200).mean()
        last, m_last, m_prev = float(c.iloc[-1]), float(ma200.iloc[-1]), float(ma200.iloc[-21])
        above = last > m_last
        rising = m_last > m_prev                       # MA200 向上=多头结构
        dev = (last / m_last - 1) * 100
        out["idx_vs_ma200"] = round(dev, 1)
        if above and rising:
            out["regime"], out["text"] = "bull", f"大盘偏多(沪深300在200日线上{dev:+.1f}%、均线上行)"
        elif (not above) and (not rising):
            out["regime"], out["text"] = "bear", f"大盘偏空(沪深300在200日线下{dev:+.1f}%、均线下行)——个股信号需大打折扣"
        else:
            out["regime"], out["text"] = "range", f"大盘震荡/转折(沪深300与200日线{dev:+.1f}%、方向不明)"
    except Exception:
        pass
    return out


def da_significance(da_pct: Optional[float], n: int) -> Dict[str, Any]:
    """判断方向准确率 DA 是否**显著**高于 50%(抛硬币)。用单侧二项检验的正态近似算 p 值。
    返回 {p, sig, text}：p 越小越显著；sig=True 表示 p<0.05 可认为真比蒙的强。
    诚实要点：样本越少(n 小)，哪怕 DA=60% 也可能只是运气，这里如实标注。"""
    if da_pct is None or n is None or n < 5:
        return {"p": None, "sig": False, "text": "样本太少，无法判断显著性"}
    import math
    phat = da_pct / 100.0
    se = math.sqrt(0.25 / n)                       # H0: p=0.5 的标准误
    z = (phat - 0.5) / (se + 1e-12)
    # 单侧正态尾概率 P(Z>z)（用 erfc，无需 scipy）
    p = 0.5 * math.erfc(z / math.sqrt(2))
    if phat <= 0.5:
        return {"p": round(p, 4), "sig": False, "text": f"≤50%(n={n})，不比抛硬币强"}
    if p < 0.01:
        txt = f"显著>50%(p={p:.3f}, n={n})"
    elif p < 0.05:
        txt = f"较显著>50%(p={p:.3f}, n={n})"
    else:
        txt = f"不显著(p={p:.2f}, n={n})，可能只是运气"
    return {"p": round(p, 4), "sig": p < 0.05, "text": txt}


def research_card(code: str, start: str = "20200101", end: Optional[str] = None,
                  target_mode: str = "return", horizon: int = 5,
                  models=("SVR", "Lasso"),
                  progress_cb: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    把一只股票的**客观信号**汇总成一张"研判卡"：风险、模型方向准确率(DA)对比基准、估值、
    主力资金、大盘/外盘、近期涨跌。**只陈列事实，不给买卖建议、不荐股**——判断留给用户。
    """
    log = progress_cb or (lambda m: None)
    end = end or dt.date.today().strftime("%Y%m%d")
    log(f"[研判卡] {code} 取数+评估 ...")
    df = StockDataFetcher().fetch(code, start, end)
    df, status = StockDataFetcher.enrich(df, code, True, True, True, True, True)
    name = StockDataFetcher.fetch_stock_name(code)
    close = pd.to_numeric(df["close"], errors="coerce")
    lc = float(close.iloc[-1])

    def last(col):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else None
        return None

    def chg(w):
        return (lc / float(close.iloc[-1 - w]) - 1) * 100 if len(close) > w else None

    # 各模型在该周期的测试 DA + "总是涨"基准
    da_rows, up_rate = [], None
    for m in models:
        try:
            cfg = TrainConfig(horizon=horizon, target_mode=target_mode, hpo_method="关闭",
                              metrics=["DA", "UP_P"])
            res = TrainingPipeline(cfg).run_batch([m], df)
            base = next((r for r in res if r.algo_name == "总是涨(方向基准)"), None)
            if base and base.metrics:
                up_rate = base.metrics.get("DA")
            mr = next((r for r in res if r.algo_name == m and not r.error), None)
            # 模型对未来该周期的"机械倾向"：预测涨跌幅的方向(不是建议，需配可信度看)
            pred_chg = None
            try:
                fc = TrainingPipeline(cfg).predict_next(m, df)
                pred_chg = fc.get("pred_change_pct")
            except Exception:
                pass
            da_v = mr.metrics.get("DA") if mr else None
            n_te = int(len(mr.y_test_true)) if (mr and mr.y_test_true is not None) else 0
            sig = da_significance(da_v, n_te)     # DA 是否显著>50%(二项检验)
            da_rows.append({"model": m, "DA": da_v,
                            "UP_P": (mr.metrics.get("UP_P") if mr else None),
                            "pred_change": pred_chg, "n_test": n_te,
                            "da_p": sig["p"], "da_sig": sig["sig"], "da_sig_text": sig["text"]})
        except Exception as e:
            da_rows.append({"model": m, "error": str(e)})
    warns = assess_risks(code, name, df, news=None)
    # 借鉴交易 skill：基本面质量(Piotroski F-Score) + 相对大盘强弱(CANSLIM RS) + 大盘状态(regime)
    q = StockDataFetcher.fetch_quality(code)
    ret_1q = chg(60)
    rs_1q = None
    reg = market_regime()
    try:
        ixc = pd.to_numeric(StockDataFetcher.fetch_index_close()["idx_px"], errors="coerce").dropna()
        if len(ixc) > 63 and ret_1q is not None:
            idx_ret_1q = (float(ixc.iloc[-1]) / float(ixc.iloc[-64]) - 1) * 100
            rs_1q = ret_1q - idx_ret_1q                # 相对强弱：个股近3月涨幅 − 沪深300近3月涨幅
    except Exception:
        pass
    return {
        "code": code, "name": name, "last_close": round(lc, 2), "horizon": horizon,
        "ret_1w": chg(5), "ret_1m": chg(20), "ret_1q": ret_1q,
        "pe": last("val_pe_ttm"), "pb": last("val_pb"),
        "mf5": (float(pd.to_numeric(df["mf_main_net"], errors="coerce").fillna(0).tail(5).sum())
                if "mf_main_net" in df.columns else None),
        "idx_ret": last("idx_ret_1d"), "us_ret": last("us_ret"), "nb_net": last("nb_net"),
        "f_score": q.get("f_score"), "f_available": q.get("f_available"), "f_detail": q.get("f_detail"),
        "roe": q.get("roe"), "rs_1q": rs_1q, "regime": reg,
        "up_rate": up_rate, "da_rows": da_rows, "warns": warns, "status": status,
    }


def research_card_html(card: Dict[str, Any]) -> str:
    """把研判卡渲染成 HTML。全程只陈述客观事实 + 一句客观判读，无买卖结论。"""
    def f(v, suf="", nd=2):
        return f"{v:.{nd}f}{suf}" if isinstance(v, (int, float)) else "-"
    title = f"{card['name']}（{card['code']}）" if card.get("name") else card["code"]
    # 风险
    if card["warns"]:
        ri = "".join(f"<li><b style='color:#c0392b'>[{w['level']}·{w['category']}]</b> {w['msg']}"
                     f"<span style='color:#888'>（{w['source']}）</span></li>" for w in card["warns"])
        risk = (f"<div style='background:#fff4f4;border:1px solid #f0b0b0;padding:6px'>"
                f"<b style='color:#c0392b'>⚠️ 风险提示（{len(card['warns'])}条）</b><ul>{ri}</ul></div>")
    else:
        risk = ("<div style='background:#f2fbf2;border:1px solid #b7e0b7;padding:6px'>"
                "<b style='color:#1e8449'>✅ 未发现明显风险信号</b>（自动筛查，不代表无风险）</div>")
    # 模型 DA 对比基准
    up = card.get("up_rate")
    da_body = ""
    for r in card["da_rows"]:
        if r.get("error"):
            da_body += f"<tr><td>{r['model']}</td><td colspan=3>失败</td></tr>"; continue
        da = r.get("DA")
        good = (up is not None and da is not None and da >= up + 3 and da >= 52)
        tag = ("<span style='color:#1e8449'>高于基准</span>" if good
               else "<span style='color:#c0392b'>≈基准/偏弱</span>")
        sig_txt = r.get("da_sig_text", "")
        da_body += (f"<tr><td>{r['model']}</td><td>{f(da,'%',1)}</td>"
                    f"<td>{f(r.get('UP_P'),'%',1)}</td><td>{tag}<br><span style='color:#888;font-size:12px'>{sig_txt}</span></td></tr>")
    # 模型「机械倾向」+ 可信度（回答"模型对后市的看法"，但绝不当建议：可信度低就明说≈抛硬币）
    lean_rows = ""
    for r in card["da_rows"]:
        if r.get("error"):
            continue
        pc, da = r.get("pred_change"), r.get("DA")
        if pc is None:
            lean = "-"
        else:
            lean = ("<span style='color:#c0392b'>偏涨</span>" if pc > 0.2 else
                    ("<span style='color:#1a9d5a'>偏跌</span>" if pc < -0.2 else "中性"))
        # 可信度须同时满足：①高于基准 ②统计上显著>50%(不是运气)。任一不满足即判为不可信
        reliable = (up is not None and da is not None and da >= up + 3 and da >= 52
                    and r.get("da_sig"))
        cred = ("<span style='color:#1e8449'>可信度尚可(仍非建议)</span>" if reliable
                else "<span style='color:#c0392b'>≈抛硬币，基本不可信</span>")
        lean_rows += (f"<tr><td>{r['model']}</td><td>{lean}</td>"
                      f"<td>{f(pc,'%',2)}</td>"
                      f"<td>DA {f(da,'%',1)} → {cred}<br>"
                      f"<span style='color:#888;font-size:12px'>{r.get('da_sig_text','')}</span></td></tr>")
    lean_html = (f"<h3>模型对后市的「机械倾向」（{card['horizon']}日，<b style='color:#c0392b'>非预言、非建议</b>）</h3>"
                 "<p style='color:#888'>下面是模型按数据算出的方向倾向，但<b>必须连着右边的可信度一起看</b>："
                 "DA≈50% 时这个倾向就等于抛硬币，别当真。</p>"
                 "<table border=1 cellpadding=4 cellspacing=0 width=100%>"
                 "<tr bgcolor=#eef><th>模型</th><th>倾向</th><th>预测涨跌</th><th>可信度</th></tr>"
                 + lean_rows + "</table>")
    # 客观判读(描述性，非建议)
    reads = []
    if card["pe"] is not None:
        reads.append("财务亏损(无有效市盈率)" if card["pe"] <= 0 else f"市盈率{f(card['pe'],'',1)}")
    if card["ret_1m"] is not None:
        reads.append(f"近1月{f(card['ret_1m'],'%',1)}")
    if card["mf5"] is not None:
        reads.append(f"主力近5日净{'流入' if card['mf5']>=0 else '流出'}{abs(card['mf5'])/1e4:.0f}万")
    # 基本面质量 F-Score / 相对强弱 RS / 大盘状态
    fs, fa = card.get("f_score"), card.get("f_available")
    if fs is not None and fa:
        lvl = "强" if fs >= max(6, fa - 1) else ("弱" if fs <= 2 else "中")
        reads.append(f"基本面质量 F-Score {fs}/{fa}({lvl})")
    if card.get("rs_1q") is not None:
        reads.append(f"相对大盘{'强' if card['rs_1q']>=0 else '弱'}{card['rs_1q']:+.1f}%(近3月超额)")
    if card.get("regime", {}).get("regime", "unknown") != "unknown":
        reads.append(card["regime"]["text"])
    best_da = max((r.get("DA") or 0) for r in card["da_rows"]) if card["da_rows"] else 0
    reads.append(f"模型方向准确率最高{f(best_da,'%',1)}"
                 + (f"(基准{f(up,'%',1)})" if up is not None else "")
                 + ("，有一定判别力" if (up and best_da >= up + 3) else "，≈抛硬币、判别力弱"))
    ctx = ("<table border=1 cellpadding=4 cellspacing=0 width=100%>"
           f"<tr><td>最新收盘</td><td>{f(card['last_close'])} 元</td>"
           f"<td>近1周/1月/1季</td><td>{f(card['ret_1w'],'%',1)} / {f(card['ret_1m'],'%',1)} / {f(card['ret_1q'],'%',1)}</td></tr>"
           f"<tr><td>市盈率/市净率</td><td>{f(card['pe'],'',1)} / {f(card['pb'],'',2)}</td>"
           f"<td>主力近5日净流入</td><td>{(f(card['mf5']/1e4,'万',0) if card['mf5'] is not None else '-')}</td></tr>"
           f"<tr><td>沪深300最新</td><td>{f((card['idx_ret'] or 0)*100,'%',2) if card['idx_ret'] is not None else '-'}</td>"
           f"<td>隔夜纳指/北向</td><td>{f((card['us_ret'] or 0)*100,'%',2) if card['us_ret'] is not None else '-'}"
           f" / {f(card['nb_net'],'',1) if card['nb_net'] is not None else '-'}</td></tr>"
           f"<tr><td>基本面质量 F-Score</td><td>{(str(card.get('f_score'))+'/'+str(card.get('f_available')) if card.get('f_score') is not None else '-')}</td>"
           f"<td>相对大盘(近3月超额)</td><td>{f(card.get('rs_1q'),'%',1)}</td></tr>"
           f"<tr><td>大盘状态</td><td colspan=3>{card.get('regime',{}).get('text','-')}</td></tr></table>")
    return (
        f"<h2>综合研判卡 · {title}</h2>"
        "<p style='color:#c0392b'><b>本卡只汇总客观事实，不构成任何买卖建议、不荐股；请自行判断。</b></p>"
        + risk +
        f"<h3>模型方向准确率（{card['horizon']}日，测试集真实表现）</h3>"
        "<table border=1 cellpadding=4 cellspacing=0 width=100%>"
        "<tr bgcolor=#eef><th>模型</th><th>DA方向准确率</th><th>UP_P上涨精确率</th><th>对比基准</th></tr>"
        + da_body + "</table>"
        + lean_html +
        "<h3>关键数据现状（真实来源，见数据溯源）</h3>" + ctx +
        "<h3>客观判读（描述事实，非建议）</h3><p>" + "；".join(reads) + "。</p>"
        "<p style='color:#888;font-size:12px'>提示：市场接近有效，单模型方向预测大多≈50%；"
        "以上为客观体检，买不买、仓位多少由你决定，风险自负。</p>")


# ==================== 第九部分：可视化 GUI 层（PySide6） ====================
# 界面布局参考截图"一键式科研软件 MIMO 多输入多输出研究平台"的设计思路：
#   左侧：数据配置（股票代码/日期区间/数据源）
#   中部：B-1.1 多算法勾选（按 3 个分类分组，可多选）+ B-1.2 功能选择（HPO/CV）
#         + B-1.3 评价指标勾选 + B-1.4 训练测试比例选择
#   右侧/下方：预测结果对比图 + 指标结果表格 + 运行日志
if HAS_PYSIDE6:

    # ---------- 9.1 后台训练线程：避免长时间训练把界面卡死 ----------
    class TrainingWorker(QThread):
        progress_signal = Signal(str)
        finished_signal = Signal(list)          # List[ModelResult]
        error_signal = Signal(str)

        def __init__(self, config: TrainConfig, raw_df: pd.DataFrame, algo_names: List[str]):
            super().__init__()
            self.config = config
            self.raw_df = raw_df
            self.algo_names = algo_names

        def run(self):
            try:
                pipeline = TrainingPipeline(self.config)
                results = pipeline.run_batch(self.algo_names, self.raw_df,
                                              progress_cb=lambda msg: self.progress_signal.emit(msg))
                self.finished_signal.emit(results)
            except Exception as e:
                self.error_signal.emit(str(e))

    class BatchWorker(QThread):
        """批量扫描后台线程：对一批股票逐只 取数→训练→预测→风险。"""
        progress_signal = Signal(str)
        finished_signal = Signal(list)          # List[dict]
        error_signal = Signal(str)

        def __init__(self, codes, algo, start, end, target_mode, horizon, flags):
            super().__init__()
            # 注意：不能叫 self.start / self.end，会覆盖 QThread.start() 方法导致无法启动线程
            self.codes = codes; self.algo = algo; self.d_start = start; self.d_end = end
            self.target_mode = target_mode; self.horizon = horizon; self.flags = flags

        def run(self):
            try:
                rows = batch_scan(self.codes, algo=self.algo, start=self.d_start, end=self.d_end,
                                  target_mode=self.target_mode, horizon=self.horizon,
                                  use_valuation=self.flags[0], use_index=self.flags[1],
                                  use_fundflow=self.flags[2],
                                  progress_cb=lambda m: self.progress_signal.emit(m))
                self.finished_signal.emit(rows)
            except Exception as e:
                self.error_signal.emit(str(e))

    class FactorWorker(QThread):
        """批量因子打分选股后台线程。"""
        progress_signal = Signal(str)
        finished_signal = Signal(list)
        error_signal = Signal(str)

        def __init__(self, codes, start, end):
            super().__init__()
            # 同上：避免 self.start 覆盖 QThread.start()
            self.codes = codes; self.d_start = start; self.d_end = end

        def run(self):
            try:
                rows = batch_factor_scan(self.codes, start=self.d_start, end=self.d_end,
                                         progress_cb=lambda m: self.progress_signal.emit(m))
                self.finished_signal.emit(rows)
            except Exception as e:
                self.error_signal.emit(str(e))

    # ---------- 9.2 主窗口 ----------
    class MainWindow(QMainWindow):

        def __init__(self):
            super().__init__()
            self.setWindowTitle("一键式 A 股预测研究软件  |  stock_predictor.py")
            self.resize(1500, 950)

            self.raw_df: Optional[pd.DataFrame] = None
            self.results: List[ModelResult] = []
            self.algo_checkboxes: Dict[str, QCheckBox] = {}
            self.metric_checkboxes: Dict[str, QCheckBox] = {}

            self._build_ui()
            self._oplog("软件启动。已就绪。")
            # 启动时自动验证：把之前记录、目标日期已到的预测，自动拉真实股价对比(哪怕你早忘了这只票)
            try:
                n = verify_predictions()
                if n > 0:
                    self._oplog(f"启动自动验证：{n} 条到期预测已对比真实股价（见「预测跟踪」页）。")
            except Exception:
                pass
            self._show_disclaimer()

        def _show_disclaimer(self):
            box = QMessageBox(self)
            box.setWindowTitle("使用须知 · 合规与免责")
            box.setIcon(QMessageBox.Warning)
            box.setText("<b style='color:#c0392b'>⚠️ 请先阅读（合规红线）</b>")
            box.setInformativeText(
                "<p><b>1. 仅供学术研究与学习，本软件不构成任何投资建议。</b>市场接近有效，"
                "任何模型的历史表现都不代表未来收益，据此交易风险自负。</p>"
                "<p><b>2. 数据来自公开接口（东财/baostock/百度），请仅作个人研究：</b></p>"
                "<ul>"
                "<li>✅ 自己拉公开数据、自己研究、自己判断 —— 基本安全</li>"
                "<li>❌ 高频轰炸接口（请保持低频，如监控 5~10 秒）</li>"
                "<li>❌ 把抓取的原始数据打包<b>售卖 / 公开再分发</b></li>"
                "<li>❌ 无证券投资咨询牌照<b>给他人荐股 / 收费</b>（属违规）</li>"
                "<li>❌ 大规模对外<b>转发实时行情</b></li>"
                "</ul>"
                "<p>做成对外产品/服务前，请先咨询证券合规专业律师。</p>")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec()

        # ---- 9.2.1 整体布局搭建 ----
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)

            # 顶部：常驻醒目合规红线条（始终可见）
            bar = QLabel("⚠️ 仅供学术研究，不构成任何投资建议 ｜ 请自用·勿高频抓取·勿无牌荐股或售卖数据 ｜ "
                         "据此交易风险自负")
            bar.setStyleSheet("background:#c0392b; color:white; font-weight:bold; padding:5px;")
            bar.setAlignment(Qt.AlignCenter)
            outer.addWidget(bar)

            content = QWidget()
            main_layout = QHBoxLayout(content)
            outer.addWidget(content, stretch=1)

            # 左：配置区（滚动，防止算法太多超出屏幕）
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(560)
            config_panel = QWidget()
            config_layout = QVBoxLayout(config_panel)
            config_layout.addWidget(self._build_data_group())
            config_layout.addWidget(self._build_external_group())
            config_layout.addWidget(self._build_algo_group())
            config_layout.addWidget(self._build_function_group())
            config_layout.addWidget(self._build_metric_group())
            config_layout.addWidget(self._build_split_group())
            config_layout.addWidget(self._build_run_group())
            config_layout.addStretch()
            scroll.setWidget(config_panel)
            main_layout.addWidget(scroll)

            # 右：结果展示区（Tab：行情K线 / 预测图 / 表格 / 日志）
            self.tabs = QTabWidget()

            # 第 1 个标签页：真实行情 K 线图（点按钮即可先预览数据，不必先跑模型）
            self.tabs.addTab(self._build_kline_tab(), "行情K线图")

            self.figure = Figure(figsize=(8, 5))
            self.canvas = FigureCanvas(self.figure)
            self.tabs.addTab(self.canvas, "预测结果对比图")

            # 未来预测走势图（多周期直接预测：1日/1周/1月/3月），可缩放看每日价格
            self.tabs.addTab(self._build_forecast_tab(), "未来预测图")

            result_panel = QWidget()
            rp_layout = QVBoxLayout(result_panel)
            self.reco_view = QTextBrowser()          # 顶部：模型推荐 + 预测依据
            self.reco_view.setOpenExternalLinks(True)
            self.reco_view.setMaximumHeight(260)
            self.reco_view.setHtml("<p style='color:#888'>运行模型后，这里给出<b>基于真实DA的模型推荐</b>与<b>预测依据</b>。</p>")
            rp_layout.addWidget(self.reco_view)
            self.result_table = QTableWidget()
            rp_layout.addWidget(self.result_table, stretch=1)
            self.tabs.addTab(result_panel, "指标结果表格")

            # 策略回测：按预测方向做多/空仓，扣手续费，和买入持有对比
            self.tabs.addTab(self._build_backtest_tab(), "策略回测")

            self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
            self.tabs.addTab(self.log_box, "运行日志")

            # 操作日志：记录用户的每一步操作（点了什么、选了什么、拉了什么数据），带时间戳
            self.oplog_box = QTextEdit(); self.oplog_box.setReadOnly(True)
            self.tabs.addTab(self.oplog_box, "操作日志")

            # 机器学习内部（真实趋势 + 输入来源 + 训练/测试数据集，透明可查）
            self.tabs.addTab(self._build_mldata_tab(), "机器学习内部")

            # 综合报告：把前面所有图表+数据汇总，可预览、可导出自包含 HTML 报告
            self.tabs.addTab(self._build_report_tab(), "综合报告")

            # 实时监控：交易时段定时刷新真实盘口 + 预测参考
            self.tabs.addTab(self._build_monitor_tab(), "实时监控")

            # 预测跟踪：保存每次预测，等日期到了对比真实股价，算真实准确率
            self.tabs.addTab(self._build_tracking_tab(), "预测跟踪")

            # 批量扫描：多股批量 取数→训练→预测→风险
            self.tabs.addTab(self._build_batch_tab(), "批量扫描")

            main_layout.addWidget(self.tabs, stretch=1)

        # ---- 9.2.1b 行情 K 线图标签页 ----
        def _build_kline_tab(self) -> QWidget:
            """
            一个独立的 K 线预览页：点"获取/刷新 K 线图"按钮，就按上方数据配置里的
            股票代码/日期/数据源拉一次行情并画出蜡烛图(含 MA5/10/20 与成交量)，
            方便先"看下情况"确认数据没问题，再去跑模型。
            """
            panel = QWidget()
            layout = QVBoxLayout(panel)

            top = QHBoxLayout()
            top.addWidget(QLabel("周期:"))
            self.kline_period = QComboBox()
            # (显示, 重采样规则, 回看根数, 说明)
            self._kline_periods = [("日K·近3月", "D", 63), ("周K·近1.5年", "W-FRI", 78),
                                   ("月K·近6年", "ME", 72)]
            self.kline_period.addItems([t for t, _, _ in self._kline_periods])
            self.kline_period.currentIndexChanged.connect(lambda _: self._on_fetch_kline())
            top.addWidget(self.kline_period)
            self.kline_btn = QPushButton("📈 获取/刷新 K 线图")
            self.kline_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.kline_btn.clicked.connect(self._on_fetch_kline)
            top.addWidget(self.kline_btn)
            self.kline_hint = QLabel("红涨绿跌；虚线=区间平均线；图下方工具条可放大缩小。")
            self.kline_hint.setStyleSheet("color:#666;")
            top.addWidget(self.kline_hint, stretch=1)
            layout.addLayout(top)

            self.kline_figure = Figure(figsize=(8, 5))
            self.kline_canvas = FigureCanvas(self.kline_figure)
            layout.addWidget(NavigationToolbar(self.kline_canvas, panel))    # 缩放/平移工具条
            layout.addWidget(self.kline_canvas, stretch=1)
            return panel

        # ---- 9.2.1c 点击"获取 K 线"后：拉数据并画图 ----
        def _on_fetch_kline(self):
            self.kline_btn.setEnabled(False); self.kline_hint.setText("⏳ 正在拉取行情并绘制 K 线 ...")
            QApplication.setOverrideCursor(Qt.WaitCursor); self._prog_open(); QApplication.processEvents()
            try:
                use_synthetic = self.data_source_combo.currentIndex() == 1
                if use_synthetic:
                    df = StockDataFetcher.generate_synthetic_data()
                    title = "合成数据 K 线（离线演示）"
                else:
                    code = self.code_edit.text().strip()
                    start = self.start_date.date().toString("yyyyMMdd")
                    end = self.end_date.date().toString("yyyyMMdd")
                    self.kline_hint.setText(f"正在拉取 {code} 的行情 ...")
                    QApplication.processEvents()
                    df = StockDataFetcher().fetch(code, start, end)   # 已自动绕过系统代理
                    title = f"{code}  日线 K 线  [{start} ~ {end}]"
                if df is None or len(df) == 0:
                    QMessageBox.warning(self, "提示", "没有取到任何行情数据，请检查代码/日期。")
                    return
                # 按所选周期重采样(日/周/月)并只取对应回看长度
                pname, rule, lookback = self._kline_periods[self.kline_period.currentIndex()]
                kdf = self._resample_kline(df, rule).tail(lookback).reset_index(drop=True)
                title = (f"{'合成数据' if use_synthetic else code}  {pname}"
                         f"（共 {len(kdf)} 根，虚线=区间平均{kdf['close'].mean():.2f}）")
                self._draw_kline(kdf, title, avg=float(kdf["close"].mean()))
                self.kline_hint.setText(f"{pname}：{len(kdf)} 根K线，红涨绿跌 + MA + 区间平均线(虚线)，工具条可缩放。")
                self._oplog(f"获取K线：{code if not use_synthetic else '合成数据'} {pname}，{len(kdf)} 根。")
            except Exception as e:
                QMessageBox.critical(self, "K 线获取失败", str(e))
                self.kline_hint.setText("获取失败，详见弹窗。若是代理/网络问题，可先用合成数据。")
            finally:
                QApplication.restoreOverrideCursor(); self._prog_close(); self.kline_btn.setEnabled(True)

        # ---- 9.2.1d 画蜡烛图（价格 + 均线 + 成交量）----
        @staticmethod
        def _resample_kline(df, rule):
            """日线重采样为 日(D)/周(W-FRI)/月(ME) K线。"""
            d = df[["date", "open", "high", "low", "close", "volume"]].copy()
            d["date"] = pd.to_datetime(d["date"])
            if rule == "D":
                return d.reset_index(drop=True)
            r = d.resample(rule, on="date").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last",
                 "volume": "sum"}).dropna().reset_index()
            return r

        def _draw_kline(self, df: pd.DataFrame, title: str, avg=None, fig=None):
            fig = fig or self.kline_figure          # 传入 fig 时画到该图(供报告离屏生成)，否则画到屏幕K线图
            fig.clear()
            gs = fig.add_gridspec(4, 1, hspace=0.08)
            ax_p = fig.add_subplot(gs[0:3, 0])
            ax_v = fig.add_subplot(gs[3, 0], sharex=ax_p)

            x = np.arange(len(df))
            o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
            h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
            up = c >= o
            red, green = "#e0334b", "#1a9d5a"          # A 股习惯：红涨绿跌
            colors = np.where(up, red, green)

            # 影线(最高-最低) + 实体(开盘-收盘)
            ax_p.vlines(x, l, h, color=colors, linewidth=0.8, zorder=2)
            body = np.where(np.abs(c - o) < 1e-9, 1e-9, c - o)   # 十字星给个极小高度以可见
            ax_p.bar(x, body, bottom=o, width=0.62, color=colors, edgecolor=colors, zorder=3)

            # 均线
            close_s = pd.Series(c)
            for w, col in [(5, "#f2a900"), (10, "#3b7dd8"), (20, "#9270CA")]:
                if len(df) >= w:
                    ax_p.plot(x, close_s.rolling(w).mean(), linewidth=1.0, label=f"MA{w}", color=col)
            # 区间平均线（水平虚线：整段所选周期的平均收盘价）
            if avg is not None:
                ax_p.axhline(avg, color="#333", linestyle="--", linewidth=1.2, label=f"区间平均 {avg:.2f}")
            ax_p.legend(loc="best", fontsize=8)
            ax_p.set_ylabel("价格")
            ax_p.set_title(title)
            ax_p.grid(True, alpha=0.25)

            # 成交量
            if "volume" in df.columns:
                ax_v.bar(x, df["volume"].to_numpy(float), width=0.62, color=colors)
            ax_v.set_ylabel("成交量")
            ax_v.grid(True, alpha=0.25)

            # x 轴用日期标签（挑约 8 个刻度，避免拥挤）
            n = len(df)
            idx = np.linspace(0, n - 1, min(8, n)).astype(int)
            ax_v.set_xticks(x[idx])
            ax_v.set_xticklabels(
                [pd.to_datetime(df["date"].iloc[i]).strftime("%y-%m-%d") for i in idx],
                rotation=30, fontsize=8)
            plt.setp(ax_p.get_xticklabels(), visible=False)
            if fig is self.kline_figure:
                self.kline_canvas.draw()

        def _report_kline_imgs(self):
            """为报告离屏生成 日K(近3月)/周K(近1.5年)/月K(近6年) 三张图(带均线+区间平均)，返回HTML。"""
            df = getattr(self, "raw_df", None)
            if df is None or "close" not in getattr(df, "columns", []):
                return ""
            html = []
            for pname, rule, lookback in self._kline_periods:
                try:
                    kdf = self._resample_kline(df, rule).tail(lookback).reset_index(drop=True)
                    if len(kdf) < 3:
                        continue
                    tmp = Figure(figsize=(8, 5))
                    self._draw_kline(kdf, f"{pname}（虚线=区间平均{kdf['close'].mean():.2f}）",
                                     avg=float(kdf["close"].mean()), fig=tmp)
                    html.append(self._fig_to_img(tmp, pname))
                except Exception:
                    continue
            return "".join(html)

        # ---- 9.2.1e 机器学习内部（数据透视）标签页 ----
        def _build_mldata_tab(self) -> QWidget:
            """
            把"喂给模型的东西"完全透明化：真实趋势图（并标出训练/测试分界，证明不泄露）、
            每个输入特征的真实来源（诚实标注哪些已接入、哪些还没接入）、以及最终数据集预览。
            """
            panel = QWidget()
            layout = QVBoxLayout(panel)

            top = QHBoxLayout()
            self.mldata_btn = QPushButton("🔬 生成/刷新 数据透视")
            self.mldata_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.mldata_btn.clicked.connect(self._on_build_mldata)
            top.addWidget(self.mldata_btn)
            self.mldata_hint = QLabel("按左侧数据配置(代码/日期/数据源/预测目标)与训练比例，生成本页透视。")
            self.mldata_hint.setStyleSheet("color:#666;")
            top.addWidget(self.mldata_hint, stretch=1)
            layout.addLayout(top)

            split = QSplitter(Qt.Vertical)
            # 上：真实趋势 + 训练/测试分界
            self.mldata_figure = Figure(figsize=(8, 3))
            self.mldata_canvas = FigureCanvas(self.mldata_figure)
            split.addWidget(self.mldata_canvas)
            # 中：输入的真实来源说明 + 溯源链接
            self.mldata_srcbox = QTextBrowser()             # QTextBrowser 支持点击超链接
            self.mldata_srcbox.setOpenExternalLinks(True)   # 让 <a href> 链接在系统浏览器打开
            split.addWidget(self.mldata_srcbox)
            # 下：左右分栏 —— 左=各来源"分析后的信息"(真实最新值+新闻)，右=数据集预览表
            bottom = QSplitter(Qt.Horizontal)
            self.mldata_analysis = QTextBrowser()
            self.mldata_analysis.setOpenExternalLinks(True)
            bottom.addWidget(self.mldata_analysis)
            self.mldata_table = QTableWidget()
            bottom.addWidget(self.mldata_table)
            bottom.setSizes([560, 460])
            split.addWidget(bottom)
            split.setSizes([280, 230, 340])
            layout.addWidget(split, stretch=1)
            return panel

        def _on_build_mldata(self):
            self.mldata_btn.setEnabled(False); self.mldata_hint.setText("⏳ 正在拉取数据并生成透视 ...")
            QApplication.setOverrideCursor(Qt.WaitCursor); self._prog_open(); QApplication.processEvents()
            try:
                use_synthetic = self.data_source_combo.currentIndex() == 1
                enrich_status, news = {}, None
                if use_synthetic:
                    df = StockDataFetcher.generate_synthetic_data()
                    code = "合成数据"
                else:
                    code = self.code_edit.text().strip()
                    start = self.start_date.date().toString("yyyyMMdd")
                    end = self.end_date.date().toString("yyyyMMdd")
                    self.mldata_hint.setText(f"正在拉取 {code} 行情与外部数据 ...")
                    QApplication.processEvents()
                    df = StockDataFetcher().fetch(code, start, end)
                    if self.chk_val.isChecked() or self.chk_idx.isChecked() or self.chk_mf.isChecked():
                        df, enrich_status = StockDataFetcher.enrich(
                            df, code, self.chk_val.isChecked(),
                            self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
                    if self.chk_news.isChecked():
                        try:
                            news = StockDataFetcher.fetch_news(code)
                        except Exception as e:
                            enrich_status["news"] = f"跳过(取数失败: {e})"
                    self._mldata_name = StockDataFetcher.fetch_stock_name(code)   # 供ST退市风险判断

                target_mode = "return" if self.target_combo.currentIndex() == 0 else "price"
                horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
                train_ratio = next(r for r, rb in self.split_radios.items() if rb.isChecked())
                fe = FeatureEngineer(window_size=20, horizon=horizon, target_mode=target_mode)
                X, y, sample_dates = fe.build_supervised_samples(df)
                if len(X) < 10:
                    QMessageBox.warning(self, "提示", "样本太少，无法透视，请拉长日期区间。")
                    return
                a = int(len(X) * train_ratio)              # 训练/验证/测试三分切点，与真实训练一致
                b = a + (len(X) - a) // 2
                self._draw_mldata_trend(code, sample_dates, fe.close_target_, a, b, target_mode)
                self._fill_mldata_source(fe, target_mode, len(X), a, b, enrich_status, None,
                                         code if not use_synthetic else "")
                name = getattr(self, "_mldata_name", "") if not use_synthetic else ""
                warns = assess_risks(code, name, df, news) if not use_synthetic else []
                self._fill_mldata_analysis(df, code if not use_synthetic else "", enrich_status, news,
                                           name, warns)
                self._fill_mldata_table(fe, y, sample_dates, a, b, target_mode)
                self.mldata_hint.setText(
                    f"{code}：预测周期 {horizon}日；共 {len(X)} 条样本 = 训练 {a} / 验证 {b-a} / 测试 {len(X)-b}。"
                    f"验证集为独立留出，不参与训练/测试/寻优。")
                self._oplog(f"数据透视：{code}，周期{horizon}日，特征{len(fe.feature_cols)}个，"
                            f"训练{a}/验证{b-a}/测试{len(X)-b}。")
            except Exception as e:
                QMessageBox.critical(self, "数据透视失败", str(e))
                self.mldata_hint.setText("失败，详见弹窗。可先用合成数据看效果。")
            finally:
                QApplication.restoreOverrideCursor(); self._prog_close(); self.mldata_btn.setEnabled(True)

        def _draw_mldata_trend(self, code, sample_dates, close_target, a, b, target_mode):
            self.mldata_figure.clear()
            ax = self.mldata_figure.add_subplot(111)
            x = np.arange(len(close_target))
            ax.plot(x, close_target, color="#333", linewidth=1.0, label="真实收盘价")
            n = len(x)
            # 训练(蓝)/验证(绿)/测试(橙)三区背景，直观证明"按时间顺序切分、不泄露"
            ax.axvspan(0, a, color="#4a90d9", alpha=0.10)
            ax.axvspan(a, b, color="#2ecc71", alpha=0.12)
            ax.axvspan(b, n, color="#e08a2c", alpha=0.12)
            for xline in (a, b):
                ax.axvline(xline, color="#c0392b", linestyle="--", linewidth=1.0)
            top = ax.get_ylim()[1]
            ax.text(a / 2, top, "训练集", ha="center", va="top", color="#2c6fbb", fontsize=9)
            ax.text((a + b) / 2, top, "验证集", ha="center", va="top", color="#1e8449", fontsize=9)
            ax.text((b + n) / 2, top, "测试集", ha="center", va="top", color="#d35400", fontsize=9)
            idx = np.linspace(0, n - 1, min(8, n)).astype(int)
            ax.set_xticks(x[idx])
            ax.set_xticklabels([pd.to_datetime(sample_dates.iloc[i]).strftime("%y-%m-%d") for i in idx],
                               rotation=30, fontsize=8)
            ax.set_ylabel("收盘价")
            ax.set_title(f"{code} 真实趋势与训练/验证/测试三分（预测目标：{'涨跌幅' if target_mode=='return' else '价格'}）")
            ax.grid(True, alpha=0.25)
            self.mldata_figure.tight_layout()
            self.mldata_canvas.draw()

        @staticmethod
        def _source_links_html(code: str) -> str:
            """为当前股票生成"数据溯源"真实链接(点击在浏览器打开对应来源网页)。"""
            if not code:
                return ("<p><b>数据溯源：</b>当前为合成数据(无真实来源)。选真实数据并填股票代码后，"
                        "这里会列出可点击跳转的真实来源网址。</p>")
            mkt = "sh" if code.startswith("6") else ("bj" if code.startswith(("4", "8", "920")) else "sz")
            MKT = mkt.upper()
            links = [
                ("行情/K线（东方财富个股页）", f"https://quote.eastmoney.com/{mkt}{code}.html"),
                ("公司财报/F10（东方财富财务分析）",
                 f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={MKT}{code}#/cwfx"),
                ("估值 PE/PB/市值（百度股市通）", f"https://gushitong.baidu.com/stock/ab-{code}"),
                ("主力资金流向（东方财富）", f"https://data.eastmoney.com/zjlx/{code}.html"),
                ("大盘环境 沪深300（东方财富）", "https://quote.eastmoney.com/zs000300.html"),
                ("个股新闻（东方财富搜索）", f"https://so.eastmoney.com/news/s?keyword={code}"),
                ("公司公告/招股书（巨潮资讯 官方披露）",
                 f"http://www.cninfo.com.cn/new/fulltextSearch?keyWord={code}"),
            ]
            rows = "".join(f"<tr><td>{n}</td><td><a href='{u}'>{u}</a></td></tr>" for n, u in links)
            return (f"<h3>数据溯源（{code} · 点击可在浏览器打开真实来源）</h3>"
                    "<p>本软件所有真实数据均来自以下公开网站，可逐一核对：</p>"
                    "<table border=1 cellpadding=4 cellspacing=0 width=100%>"
                    f"<tr bgcolor=#eef><th>数据类别</th><th>真实来源网址（可点击）</th></tr>{rows}</table>")

        def _fill_mldata_source(self, fe, target_mode, n_samples, a, b,
                                enrich_status=None, news=None, code=""):
            cols = list(getattr(fe, "feature_cols", []))
            n_feat = len(cols)
            has = lambda pref: any(c.startswith(pref) for c in cols)   # 该类特征是否真的存在于特征列
            enrich_status = enrich_status or {}

            # 已接入（当前真实使用）的输入，按来源分组；状态根据"特征列里是否真有对应列"如实判定
            groups = [
                (True, "原始行情(日线)", "东方财富/baostock", "open/close/high/low/volume/amount 等"),
                (True, "技术指标", "由日线计算", "MA/EMA/MACD、RSI14、布林带、KDJ"),
                (True, "量能 & 动量波动", "由行情计算", "volume_ma5/volume_change、return_1d、volatility_10d"),
                (True, "市场情绪(量价代理)", "由行情计算", "量比、换手率、日内振幅、连涨跌天数、近20日区间位置"),
                (has("ret_") or has("pos_w") or has("madev_"), "周/月/季 多周期趋势(≈周K/月K线)",
                 "由日线滚动计算(无泄露)", "ret_w1/m1/q1、pos_*、madev_*（1周/1月/1季 涨跌·区间·均线偏离）"),
                (has("val_"), "财报估值(日频)", f"乐咕乐股/百度(自动) · {enrich_status.get('valuation','')}",
                 "val_pe_ttm / val_pb / val_total_mv 等(实际列见运行日志)"),
                (has("idx_"), "大盘环境(沪深300)", f"东方财富 · {enrich_status.get('index','')}",
                 "idx_ret_1d、idx_madev20"),
                (has("mf_"), "主力资金 & 主力意图代理", f"东方财富资金流 · {enrich_status.get('fundflow','')}",
                 "主力/超大单/大单净流入 mf_*；连续进/出场 mf_streak、累计 mf_cum5/20、"
                 "资金-价格背离 mf_price_div(吸筹/洗盘 vs 派发 代理)"),
            ]
            rows = []
            for ok, g, src, ex in groups:
                mark = "✅" if ok else "⛔ 未接入"
                color = "" if ok else " bgcolor=#fee"
                rows.append(f"<tr{color}><td>{mark} {g}</td><td>{src}</td><td>{ex}</td></tr>")
            rows_html = "".join(rows)

            # 新闻：真实标题+链接，仅展示、不作历史训练特征（诚实说明原因）
            if news is not None and len(news) > 0:
                items = []
                for _, r in news.iterrows():
                    t = str(r.get("title", "")); u = str(r.get("url", "")); tm = str(r.get("time", ""))
                    items.append(f"<li>{tm}　<a href='{u}'>{t}</a></li>")
                news_html = ("<p><b>个股新闻(真实标题+链接，来自东方财富)：</b>"
                             "<span style='color:#c0392b'>仅展示、不作为历史训练特征</span>"
                             "——新闻接口只覆盖近期，且把文本变成可靠的历史情绪分需要 NLP 模型，"
                             "为避免臆造，本软件不把编造的情绪分喂给模型。</p>"
                             f"<ul>{''.join(items)}</ul>")
            else:
                news_html = ("<p><b>个股新闻：</b>真实标题+链接见右下「各来源数据分析速览」面板；"
                             "真实来源=东方财富 stock_news_em；仅展示、不作历史训练特征(避免臆造情绪分)。</p>")

            html = f"""
            <h3>输入的真实来源（当前共 {n_feat} 个特征 × 窗口20天 = 每条样本 {n_feat*20} 维）</h3>
            <p>下表状态按"特征列里是否真有对应列"如实判定；⛔ 表示该来源本次未接入(取数失败/未勾选)，
            <b>软件绝不用假数据填充</b>。</p>
            <table border=1 cellpadding=4 cellspacing=0 width=100%>
              <tr bgcolor=#eef><th>输入类别</th><th>真实来源</th><th>具体特征</th></tr>{rows_html}
            </table>
            {self._source_links_html(code)}
            {news_html}
            <p><b>关于"主力意图"与"买卖手"（诚实说明）：</b>
            "主力进场/退场/洗盘"没有权威标签，本软件不臆造"主力预言"，而是接入<b>真实的每日主力资金净流入</b>
            并派生代理特征(连续净流入天数、资金-价格背离)喂给模型，由模型自己学。
            实时盘口"买1-5/卖1-5(买卖手)"是<b>快照、无免费历史序列</b>，用于历史回测会造成泄露或造假，
            因此用可回测的"每日主力/大单净额"作为其历史等价物。</p>
            <p style="color:#2c6fbb"><b>为什么不会泄露未来数据：</b>
            ① 所有特征(含估值/大盘)都按日期"向后对齐"，只用当天及以前的已知值；
            ② 严格按时间顺序三分——训练 {a} 条 / 验证 {b-a} 条 / 测试 {n_samples-b} 条，绝不打乱；
            ③ 验证集为<b>独立留出</b>(不参与训练/测试/寻优)；标准化器只在训练集上 fit，再套用到验证/测试集。</p>
            """
            self._html_source = html                 # 供"综合报告"复用
            self.mldata_srcbox.setHtml(html)

        def _fill_mldata_analysis(self, df, code, enrich_status=None, news=None, name="", warns=None):
            """各来源"分析后的信息"：从真实数据里算出每个来源的最新值/近况(不预测、不臆造)。
            顶部为风险提示(退市/亏损/暴跌/负面新闻)，全部基于真实数据+真实新闻关键词命中。"""
            enrich_status = enrich_status or {}
            warns = warns or []

            # ---- 顶部：风险提示（红色，最醒目）----
            if warns:
                color = {"高": "#c0392b", "中": "#e67e22", "低": "#888"}
                items = []
                for wn in warns:
                    c = color.get(wn["level"], "#888")
                    link = f" <a href='{wn['url']}'>[原文]</a>" if wn.get("url") else ""
                    items.append(f"<li><b style='color:{c}'>【{wn['level']}·{wn['category']}】</b>"
                                 f"{wn['msg']}<span style='color:#888'>（来源：{wn['source']}）</span>{link}</li>")
                risk_html = ("<div style='background:#fff4f4;border:1px solid #f0b0b0;padding:6px'>"
                             f"<b style='color:#c0392b'>⚠️ 风险提示（{len(warns)} 条，均基于真实数据/新闻自动筛查，非投资建议）</b>"
                             f"<ul>{''.join(items)}</ul></div>")
            else:
                risk_html = ("<div style='background:#f2fbf2;border:1px solid #b7e0b7;padding:6px'>"
                             "<b style='color:#1e8449'>✅ 未发现明显风险信号</b>"
                             "（基于名称ST/价格/亏损/近期跌幅/新闻关键词的自动筛查；不代表无风险，非投资建议）</div>")

            def _money(v):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return "-"
                if abs(v) >= 1e8:
                    return f"{v/1e8:.2f} 亿"
                return f"{v/1e4:.0f} 万"

            def _last(col):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s):
                        return float(s.iloc[-1])
                return None

            secs = []
            # ---- 行情速览（一定有）----
            close = pd.to_numeric(df["close"], errors="coerce")
            lc = float(close.iloc[-1]); ld = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")
            def _chg(w):
                return (lc / float(close.iloc[-1-w]) - 1) * 100 if len(close) > w else float("nan")
            vol20 = close.pct_change().tail(20).std() * 100
            ma20 = close.tail(20).mean()
            secs.append(
                "<b>📈 行情/K线：</b>最新收盘 <b>{:.2f}</b> 元（{}）；近1周 {:+.2f}%、近1月 {:+.2f}%、"
                "近1季 {:+.2f}%；近20日波动率 {:.2f}%；{}20日均线({:.2f})。".format(
                    lc, ld, _chg(5), _chg(20), _chg(60), vol20,
                    "站上" if lc >= ma20 else "跌破", ma20))
            # ---- 财报估值 ----
            pe, pb, mv = _last("val_pe_ttm"), _last("val_pb"), _last("val_total_mv")
            if any(v is not None for v in (pe, pb, mv)):
                parts = []
                if pe is not None: parts.append(f"PE(TTM) {pe:.1f}" if pe > 0 else "PE(TTM) 亏损/无")
                if pb is not None: parts.append(f"PB {pb:.2f}")
                if mv is not None: parts.append(f"总市值 {_money(mv)}元")
                secs.append("<b>💰 财报估值：</b>" + "；".join(parts) +
                            "　<span style='color:#888'>（乐咕乐股/百度）</span>")
            elif enrich_status.get("valuation"):
                secs.append(f"<b>💰 财报估值：</b>{enrich_status['valuation']}")
            # ---- 主力资金 ----
            mf = _last("mf_main_net")
            if mf is not None:
                mfs = pd.to_numeric(df["mf_main_net"], errors="coerce").fillna(0.0)
                cum5 = mfs.tail(5).sum()
                sgn = np.sign(mfs.to_numpy())
                streak = 1
                for k in range(len(sgn) - 2, -1, -1):
                    if sgn[k] == sgn[-1] and sgn[-1] != 0:
                        streak += 1
                    else:
                        break
                trend = "连续净流入" if sgn[-1] > 0 else ("连续净流出" if sgn[-1] < 0 else "持平")
                secs.append(
                    f"<b>🏦 主力资金：</b>最新主力净流入 <b>{_money(mf)}</b>元；近5日累计 {_money(cum5)}元；"
                    f"{trend} {streak} 天。<span style='color:#888'>（东方财富）</span>")
            elif enrich_status.get("fundflow"):
                secs.append(f"<b>🏦 主力资金：</b>{enrich_status['fundflow']}")
            # ---- 大盘环境 ----
            idxr = _last("idx_ret_1d")
            if idxr is not None:
                secs.append(f"<b>🌐 大盘环境：</b>沪深300 最新日涨跌 {idxr*100:+.2f}%。"
                            "<span style='color:#888'>（东方财富）</span>")
            # ---- 新闻（真实标题+链接）----
            if news is not None and len(news) > 0:
                items = "".join(
                    f"<li>{str(r.get('time',''))}　<a href='{r.get('url','')}'>{r.get('title','')}</a></li>"
                    for _, r in news.head(6).iterrows())
                secs.append("<b>📰 最近新闻（东方财富·点击可看原文）：</b><ul>" + items + "</ul>")
            elif enrich_status.get("news"):
                secs.append(f"<b>📰 新闻：</b>{enrich_status['news']}")

            title = f"{name}（{code}）" if (name and code) else (code or "合成数据")
            head = (f"<h3>各来源数据分析速览 · {title}</h3>"
                    "<p style='color:#888'>以下均为<b>真实数据的最新值/近况</b>，每项都注明来源，不是预测、不臆造。</p>")
            # ---- 底部：诚实的"分析依据"说明（避免被误当成因果归因）----
            foot = ("<hr><p style='color:#2c6fbb'><b>关于「为什么这样分析」（诚实说明）：</b>"
                    "上面是模型可参考的<b>真实因素现状</b>——估值高低、主力资金近期流入/流出、大盘强弱、"
                    "近期涨跌、以及是否命中风险新闻。模型据此学习，但<b>股价由多因素共同驱动</b>，"
                    "本软件<b>不会</b>替你断言「这次涨跌是机构/主力/散户/某条新闻造成的」——那种单一因果归因"
                    "免费数据无法可靠判定，谁那么说谁在误导你。请把以上当作客观事实参考，自行判断。</p>")
            self._html_risk = risk_html              # 供"综合报告"复用
            self._html_analysis = head + "".join(f"<p>{s}</p>" for s in secs) + foot
            self.mldata_analysis.setHtml(risk_html + self._html_analysis)

        def _fill_mldata_table(self, fe, y, sample_dates, a, b, target_mode):
            n = len(y)
            tgt_label = "目标(涨跌幅%)" if target_mode == "return" else "目标(收盘价)"
            headers = ["序号", "目标日期", "基准日收盘", tgt_label, "所属集合"]
            self.mldata_table.setColumnCount(len(headers))
            self.mldata_table.setHorizontalHeaderLabels(headers)
            # 样本可能上千条：为流畅只展示各集合的头尾片段(含验证集)，中间用省略行占位
            if n <= 150:
                show_idx = list(range(n))
            else:
                def _seg(lo, hi, k=12):        # 取某区间的前 k 与后 k
                    lo, hi = max(0, lo), min(n, hi)
                    r = list(range(lo, hi))
                    return r if len(r) <= 2 * k else r[:k] + [-1] + r[-k:]
                show_idx = _seg(0, a) + [-1] + _seg(a, b) + [-1] + _seg(b, n)
            self.mldata_table.setRowCount(len(show_idx))
            for row, i in enumerate(show_idx):
                if i == -1:
                    self.mldata_table.setItem(row, 0, QTableWidgetItem("..."))
                    for c in range(1, len(headers)):
                        self.mldata_table.setItem(row, c, QTableWidgetItem("..."))
                    continue
                d = pd.to_datetime(sample_dates.iloc[i]).strftime("%Y-%m-%d")
                pc = fe.prev_close_[i]
                tv = (y[i] * 100) if target_mode == "return" else y[i]
                grp = "训练集" if i < a else ("验证集" if i < b else "测试集")
                vals = [str(i), d, f"{pc:.2f}", f"{tv:.3f}", grp]
                for c, v in enumerate(vals):
                    self.mldata_table.setItem(row, c, QTableWidgetItem(v))
            self.mldata_table.resizeColumnsToContents()

        # ---- 9.2.1h 综合报告标签页（汇总所有图表+数据，可预览、可导出HTML） ----
        def _build_report_tab(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            self.rcard_btn = QPushButton("🧭 综合研判卡(单只·只给事实不荐股)")
            self.rcard_btn.setStyleSheet("font-weight:bold; padding:6px; background:#8e44ad; color:white;")
            self.rcard_btn.clicked.connect(self._on_research_card)
            top.addWidget(self.rcard_btn)
            self.report_btn = QPushButton("🧾 生成/刷新 报告预览")
            self.report_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.report_btn.clicked.connect(self._on_gen_report)
            top.addWidget(self.report_btn)
            self.report_export_btn = QPushButton("💾 导出 HTML 报告")
            self.report_export_btn.setStyleSheet("font-weight:bold; padding:6px; background:#2c6fbb; color:white;")
            self.report_export_btn.clicked.connect(self._on_export_report)
            top.addWidget(self.report_export_btn)
            self.report_hint = QLabel("先在各页生成内容(K线/预测/回测/未来/数据透视)，再点这里汇总；报告可上下滚动、可导出。")
            self.report_hint.setStyleSheet("color:#666;")
            top.addWidget(self.report_hint, stretch=1)
            layout.addLayout(top)
            self.report_view = QTextBrowser()        # 可滚动预览，图表以内嵌图片原样呈现
            self.report_view.setOpenExternalLinks(True)
            layout.addWidget(self.report_view, stretch=1)
            return panel

        @staticmethod
        def _fig_to_img(fig, title):
            """把一个 matplotlib 图转成 <h2>标题</h2><img base64>；图为空则返回空串。"""
            if fig is None or not fig.get_axes():
                return ""
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            except Exception:
                return ""
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return (f"<h2>{title}</h2>"
                    f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;"
                    "border:1px solid #ddd'/>")

        def _results_table_html(self):
            """把训练结果(每格 训练/验证/测试)拼成 HTML 表格。"""
            valid = [r for r in self.results if not r.error]
            if not valid:
                return ""
            metric_names = list(valid[0].metrics.keys())

            def _f(d, k):
                if not d or k not in d:
                    return "-"
                v = d[k]
                return f"{v:.3f}" if isinstance(v, float) else str(v)
            head = "<tr bgcolor=#eef><th>算法</th>" + "".join(f"<th>{m}(训/验/测)</th>" for m in metric_names) + "</tr>"
            rows = []
            for r in self.results:
                if r.error:
                    rows.append(f"<tr><td>{r.algo_name}</td><td colspan={len(metric_names)}>失败: {r.error}</td></tr>")
                    continue
                cells = "".join(f"<td>{_f(r.metrics_train,m)} / {_f(r.metrics_val,m)} / {_f(r.metrics,m)}</td>"
                                for m in metric_names)
                rows.append(f"<tr><td><b>{r.algo_name}</b></td>{cells}</tr>")
            return ("<h2>指标结果（训练 / 验证 / 测试）</h2>"
                    "<p style='color:#888'>训练误差远小于验证/测试即为过拟合；DA/UP_P 需明显高于「总是涨」基准。</p>"
                    f"<table border=1 cellpadding=4 cellspacing=0 width=100%>{head}{''.join(rows)}</table>")

        def _build_report_html(self):
            """把前面所有图表(原样内嵌为图片)+数据(表格/风险/分析/溯源)汇总成一份 HTML。"""
            code = self.code_edit.text().strip() if hasattr(self, "code_edit") else ""
            when = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = [
                "<h1 style='color:#2c6fbb'>A 股综合分析报告</h1>",
                f"<p>股票代码：<b>{code or '合成数据'}</b>　生成时间：{when}</p>",
                "<div style='background:#fff4f4;border:2px solid #c0392b;padding:8px;margin:6px 0'>"
                "<b style='color:#c0392b'>⚠️ 合规与免责（务必阅读）</b>"
                "<ul style='margin:4px 0'>"
                "<li><b>仅供学术研究与学习，不构成任何投资建议</b>，据此交易风险自负；"
                "市场接近有效，历史表现不代表未来。</li>"
                "<li>数据来自公开接口，请仅作个人研究：勿高频抓取、勿售卖/再分发数据、"
                "勿无牌荐股或收费、勿大规模转发实时行情。</li>"
                "</ul></div>",
                # 大白话导读：把界面里讲的几件事一并写进报告，报告可独立阅读
                "<div style='background:#f4f8ff;border:1px solid #2c6fbb;padding:8px;margin:6px 0'>"
                "<b style='color:#2c6fbb'>📖 如何看懂本报告(写给不懂金融的你)</b>"
                "<ul style='margin:4px 0'>"
                "<li><b>「预测结果对比图」两条线几乎重合 ≠ 预测神准</b>。因为股票『明天价≈今天价』，"
                "把昨天的价格照抄过来(Naive基准)也会几乎重合。真正该看的是<b>方向准确率DA</b>"
                "(猜涨跌方向对的比例，50%≈抛硬币)和<b>策略回测</b>能不能真赚钱。</li>"
                "<li><b>策略回测</b>：假设『模型说涨就买、说跌就空仓』走一遍历史(已扣手续费)，"
                "看最后赚多少，并和『一直持有不动』比。<b>跑不赢一直持有，就说明这个模型没用</b>。</li>"
                "<li><b>市盈率PE(TTM)</b>=股价÷每股年利润；<b>为负数=公司在亏损</b>(没有利润)，"
                "并非越低越好，负数要警惕。其余名词见软件「❓名词解释」按钮。</li>"
                "<li><b>「模型对后市的机械倾向」表里的『偏涨/偏跌』只是模型按历史算出的机械方向，"
                "不是预言、更不是买卖建议</b>；每行都并列了它的可信度(DA)，"
                "DA≈50%就等于抛硬币、请直接忽略。最终买不买由你自己判断。</li>"
                "</ul></div>",
            ]
            # 综合研判卡放最前(客观事实汇总，非建议)
            if getattr(self, "_html_card", ""):
                parts.append(self._html_card)
            # 风险提示
            if getattr(self, "_html_risk", ""):
                parts.append("<h2>⚠️ 风险提示</h2>" + self._html_risk)
            # 模型推荐与预测依据
            if getattr(self, "_html_reco", ""):
                parts.append("<h2>🎯 模型推荐与预测依据</h2>" + self._html_reco)
            # 日K/周K/月K 三张图（报告自动生成，带均线+区间平均，无需先打开K线页）
            parts.append("<h2>行情 K 线（日K近3月 / 周K近1.5年 / 月K近6年）</h2>")
            parts.append(self._report_kline_imgs())
            # 其它图表（原样内嵌，优先级高）
            for title, fig in [
                ("真实趋势与训练/验证/测试划分", getattr(self, "mldata_figure", None)),
                ("预测结果对比图", getattr(self, "figure", None)),
                ("未来走势预测图", getattr(self, "fc_figure", None)),
                ("策略回测（净值曲线）", getattr(self, "bt_figure", None)),
            ]:
                parts.append(self._fig_to_img(fig, title))
            # 指标结果表格
            parts.append(self._results_table_html())
            # 数据分析速览（含各来源真实最新值 + 新闻）
            if getattr(self, "_html_analysis", ""):
                parts.append("<h2>各来源数据分析速览</h2>" + self._html_analysis)
            # 输入来源 + 数据溯源链接
            if getattr(self, "_html_source", ""):
                parts.append(self._html_source)
            body = "".join(p for p in parts if p)
            # 表格/图片样式（优先级最高：撑满宽度、可横向不溢出）
            style = ("<style>body{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:1000px;"
                     "margin:12px auto;line-height:1.6;color:#222} h1,h2{border-bottom:1px solid #eee;padding-bottom:4px}"
                     "table{border-collapse:collapse;width:100%} img{max-width:100%;height:auto}"
                     "a{color:#2c6fbb;word-break:break-all}</style>")
            return f"<html><head><meta charset='utf-8'>{style}</head><body>{body}</body></html>"

        def _on_research_card(self):
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "综合研判卡需真实数据，请把数据源切到「真实数据」。"); return
            code = self.code_edit.text().strip()
            tm = "return" if self.target_combo.currentIndex() == 0 else "price"
            horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
            self.rcard_btn.setEnabled(False); self.report_hint.setText("⏳ 正在生成综合研判卡 ...")
            QApplication.setOverrideCursor(Qt.WaitCursor); self._prog_open(); QApplication.processEvents()
            try:
                start = self.start_date.date().toString("yyyyMMdd")
                end = self.end_date.date().toString("yyyyMMdd")
                card = research_card(code, start, end, target_mode=tm, horizon=horizon,
                                     models=("SVR", "Lasso"), progress_cb=self._log)
                self._html_card = research_card_html(card)
                # 研判卡后面附上数据溯源链接，方便逐条核对
                links = self._source_links_html(code)
                self.report_view.setHtml(self._html_card + links)
                self.report_hint.setText("研判卡已生成(只给客观事实、不荐股)；它也会出现在「生成报告预览」的最前面。")
                self._oplog(f"生成综合研判卡：{code}。")
            except Exception as e:
                QMessageBox.critical(self, "研判卡生成失败", str(e))
            finally:
                QApplication.restoreOverrideCursor(); self._prog_close(); self.rcard_btn.setEnabled(True)

        def _on_gen_report(self):
            if not self.results and not getattr(self, "_html_analysis", "") \
                    and not getattr(self, "_html_card", ""):
                QMessageBox.information(self, "提示",
                                        "报告为空。请先在各页生成内容：\n"
                                        "① 行情K线图→获取K线  ② 运行模型  ③ 未来预测/策略回测  ④ 机器学习内部→数据透视\n"
                                        "(真实数据下生成报告时会自动补一张含『模型心情/倾向』的综合研判卡)\n"
                                        "然后回来点「生成/刷新 报告预览」。")
                return
            self.report_btn.setEnabled(False); self.report_hint.setText("⏳ 正在生成报告(含K线图) ...")
            QApplication.setOverrideCursor(Qt.WaitCursor); self._prog_open(); QApplication.processEvents()
            try:
                # 落实"全部弄进综合报告"：缺研判卡(含模型对后市的心情/倾向)但已用真实数据时，自动补一张
                if not getattr(self, "_html_card", "") and self.data_source_combo.currentIndex() != 1:
                    code0 = self.code_edit.text().strip()
                    if code0:
                        try:
                            self._prog_open("⏳ 报告缺研判卡，正在自动生成(含模型心情/倾向) ……")
                            tm = "return" if self.target_combo.currentIndex() == 0 else "price"
                            horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
                            start0 = self.start_date.date().toString("yyyyMMdd")
                            end0 = self.end_date.date().toString("yyyyMMdd")
                            card0 = research_card(code0, start0, end0, target_mode=tm, horizon=horizon,
                                                  models=("SVR", "Lasso"), progress_cb=self._log)
                            self._html_card = research_card_html(card0)
                            self._prog_open("⏳ 正在生成报告(含K线图) ……")
                        except Exception as e:
                            self._log(f"报告自动补研判卡失败(跳过，不影响其余内容)：{e}")
                self.report_view.setHtml(self._build_report_html())
                self.report_hint.setText("报告已生成，可上下滚动查看；点「导出 HTML 报告」保存为网页。")
                self._oplog("生成综合报告预览。")
            finally:
                QApplication.restoreOverrideCursor(); self._prog_close(); self.report_btn.setEnabled(True)

        def _on_export_report(self):
            html = self._build_report_html()
            code = self.code_edit.text().strip() or "report"
            default = os.path.join(BASE_DIR, f"报告_{code}_{dt.datetime.now():%Y%m%d_%H%M}.html")
            path, _ = QFileDialog.getSaveFileName(self, "导出 HTML 报告", default, "HTML 文件 (*.html)")
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._oplog(f"导出HTML报告：{path}")
                QMessageBox.information(self, "导出成功",
                                       f"报告已导出：\n{path}\n\n图表已内嵌为图片，双击即可用浏览器打开(离线可看)。")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

        # ---- 9.2.1i 实时监控标签页 ----
        def _build_monitor_tab(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            top.addWidget(QLabel("刷新间隔:"))
            self.mon_interval = QComboBox(); self.mon_interval.addItems(["3 秒", "5 秒", "10 秒", "30 秒"])
            self.mon_interval.setCurrentIndex(1)
            top.addWidget(self.mon_interval)
            self.mon_start_btn = QPushButton("▶ 开始监控")
            self.mon_start_btn.setStyleSheet("font-weight:bold; padding:6px; background:#1e8449; color:white;")
            self.mon_start_btn.clicked.connect(self._on_start_monitor)
            top.addWidget(self.mon_start_btn)
            self.mon_stop_btn = QPushButton("■ 停止")
            self.mon_stop_btn.clicked.connect(self._on_stop_monitor); self.mon_stop_btn.setEnabled(False)
            top.addWidget(self.mon_stop_btn)
            self.mon_fc_btn = QPushButton("🔮 预测明日/一周/一月")
            self.mon_fc_btn.clicked.connect(self._on_monitor_forecast)
            top.addWidget(self.mon_fc_btn)
            self.mon_view_snap_btn = QPushButton("📂 查看已记录快照")
            self.mon_view_snap_btn.clicked.connect(self._on_view_snapshots)
            top.addWidget(self.mon_view_snap_btn)
            self.mon_sincerity_btn = QPushButton("🔍 挂单诚意分析")
            self.mon_sincerity_btn.clicked.connect(self._on_order_sincerity)
            top.addWidget(self.mon_sincerity_btn)
            top.addStretch()
            layout.addLayout(top)
            self.mon_view = QTextBrowser()
            layout.addWidget(self.mon_view, stretch=1)
            self._mon_fc_html = ""
            self._monitor_timer = QTimer(self)
            self._monitor_timer.timeout.connect(self._refresh_monitor)
            return panel

        def _on_start_monitor(self):
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "实时监控需真实数据源，请把数据源切换为「真实数据」。")
                return
            sec = int(self.mon_interval.currentText().split()[0])
            self._mon_count = 0                      # 本次监控已记录的快照数
            self._mon_file = ""
            self._monitor_timer.start(sec * 1000)
            self.mon_start_btn.setEnabled(False); self.mon_stop_btn.setEnabled(True)
            self._oplog(f"开始实时监控 {self.code_edit.text().strip()}，每 {sec} 秒刷新，快照将记录到 monitor_log/。")
            self._refresh_monitor()

        def _record_snapshot(self, code, q):
            """把一次实时盘口快照追加写入 CSV（与 CLI --snapshot 同格式，供挂单分析）。"""
            try:
                self._mon_file = write_snapshot(code, q)
                self._mon_count = getattr(self, "_mon_count", 0) + 1
            except Exception:
                pass

        def _on_stop_monitor(self):
            self._monitor_timer.stop()
            self.mon_start_btn.setEnabled(True); self.mon_stop_btn.setEnabled(False)
            self._oplog("停止实时监控。")

        def _refresh_monitor(self):
            code = self.code_edit.text().strip()
            trading, status = is_trading_now()
            try:
                q = StockDataFetcher.fetch_realtime(code)
            except Exception as e:
                self.mon_view.setHtml(f"<p style='color:#c0392b'>实时行情获取失败：{e}</p>"
                                      "<p>可能是网络/接口波动，会在下次刷新重试；或先确认代码正确。</p>"
                                      f"<p style='color:#888'>已累计记录 {getattr(self,'_mon_count',0)} 条快照到 monitor_log/。</p>")
                return
            if self._monitor_timer.isActive():
                self._record_snapshot(code, q)       # 记录本次真实盘口快照

            def num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            price, prev = num(q.get("price")), num(q.get("prev"))
            pct = num(q.get("pct"))
            col = "#c0392b" if (pct or 0) >= 0 else "#1a9d5a"
            now = dt.datetime.now().strftime("%H:%M:%S")
            st_col = "#1e8449" if trading else "#888"
            head = (f"<h2>{code} 实时监控　<span style='color:{st_col}'>[{status}]</span>"
                    f"　<span style='color:#888;font-size:12px'>更新 {now}</span></h2>"
                    f"<div style='font-size:22px;color:{col}'><b>{q.get('price','-')}</b>　"
                    f"{('%+.2f%%' % pct) if pct is not None else ''}　"
                    f"{('(%+.2f)' % (price-prev)) if (price is not None and prev is not None) else ''}</div>")
            info = ("<table border=0 cellpadding=3><tr>"
                    f"<td>今开 {q.get('open','-')}</td><td>最高 <span style='color:#c0392b'>{q.get('high','-')}</span></td>"
                    f"<td>最低 <span style='color:#1a9d5a'>{q.get('low','-')}</span></td><td>昨收 {q.get('prev','-')}</td></tr>"
                    f"<tr><td>量比 {q.get('vol_ratio','-')}</td><td>换手 {q.get('turnover','-')}%</td>"
                    f"<td>成交量 {q.get('vol','-')}手</td><td>成交额 {q.get('amount','-')}</td></tr></table>")
            # 盘口五档（买卖手）
            def _lvl(rows, label, color):
                out = []
                for i, (p, v) in enumerate(rows):
                    out.append(f"<tr><td style='color:#888'>{label}{i+1}</td>"
                               f"<td style='color:{color}'>{p if p is not None else '-'}</td><td>{v if v is not None else '-'}</td></tr>")
                return "".join(out)
            pankou = ("<h3>盘口五档（买卖手）</h3><table border=1 cellpadding=3 cellspacing=0>"
                      "<tr bgcolor=#eef><th>档位</th><th>价</th><th>量(手)</th></tr>"
                      + _lvl(list(reversed(q.get("asks", []))), "卖", "#1a9d5a")
                      + "<tr><td colspan=3 bgcolor=#f6f6f6></td></tr>"
                      + _lvl(q.get("bids", []), "买", "#c0392b") + "</table>")
            rec = getattr(self, "_mon_count", 0)
            recfile = os.path.basename(getattr(self, "_mon_file", "")) or "monitor_log/"
            note = ("<p style='color:#888;font-size:12px'>盘口为实时快照(东方财富)；"
                    "本页只做<b>真实行情监控</b>，不预测下一分钟涨跌(那是噪声)。</p>"
                    f"<p style='color:#1e8449;font-size:12px'>📝 本次已记录 <b>{rec}</b> 条真实快照 → "
                    f"<b>{recfile}</b>（自采集的盘口/买卖手历史，长期积累后可供模型分析）。</p>")
            self.mon_view.setHtml(head + info + pankou + note + (self._mon_fc_html or ""))

        def _on_monitor_forecast(self):
            code = self.code_edit.text().strip()
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "预测需真实数据，请切换数据源为「真实数据」。"); return
            self.mon_fc_btn.setEnabled(False); self.mon_fc_btn.setText("预测中...")
            QApplication.processEvents()
            try:
                start = self.start_date.date().toString("yyyyMMdd")
                end = self.end_date.date().toString("yyyyMMdd")
                df = StockDataFetcher().fetch(code, start, end)
                if self.chk_val.isChecked() or self.chk_idx.isChecked() or self.chk_mf.isChecked():
                    df, _ = StockDataFetcher.enrich(df, code, self.chk_val.isChecked(),
                                                    self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
                tm = "return" if self.target_combo.currentIndex() == 0 else "price"
                algo = self.fc_model_combo.currentText() if hasattr(self, "fc_model_combo") else "Lasso"
                # 未来一周逐日(下1~5交易日) + 1月/3月锚点
                fc = forecast_curve(algo, df, target_mode=tm, horizons=(1, 2, 3, 4, 5, 20, 60),
                                    progress_cb=self._log)
                last_d = pd.to_datetime(fc["last_date"])
                mon_fut = next_trading_days(last_d, max((p["horizon"] for p in fc["points"]
                                                         if "pred_close" in p), default=1),
                                            progress_cb=self._log)
                nm = {20: "1个月后", 60: "3个月后"}
                def _row(p):
                    if p["horizon"] <= 5:
                        fd = mon_fut[p["horizon"] - 1]
                        label = f"下{('一二三四五')[p['horizon']-1]}({self._cn_weekday(fd)} {fd.strftime('%m-%d')})"
                    else:
                        label = nm.get(p["horizon"], f"{p['horizon']}日后")
                    col = '#c0392b' if p['change_pct'] >= 0 else '#1a9d5a'
                    return (f"<tr><td>{label}</td><td><b>{p['pred_close']:.2f}</b></td>"
                            f"<td style='color:{col}'>{p['change_pct']:+.2f}%</td></tr>")
                rows = "".join(_row(p) for p in fc["points"] if "pred_close" in p)
                self._mon_fc_html = (
                    f"<hr><h3>模型预测参考（{algo}，基于截至 {fc['last_date']} 的数据）</h3>"
                    "<p style='font-size:12px;color:#555'>未来一周逐日(下1~5交易日，约周一~周五)每天各一个独立直接预测。</p>"
                    "<table border=1 cellpadding=4 cellspacing=0><tr bgcolor=#eef><th>时点</th><th>预测价</th><th>涨跌</th></tr>"
                    + rows + "</table>"
                    "<p style='color:#c0392b;font-size:12px'>⚠ 日期按工作日推算(节假日顺延)；每天独立直接预测(非递归)，"
                    "越往后越不可靠，请对照 DA 判断，绝不构成投资建议。</p>")
                self._refresh_monitor() if self._monitor_timer.isActive() else self.mon_view.setHtml(
                    self.mon_view.toHtml() + self._mon_fc_html)
            except Exception as e:
                QMessageBox.critical(self, "预测失败", str(e))
            finally:
                self.mon_fc_btn.setEnabled(True); self.mon_fc_btn.setText("🔮 预测明日/一周/一月")

        def _on_view_snapshots(self):
            """列出 monitor_log/ 已采集的快照文件(股票/日期/条数)。"""
            try:
                files = sorted(f for f in os.listdir(MONITOR_DIR) if f.endswith(".csv"))
            except Exception:
                files = []
            if not files:
                self.mon_view.setHtml("<h3>暂无已记录快照</h3>"
                                      "<p>开始实时监控后，每次刷新会把真实盘口快照记录到 monitor_log/。</p>")
                return
            rows = []
            for f in files:
                path = os.path.join(MONITOR_DIR, f)
                try:
                    with open(path, encoding="utf-8-sig") as fh:
                        n = sum(1 for _ in fh) - 1        # 减表头
                    size = os.path.getsize(path) / 1024
                except Exception:
                    n, size = "?", 0
                rows.append(f"<tr><td>{f}</td><td align=right>{n}</td><td align=right>{size:.1f} KB</td></tr>")
            self.mon_view.setHtml(
                f"<h3>已记录快照文件（共 {len(files)} 个 · 目录 monitor_log/）</h3>"
                "<p style='color:#888'>这是你自采集的真实盘口/买卖手历史；长期积累后可供盘中建模分析。</p>"
                "<table border=1 cellpadding=4 cellspacing=0><tr bgcolor=#eef><th>文件</th><th>快照条数</th><th>大小</th></tr>"
                + "".join(rows) + "</table>"
                f"<p style='color:#888;font-size:12px'>文件夹路径：{MONITOR_DIR}</p>")
            self._oplog(f"查看已记录快照：{len(files)} 个文件。")

        def _on_order_sincerity(self):
            """挂单量变化(诚意粗判)：对当天已记录的盘口快照按价位作差。"""
            code = self.code_edit.text().strip()
            r = analyze_order_sincerity(code)
            if "error" in r:
                self.mon_view.setHtml(f"<h3>挂单诚意分析</h3><p style='color:#c0392b'>{r['error']}</p>"
                                      "<p style='color:#888'>做法：交易时段(尤其集合竞价9:15-9:25)多记录几次盘口，"
                                      "或用 <code>python stock_predictor.py --snapshot 代码</code> 定时抓取。</p>")
                return

            def tbl(rows, title, color):
                body = "".join(
                    f"<tr><td>{x['price']}</td><td align=right>{x['v0']:.0f}</td>"
                    f"<td align=right>{x['v1']:.0f}</td>"
                    f"<td align=right style='color:{'#c0392b' if x['chg_pct']<-20 else ('#1e8449' if x['chg_pct']>20 else '#333')}'>"
                    f"{x['chg_pct']:+.1f}%</td></tr>" for x in rows)
                return (f"<h4 style='color:{color}'>{title}</h4>"
                        "<table border=1 cellpadding=3 cellspacing=0 width=100%>"
                        "<tr bgcolor=#eef><th>价位</th><th>首次挂单</th><th>最新挂单</th><th>变化</th></tr>"
                        + body + "</table>")

            def verdict(keep, side):
                if keep is None:
                    return ""
                if keep < 60:
                    return f"<b style='color:#c0392b'>{side}大单存活 {keep}%——大量挂单已消失(撤单或成交)，谨慎</b>"
                if keep > 90:
                    return f"<b style='color:#1e8449'>{side}大单存活 {keep}%——挂单稳定，意愿相对真实</b>"
                return f"{side}大单存活 {keep}%"

            html = (f"<h3>挂单诚意分析 · {code}（{r['day']}，{r['n_snap']} 次快照）</h3>"
                    f"<p>对比时间：{r['t_first']} → {r['t_last']}</p>"
                    f"<p>{verdict(r['buy_keep_pct'],'买盘')}　|　{verdict(r['sell_keep_pct'],'卖盘')}</p>"
                    + tbl(r["buy"], "买盘(买手)各价位挂单变化", "#c0392b")
                    + tbl(r["sell"], "卖盘(卖手)各价位挂单变化", "#1a9d5a")
                    + "<p style='color:#c0392b;font-size:12px'>⚠️ 诚实边界：免费数据只看得到「各价位聚合挂单量」，"
                    "<b>无法区分撤单与成交</b>；要真正判断主力真伪需付费 Level-2 逐笔委托。本分析仅粗略参考，非定论、非投资建议。</p>")
            self.mon_view.setHtml(html)
            self._oplog(f"挂单诚意分析：{code}。")

        # ---- 9.2.1j 批量扫描标签页 ----
        def _build_batch_tab(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            top.addWidget(QLabel("股票代码(逗号/换行分隔):"))
            self.batch_codes = QLineEdit("600519,000001,002418,300750")
            top.addWidget(self.batch_codes, stretch=1)
            top.addWidget(QLabel("模型:"))
            self.batch_algo = QComboBox()
            self.batch_algo.addItems([n for n in ALGO_REGISTRY if ALGO_AVAILABILITY.get(n, True)])
            self.batch_algo.setCurrentText("Lasso")
            top.addWidget(self.batch_algo)
            self.batch_run_btn = QPushButton("▶ 预测扫描")
            self.batch_run_btn.setStyleSheet("font-weight:bold; padding:6px; background:#2c6fbb; color:white;")
            self.batch_run_btn.clicked.connect(self._on_batch_run)
            top.addWidget(self.batch_run_btn)
            self.factor_run_btn = QPushButton("🏆 因子打分选股")
            self.factor_run_btn.setStyleSheet("font-weight:bold; padding:6px; background:#8e44ad; color:white;")
            self.factor_run_btn.clicked.connect(self._on_factor_run)
            top.addWidget(self.factor_run_btn)
            self.batch_export_btn = QPushButton("💾 导出CSV")
            self.batch_export_btn.clicked.connect(self._on_batch_export)
            top.addWidget(self.batch_export_btn)
            layout.addLayout(top)
            self.batch_hint = QLabel("研究筛查工具，非荐股：用风险列避开雷；预测/DA高不等于该买。逐只训练，股票多会慢。")
            self.batch_hint.setStyleSheet("color:#c0392b;")
            layout.addWidget(self.batch_hint)
            self.batch_table = QTableWidget()
            layout.addWidget(self.batch_table, stretch=1)
            self._batch_rows = []
            return panel

        def _on_batch_run(self):
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "批量扫描需真实数据，请把数据源切换为「真实数据」。"); return
            raw = self.batch_codes.text().replace("\n", ",").replace("，", ",")
            codes = [c.strip() for c in raw.split(",") if c.strip()]
            if not codes:
                QMessageBox.warning(self, "提示", "请先填写股票代码。"); return
            start = self.start_date.date().toString("yyyyMMdd")
            end = self.end_date.date().toString("yyyyMMdd")
            tm = "return" if self.target_combo.currentIndex() == 0 else "price"
            horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
            flags = (self.chk_val.isChecked(), self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
            self.batch_run_btn.setEnabled(False); self.batch_run_btn.setText("扫描中...")
            self._oplog(f"批量扫描 {len(codes)} 只：{', '.join(codes)}（模型{self.batch_algo.currentText()}，周期{horizon}日）")
            self.batch_hint.setText("⏳ 正在批量扫描(每只都要训练模型，请耐心)... 进度见下方；也可切到「运行日志」看细节。")
            self._prog_open("⏳ 批量扫描进行中：逐只训练模型，请稍候 ……")
            self.batch_worker = BatchWorker(codes, self.batch_algo.currentText(), start, end, tm, horizon, flags)
            self.batch_worker.progress_signal.connect(self._log)
            self.batch_worker.progress_signal.connect(self.batch_hint.setText)   # 进度也显示在本页
            self.batch_worker.finished_signal.connect(self._on_batch_finished)
            self.batch_worker.error_signal.connect(lambda m: (QMessageBox.critical(self, "批量出错", m),
                                                              self._batch_reset_btn()))
            self.batch_worker.start()

        def _batch_reset_btn(self):
            self._prog_close()
            self.batch_run_btn.setEnabled(True); self.batch_run_btn.setText("▶ 预测扫描")
            self.factor_run_btn.setEnabled(True); self.factor_run_btn.setText("🏆 因子打分选股")

        def _on_factor_run(self):
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "因子选股需真实数据，请把数据源切换为「真实数据」。"); return
            raw = self.batch_codes.text().replace("\n", ",").replace("，", ",")
            codes = [c.strip() for c in raw.split(",") if c.strip()]
            if len(codes) < 2:
                QMessageBox.warning(self, "提示", "因子选股是横截面排名，至少填 2 只股票(越多排名越有意义)。"); return
            start = self.start_date.date().toString("yyyyMMdd")
            end = self.end_date.date().toString("yyyyMMdd")
            self.factor_run_btn.setEnabled(False); self.factor_run_btn.setText("打分中...")
            self.batch_run_btn.setEnabled(False)
            self._oplog(f"因子打分选股 {len(codes)} 只：{', '.join(codes)}")
            self.batch_hint.setText("⏳ 正在因子打分选股(逐只采集因子)... 进度见下方；也可切到「运行日志」看细节。")
            self._prog_open("⏳ 因子打分进行中：逐只采集因子并排名，请稍候 ……")
            self.factor_worker = FactorWorker(codes, start, end)
            self.factor_worker.progress_signal.connect(self._log)
            self.factor_worker.progress_signal.connect(self.batch_hint.setText)
            self.factor_worker.finished_signal.connect(self._on_factor_finished)
            self.factor_worker.error_signal.connect(lambda m: (QMessageBox.critical(self, "因子选股出错", m),
                                                               self._batch_reset_btn()))
            self.factor_worker.start()

        def _on_factor_finished(self, rows):
            self._batch_rows = rows                 # 复用导出
            self._batch_reset_btn()
            headers = ["排名", "代码", "名称", "最新价", "综合分", "价值分", "动量分", "资金分",
                       "质量分", "低波动分", "近3月%", "风险数", "主要风险"]
            self.batch_table.setColumnCount(len(headers))
            self.batch_table.setHorizontalHeaderLabels(headers)
            self.batch_table.setRowCount(len(rows))
            rank = 0
            for i, r in enumerate(rows):
                if r.get("error"):
                    self.batch_table.setItem(i, 1, QTableWidgetItem(r["code"]))
                    self.batch_table.setItem(i, 2, QTableWidgetItem(f"失败: {r['error'][:30]}"))
                    continue
                rank += 1
                m3 = r.get("mom3m")
                vals = [str(rank), r["code"], r.get("name", ""), f"{r.get('last_close','')}",
                        f"{r.get('score','')}", f"{r.get('value_score','') or '-'}",
                        f"{r.get('mom_score','') or '-'}", f"{r.get('money_score','') or '-'}",
                        f"{r.get('qual_score','') or '-'}", f"{r.get('lowvol_score','') or '-'}",
                        (f"{m3*100:+.1f}%" if m3 is not None else "-"),
                        str(r.get("risk_n", "")), r.get("risk_top", "")]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if c == 11 and r.get("risk_n", 0) > 0:      # "风险数"列
                        it.setForeground(Qt.red)
                    self.batch_table.setItem(i, c, it)
            self.batch_table.resizeColumnsToContents()
            self._oplog(f"因子打分完成：{sum(1 for r in rows if not r.get('error'))} 只已排名。")

        def _on_batch_finished(self, rows):
            self._batch_rows = rows
            self._batch_reset_btn()
            # 按预测涨跌幅降序（失败的排最后）
            rows_sorted = sorted(rows, key=lambda r: (r.get("pred_change_pct") is None,
                                                      -(r.get("pred_change_pct") or 0)))
            headers = ["代码", "名称", "最新价", "预测价", "预测涨跌%", "测试DA%", "UP_P%", "风险数", "主要风险"]
            self.batch_table.setColumnCount(len(headers))
            self.batch_table.setHorizontalHeaderLabels(headers)
            self.batch_table.setRowCount(len(rows_sorted))
            for i, r in enumerate(rows_sorted):
                if r.get("error"):
                    self.batch_table.setItem(i, 0, QTableWidgetItem(r["code"]))
                    self.batch_table.setItem(i, 1, QTableWidgetItem(f"失败: {r['error'][:30]}"))
                    continue
                vals = [r["code"], r.get("name", ""), f"{r.get('last_close','')}",
                        f"{r.get('pred_close','')}", f"{r.get('pred_change_pct','')}",
                        f"{r.get('DA','')}", f"{r.get('UP_P','')}", str(r.get("risk_n", "")),
                        r.get("risk_top", "")]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if c == 7 and r.get("risk_n", 0) > 0:
                        it.setForeground(Qt.red)
                    self.batch_table.setItem(i, c, it)
            self.batch_table.resizeColumnsToContents()
            self._oplog(f"批量扫描完成：{sum(1 for r in rows if not r.get('error'))} 只成功。")

        def _on_batch_export(self):
            if not self._batch_rows:
                QMessageBox.information(self, "提示", "请先运行批量扫描。"); return
            default = os.path.join(BASE_DIR, f"批量扫描_{dt.datetime.now():%Y%m%d_%H%M}.csv")
            path, _ = QFileDialog.getSaveFileName(self, "导出批量扫描结果", default, "CSV (*.csv)")
            if not path:
                return
            try:
                pd.DataFrame(self._batch_rows).to_csv(path, index=False, encoding="utf-8-sig")
                QMessageBox.information(self, "导出成功", f"已导出：\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

        # ---- 9.2.1i 预测跟踪标签页（存预测→到期对比真实→算真实准确率） ----
        def _build_tracking_tab(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            top.addWidget(QLabel("模型:"))
            self.trk_model_combo = QComboBox()
            self.trk_model_combo.addItems([n for n in ALGO_REGISTRY if ALGO_AVAILABILITY.get(n, True)])
            self.trk_model_combo.setCurrentText("Lasso")
            top.addWidget(self.trk_model_combo)
            b1 = QPushButton("📌 记录当前预测(1/3/5/20/60日)")
            b1.setStyleSheet("font-weight:bold; padding:6px;")
            b1.clicked.connect(self._on_record_prediction)
            top.addWidget(b1)
            b2 = QPushButton("🔄 验证并刷新(对比真实股价)")
            b2.setStyleSheet("font-weight:bold; padding:6px; background:#2c6fbb; color:white;")
            b2.clicked.connect(self._on_verify_predictions)
            top.addWidget(b2)
            top.addStretch()
            layout.addLayout(top)
            self.trk_summary = QTextBrowser(); self.trk_summary.setMaximumHeight(150)
            layout.addWidget(self.trk_summary)
            self.trk_table = QTableWidget()
            layout.addWidget(self.trk_table, stretch=1)
            self._refresh_tracking()
            return panel

        def _on_record_prediction(self):
            if self.data_source_combo.currentIndex() == 1:
                QMessageBox.information(self, "提示", "预测跟踪需真实数据(合成数据无意义)。请把数据源切到真实数据。")
                return
            code = self.code_edit.text().strip()
            model = self.trk_model_combo.currentText()
            target_mode = "return" if self.target_combo.currentIndex() == 0 else "price"
            try:
                self.trk_summary.setHtml(f"<p>正在用 {model} 为 {code} 生成 1/3/5/20/60 日预测并记录 ...</p>")
                QApplication.processEvents()
                df = StockDataFetcher().fetch(code, self.start_date.date().toString("yyyyMMdd"),
                                              self.end_date.date().toString("yyyyMMdd"))
                if self.chk_val.isChecked() or self.chk_idx.isChecked() or self.chk_mf.isChecked():
                    df, _ = StockDataFetcher.enrich(df, code, self.chk_val.isChecked(),
                                                    self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
                name = StockDataFetcher.fetch_stock_name(code)
                # 为每个周期先算一次"记录时的模型测试DA"(方向准确率)，随预测一起存档，增加可信力
                da_by_h = {}
                for h in (1, 3, 5, 20, 60):
                    try:
                        cfg = TrainConfig(horizon=h, target_mode=target_mode, hpo_method="关闭",
                                          metrics=["DA"])
                        res = TrainingPipeline(cfg).run_batch([model], df)
                        mr = next((r for r in res if r.algo_name == model and not r.error), None)
                        da_by_h[h] = round(float(mr.metrics.get("DA", 0)), 1) if mr else ""
                    except Exception:
                        da_by_h[h] = ""
                fc = forecast_curve(model, df, target_mode=target_mode,
                                    horizons=(1, 3, 5, 20, 60), progress_cb=self._log)
                base_date = fc["last_date"]; base_close = fc["last_close"]
                tgt_days = next_trading_days(pd.to_datetime(base_date), 60, progress_cb=self._log)
                rows = []
                for p in fc["points"]:
                    if "pred_close" not in p:
                        continue
                    h = p["horizon"]
                    tgt = tgt_days[h - 1].strftime("%Y-%m-%d")   # 真实交易日历定到期日(跳过周末+节假日)
                    rows.append({
                        "id": f"{code}_{model}_{h}_{base_date}",
                        "made_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "code": code, "name": name, "model": model, "horizon": h,
                        "model_da": da_by_h.get(h, ""),
                        "base_date": base_date, "base_close": base_close,
                        "pred_close": p["pred_close"], "pred_change_pct": p["change_pct"],
                        "pred_dir": "涨" if p["pred_close"] >= base_close else "跌",
                        "target_date": tgt, "status": "pending"})
                save_predictions(rows)
                self._oplog(f"记录预测：{code} {model}，{len(rows)} 个周期。")
                self._refresh_tracking()
                QMessageBox.information(self, "已记录",
                                       f"已记录 {len(rows)} 条预测(基准日 {base_date})。\n"
                                       "等目标日期到了，点「验证并刷新」用真实股价对比，即得真实准确率。")
            except Exception as e:
                QMessageBox.critical(self, "记录失败", str(e))

        def _on_verify_predictions(self):
            try:
                n = verify_predictions(progress_cb=self._log)
                self._oplog(f"验证预测：新对比真实股价 {n} 条。")
                self._refresh_tracking()
                QMessageBox.information(self, "验证完成",
                                       f"本次新验证 {n} 条(目标日期已到、拉真实股价对比)。\n"
                                       "未到期的显示「待验证」，到期后再来验证即可。")
            except Exception as e:
                QMessageBox.critical(self, "验证失败", str(e))

        def _refresh_tracking(self):
            # 顶部：真实准确率(仅已验证)
            acc = prediction_accuracy()
            if acc:
                rows = "".join(f"<tr><td>{k}</td><td>{v['n']}</td>"
                               f"<td><b>{v['dir_acc']}%</b></td><td>{v['mae_ret']}%</td></tr>"
                               for k, v in acc.items())
                html = ("<h3>真实预测准确率（仅统计已到期、已用真实股价验证的记录）</h3>"
                        "<table border=1 cellpadding=3 cellspacing=0>"
                        "<tr bgcolor=#eef><th>周期</th><th>已验证条数</th><th>方向准确率</th><th>涨跌幅平均误差</th></tr>"
                        f"{rows}</table>"
                        "<p style='color:#c0392b'>方向准确率需明显>50%才有意义；样本少时仅供参考，不构成投资建议。</p>")
            else:
                html = ("<h3>真实预测准确率</h3><p style='color:#888'>还没有已验证的预测。"
                        "先「记录当前预测」，等目标日期到了再「验证并刷新」——这样得到的才是<b>真实的未来对比</b>，"
                        "不是回测假象。</p>")
            self.trk_summary.setHtml(html)
            # 下：预测明细表
            df = load_pred_log()
            cols = ["made_at", "code", "name", "model", "horizon", "model_da", "base_date", "base_close",
                    "pred_close", "pred_change_pct", "pred_dir", "target_date", "status",
                    "actual_close", "actual_change_pct", "hit_dir"]
            heads = ["记录时间", "代码", "名称", "模型", "周期", "记录时DA%", "基准日", "基准价", "预测价",
                     "预测涨跌%", "方向", "目标日", "状态", "实际价", "实际涨跌%", "命中"]
            self.trk_table.setColumnCount(len(cols))
            self.trk_table.setHorizontalHeaderLabels(heads)
            df = df.tail(300).reset_index(drop=True) if len(df) else df
            self.trk_table.setRowCount(len(df))
            for i in range(len(df)):
                for j, c in enumerate(cols):
                    val = str(df.at[i, c]) if c in df.columns else ""
                    if c == "hit_dir":
                        val = "✓命中" if val == "1" else ("✗未中" if val == "0" else "-")
                    self.trk_table.setItem(i, j, QTableWidgetItem(val))
            self.trk_table.resizeColumnsToContents()

        # ---- 9.2.1f 策略回测标签页 ----
        def _build_backtest_tab(self) -> QWidget:
            """
            方向性收益回测：跑完模型后，选一个模型，按"预测涨就满仓、否则空仓"扣手续费模拟，
            与"买入持有"对比净值曲线。这是"预测涨跌到底能不能赚钱"的终极诚实检验。
            """
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            top.addWidget(QLabel("模型:"))
            self.bt_model_combo = QComboBox()
            top.addWidget(self.bt_model_combo)
            top.addWidget(QLabel("往返成本(%):"))
            self.bt_cost_edit = QLineEdit("0.2")
            self.bt_cost_edit.setFixedWidth(60)
            self.bt_cost_edit.setToolTip("一次买入+卖出的总成本：佣金+印花税+滑点。A股大致 0.15%~0.3%。")
            top.addWidget(self.bt_cost_edit)
            top.addWidget(QLabel("波动率目标(年化%,0=关):"))
            self.bt_voltgt_edit = QLineEdit("0")
            self.bt_voltgt_edit.setFixedWidth(46)
            self.bt_voltgt_edit.setToolTip("风控：把仓位缩放到目标年化波动。如填20，则高波动时自动减仓、低波动时满仓(不加杠杆)。0=不启用。")
            top.addWidget(self.bt_voltgt_edit)
            top.addWidget(QLabel("止损(%,0=关):"))
            self.bt_stop_edit = QLineEdit("0")
            self.bt_stop_edit.setFixedWidth(40)
            self.bt_stop_edit.setToolTip("风控：单段跌幅超过此值就止损离场(近似)。0=不启用。")
            top.addWidget(self.bt_stop_edit)
            self.bt_btn = QPushButton("▶ 回测")
            self.bt_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.bt_btn.clicked.connect(self._on_run_backtest)
            top.addWidget(self.bt_btn)
            top.addStretch()
            layout.addLayout(top)

            intro = QLabel("💡 什么是策略回测：假设你「模型说涨就买、说跌就空仓」，用过去的真实行情走一遍(还扣手续费)，"
                           "看最后赚了多少，再和「一直持有不动」对比。红线(策略)明显在蓝线(买入持有)之上才算有用；"
                           "跑不赢就说明这个预测不值得跟。这是检验预测能不能赚钱的照妖镜。")
            intro.setWordWrap(True); intro.setStyleSheet("color:#555; background:#f6f8fb; padding:6px;")
            layout.addWidget(intro)
            self.bt_stat = QLabel("先在左侧运行模型，再来这里选模型回测。")
            self.bt_stat.setWordWrap(True)
            self.bt_stat.setStyleSheet("padding:4px;")
            layout.addWidget(self.bt_stat)

            self.bt_figure = Figure(figsize=(8, 4))
            self.bt_canvas = FigureCanvas(self.bt_figure)
            layout.addWidget(self.bt_canvas, stretch=1)
            return panel

        def _on_run_backtest(self):
            r = next((x for x in self.results
                      if x.algo_name == self.bt_model_combo.currentText() and not x.error), None)
            if r is None or r.prev_close is None:
                QMessageBox.warning(self, "提示", "请先运行模型，再选择一个成功的模型回测。")
                return
            try:
                cost = float(self.bt_cost_edit.text())
            except ValueError:
                cost = 0.2
            horizon = getattr(self, "_last_horizon", 1)
            code = self.code_edit.text().strip()
            name = getattr(self, "_last_stock_name", "") or ""
            is_st = "ST" in (name or "").upper()
            # 相对大盘(沪深300)基准：按各段目标日期 asof 对齐指数收盘(取数失败则跳过，不报错)
            bench = None
            if self.data_source_combo.currentIndex() != 1 and HAS_AKSHARE:
                try:
                    segd = pd.to_datetime(pd.Series(r.test_dates).reset_index(drop=True))
                    seg_idx = segd.iloc[np.arange(0, len(segd), max(1, horizon))].reset_index(drop=True)
                    ixdf = StockDataFetcher.fetch_index_close()
                    merged = pd.merge_asof(pd.DataFrame({"date": seg_idx}).sort_values("date"),
                                           ixdf.sort_values("date"), on="date", direction="backward")
                    if merged["idx_px"].notna().sum() >= 2:
                        bench = merged["idx_px"].to_numpy(float)
                except Exception as e:
                    self._log(f"[回测] 沪深300基准对齐失败(跳过)：{e}")
            try:
                voltgt = float(self.bt_voltgt_edit.text()) / 100.0
            except ValueError:
                voltgt = 0.0
            try:
                stoploss = float(self.bt_stop_edit.text())
            except ValueError:
                stoploss = 0.0
            bt = backtest_directional(r.prev_close, r.y_test_true, r.y_test_pred,
                                      r.test_dates, horizon=horizon, cost_bps=cost,
                                      bar_info=r.bar_info, code=code, is_st=is_st, bench_close=bench,
                                      vol_target_annual=voltgt, stop_loss_pct=stoploss)
            if "error" in bt:
                QMessageBox.warning(self, "提示", bt["error"]); return
            self._draw_backtest(r.algo_name, bt)
            ex_b = f"，超大盘{bt['excess_vs_bench_pct']}%" if "excess_vs_bench_pct" in bt else ""
            self._oplog(f"回测：{r.algo_name}，成本{cost}%，策略{bt['total_return_pct']}% vs "
                        f"买入持有{bt['buyhold_return_pct']}%，超额{bt['excess_vs_buyhold_pct']}%{ex_b}"
                        f"；涨停买不进{bt['n_block_buy']}/停牌{bt['n_suspend']}/跌停卖不出{bt['n_stuck_sell']}段。")

        def _draw_backtest(self, algo, bt):
            beat_bh = bt["excess_vs_buyhold_pct"] > 0
            beat_bench = bt.get("excess_vs_bench_pct", None)
            verdict = "✅ 跑赢买入持有" if beat_bh else "❌ 没跑赢买入持有"
            if beat_bench is not None:
                verdict += "、" + ("✅跑赢大盘" if beat_bench > 0 else "❌没跑赢大盘")
            bench_txt = ""
            if "bench_return_pct" in bt:
                bench_txt = (f" · 沪深300 {bt['bench_return_pct']}%(年化{bt['bench_annual_pct']}%)"
                             f" · 超大盘 {bt['excess_vs_bench_pct']}%")
            # A 股制度真实化统计
            inst = (f" · 涨停买不进{bt['n_block_buy']}段/停牌{bt['n_suspend']}段/跌停卖不出{bt['n_stuck_sell']}段"
                    f"(涨跌停幅{bt['limit_pct']}%)")
            risk_txt = ""
            if bt.get("avg_position", 1) < 0.999:
                risk_txt += f" · 平均仓位{bt['avg_position']*100:.0f}%(波动率目标)"
            if bt.get("n_stop", 0) > 0:
                risk_txt += f" · 触发止损{bt['n_stop']}段"
            self.bt_stat.setText(
                f"【{algo}】周期{getattr(self,'_last_horizon',1)}日 · 往返成本{bt['cost_bps']}% · "
                f"交易{bt['n_trades']}/{bt['n_segments']}段 · 胜率{bt['win_rate_pct']}%{inst}{risk_txt}　||　"
                f"策略 {bt['total_return_pct']}%(年化{bt['annual_return_pct']}%) vs "
                f"买入持有 {bt['buyhold_return_pct']}%(年化{bt['buyhold_annual_pct']}%) · "
                f"超额 {bt['excess_vs_buyhold_pct']}%{bench_txt} · 最大回撤 {bt['max_drawdown_pct']}% · "
                f"夏普 {bt['sharpe']}　→　{verdict}")
            self.bt_figure.clear()
            ax = self.bt_figure.add_subplot(111)
            d = pd.to_datetime(bt["seg_dates"])
            ax.plot(d, bt["eq_strat"], label="跟随预测策略", color="#c0392b", linewidth=1.6)
            ax.plot(d, bt["eq_bh"], label="买入持有(该股)", color="#4a90d9", linewidth=1.4, linestyle="--")
            if "eq_bench" in bt:
                ax.plot(d, bt["eq_bench"], label="沪深300(大盘)", color="#e0a030", linewidth=1.3, linestyle=":")
            ax.axhline(1.0, color="#999", linewidth=0.8)
            ax.set_ylabel("净值(初始=1)"); ax.set_title(f"{algo} 方向性收益回测（扣 {bt['cost_bps']}% 往返成本 · 含A股涨跌停/停牌约束）")
            ax.legend(loc="best"); ax.grid(True, alpha=0.25)
            self.bt_figure.autofmt_xdate()
            self.bt_canvas.draw()

        # ---- 9.2.1g 未来预测走势标签页 ----
        def _build_forecast_tab(self) -> QWidget:
            """未来走势预测：对 1日/1周/2周/1月/2月/3月 分别直接预测，连成未来曲线并给出每股价格。
            图带缩放工具条(放大镜可放大看每日价格)。右侧列出各周期的预测价。"""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            top = QHBoxLayout()
            top.addWidget(QLabel("模型:"))
            self.fc_model_combo = QComboBox()
            self.fc_model_combo.addItems([n for n in ALGO_REGISTRY if ALGO_AVAILABILITY.get(n, True)])
            self.fc_model_combo.setCurrentText("Lasso")
            top.addWidget(self.fc_model_combo)
            top.addWidget(QLabel("范围:"))
            self.fc_range_combo = QComboBox()
            # 默认"未来一周·逐日"：下 1~5 个交易日(约下周一~周五)每天各一个预测
            self.fc_range_combo.addItems(["未来一周·逐日 (下1~5交易日)",
                                          "多周期 (1日/1周/2周/1月/2月/3月)"])
            self.fc_range_combo.setToolTip("『未来一周·逐日』：对下 1~5 个交易日分别直接预测，"
                                           "得到约周一~周五每天的预测价(法定节假日会顺延)。")
            top.addWidget(self.fc_range_combo)
            self.fc_btn = QPushButton("🔮 预测未来走势")
            self.fc_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.fc_btn.clicked.connect(self._on_run_forecast)
            top.addWidget(self.fc_btn)
            self.fc_hint = QLabel("默认『未来一周·逐日』：下1~5交易日(约周一~周五)每天各一个直接预测(不递归、不造假)。")
            self.fc_hint.setStyleSheet("color:#666;")
            top.addWidget(self.fc_hint, stretch=1)
            layout.addLayout(top)

            body = QHBoxLayout()
            left = QVBoxLayout()
            self.fc_figure = Figure(figsize=(7, 4.5))
            self.fc_canvas = FigureCanvas(self.fc_figure)
            left.addWidget(NavigationToolbar(self.fc_canvas, panel))   # 缩放/平移工具条
            left.addWidget(self.fc_canvas, stretch=1)
            body.addLayout(left, stretch=3)
            self.fc_side = QTextEdit(); self.fc_side.setReadOnly(True)
            self.fc_side.setFixedWidth(260)
            body.addWidget(self.fc_side)
            layout.addLayout(body, stretch=1)
            return panel

        def _on_run_forecast(self):
            self.fc_btn.setEnabled(False); self.fc_hint.setText("⏳ 正在训练多周期预测模型 ...")
            QApplication.setOverrideCursor(Qt.WaitCursor); self._prog_open(); QApplication.processEvents()
            try:
                use_synthetic = self.data_source_combo.currentIndex() == 1
                if use_synthetic:
                    df = StockDataFetcher.generate_synthetic_data(); code = "合成数据"
                else:
                    code = self.code_edit.text().strip()
                    start = self.start_date.date().toString("yyyyMMdd")
                    end = self.end_date.date().toString("yyyyMMdd")
                    self.fc_hint.setText(f"正在拉取 {code} 并训练多周期预测模型 ...")
                    QApplication.processEvents()
                    df = StockDataFetcher().fetch(code, start, end)
                    if self.chk_val.isChecked() or self.chk_idx.isChecked() or self.chk_mf.isChecked():
                        df, _ = StockDataFetcher.enrich(df, code, self.chk_val.isChecked(),
                                                        self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
                target_mode = "return" if self.target_combo.currentIndex() == 0 else "price"
                algo = self.fc_model_combo.currentText()
                weekly = self.fc_range_combo.currentIndex() == 0
                horizons = (1, 2, 3, 4, 5) if weekly else (1, 5, 10, 20, 40, 60)
                self.fc_hint.setText(f"正在用 {algo} 训练 {len(horizons)} 个"
                                     f"{'交易日' if weekly else '周期'}的直接预测模型（稍候）...")
                QApplication.processEvents()
                fc = forecast_curve(algo, df, target_mode=target_mode,
                                    horizons=horizons, progress_cb=self._log)
                self._draw_forecast(code, df, fc, weekly=weekly)
                self._oplog(f"未来预测：{code}，模型{algo}。")
            except Exception as e:
                QMessageBox.critical(self, "未来预测失败", str(e))
                self.fc_hint.setText("失败，详见弹窗。可先用合成数据看效果。")
            finally:
                QApplication.restoreOverrideCursor(); self._prog_close(); self.fc_btn.setEnabled(True)

        @staticmethod
        def _cn_weekday(d):
            """把日期转成中文星期(周一~周日)。"""
            return "周" + "一二三四五六日"[pd.Timestamp(d).weekday()]

        def _draw_forecast(self, code, df, fc, weekly=False):
            self.fc_figure.clear()
            ax = self.fc_figure.add_subplot(111)
            # 历史真实收盘：逐日模式只看最近 ~40 天(看得清)，多周期看 ~120 天
            hist = df.tail(40 if weekly else 120).reset_index(drop=True)
            hd = pd.to_datetime(hist["date"]); hc = hist["close"].to_numpy(float)
            ax.plot(hd, hc, color="#333", linewidth=1.2, marker=("o" if weekly else None),
                    markersize=3, label="历史真实收盘")
            last_date = pd.to_datetime(fc["last_date"]); last_close = fc["last_close"]

            # 未来锚点：每个周期 h -> 未来第 h 个『真实交易日』(优先真实日历，跳过周末+节假日)
            ok_points = [p for p in fc["points"] if "pred_close" in p]
            max_h = max((p["horizon"] for p in ok_points), default=1)
            fut_list = next_trading_days(last_date, max_h, progress_cb=self._log)
            fut_map = {h: fut_list[h - 1] for h in range(1, max_h + 1)}
            fut_dates = [last_date] + [fut_map[p["horizon"]] for p in ok_points]
            fut_prices = [last_close] + [p["pred_close"] for p in ok_points]
            lbl = "未来一周·逐日预测(每日直接预测)" if weekly else "未来预测(各周期直接预测)"
            ax.plot(fut_dates, fut_prices, color="#c0392b", linewidth=1.6, linestyle="--",
                    marker="o", markersize=5, label=lbl)
            # 约80%置信区间(基于历史波动率)：点预测必错，区间才诚实
            if any("lo" in p for p in ok_points):
                lo_band = [last_close] + [p.get("lo", p["pred_close"]) for p in ok_points]
                hi_band = [last_close] + [p.get("hi", p["pred_close"]) for p in ok_points]
                ax.fill_between(fut_dates, lo_band, hi_band, color="#c0392b", alpha=0.12,
                                label="约80%置信区间(历史波动)")
            if weekly:
                # 逐日模式：每个交易日都标注(周几 + 预测价 + 相对最新收盘涨跌)
                for p in ok_points:
                    fdate = fut_map[p["horizon"]]
                    chg = p["change_pct"]; col = "#c0392b" if chg >= 0 else "#1a9d5a"
                    ax.scatter([fdate], [p["pred_close"]], color=col, s=45, zorder=5)
                    ax.annotate(f"{self._cn_weekday(fdate)}\n{p['pred_close']:.2f}\n{chg:+.1f}%",
                                (fdate, p["pred_close"]), textcoords="offset points",
                                xytext=(0, 10), ha="center", color=col, fontsize=8)
            else:
                label_map = {20: ("1个月", "#1e8449"), 60: ("3个月", "#d35400")}
                for p in ok_points:
                    fdate = fut_map[p["horizon"]]
                    if p["horizon"] in label_map:
                        name, col = label_map[p["horizon"]]
                        ax.scatter([fdate], [p["pred_close"]], color=col, s=60, zorder=5)
                        ax.annotate(f"{name}\n{p['pred_close']:.2f}", (fdate, p["pred_close"]),
                                    textcoords="offset points", xytext=(6, 8), color=col, fontsize=9)
            ax.axvline(last_date, color="#aaa", linestyle=":", linewidth=1)
            ttl = "未来一周·逐日" if weekly else "1日~3个月"
            ax.set_ylabel("收盘价"); ax.set_title(f"{code} 未来走势预测（{fc['algo']}，{ttl}）")
            ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.25)
            self.fc_figure.autofmt_xdate(); self.fc_canvas.draw()

            # 右侧：每股预测价
            name_map = {1: "明日", 5: "1周后", 10: "2周后", 20: "1个月后", 40: "2个月后", 60: "3个月后"}
            title = "未来一周·逐日预测" if weekly else "未来多周期预测"
            rows = [f"<h3>{code} · {fc['algo']}</h3>",
                    f"<p>{title}</p>",
                    f"<p>最新收盘（{fc['last_date']}）：<b>{last_close:.2f} 元</b></p>",
                    "<table border=1 cellpadding=4 cellspacing=0 width=100%>"]
            if weekly:
                cal_note = "真实交易日" if _TRADE_CAL_CACHE["dates"] is not None else "日期(约)"
                rows.append(f"<tr bgcolor=#eef><th>交易日</th><th>{cal_note}</th><th>预测价</th>"
                            f"<th>较今涨跌</th><th>约80%区间</th></tr>")
                for p in fc["points"]:
                    if "pred_close" in p:
                        fdate = fut_map[p["horizon"]]
                        chg = p["change_pct"]; col = "#c0392b" if chg >= 0 else "#1a9d5a"
                        band = (f"{p['lo']:.2f}~{p['hi']:.2f}" if "lo" in p else "-")
                        rows.append(f"<tr><td>下{('一二三四五')[p['horizon']-1]}({self._cn_weekday(fdate)})</td>"
                                    f"<td>{fdate.strftime('%m-%d')}</td>"
                                    f"<td><b>{p['pred_close']:.2f}</b></td>"
                                    f"<td style='color:{col}'>{chg:+.2f}%</td>"
                                    f"<td style='color:#888;font-size:12px'>{band}</td></tr>")
                    else:
                        rows.append(f"<tr><td>下第{p['horizon']}日</td><td colspan=4>失败</td></tr>")
            else:
                rows.append("<tr bgcolor=#eef><th>时点</th><th>预测价(元)</th><th>涨跌</th></tr>")
                for p in fc["points"]:
                    nm = name_map.get(p["horizon"], f"{p['horizon']}日后")
                    if "pred_close" in p:
                        chg = p["change_pct"]; col = "#c0392b" if chg >= 0 else "#1a9d5a"
                        rows.append(f"<tr><td>{nm}</td><td><b>{p['pred_close']:.2f}</b></td>"
                                    f"<td style='color:{col}'>{chg:+.2f}%</td></tr>")
                    else:
                        rows.append(f"<tr><td>{nm}</td><td colspan=2>失败</td></tr>")
            rows.append("</table>")
            if weekly:
                ups = sum(1 for p in fc["points"] if p.get("change_pct", 0) > 0)
                downs = sum(1 for p in fc["points"] if p.get("change_pct", 1) < 0)
                rows.append(f"<p style='color:#555'>一周机械倾向：{ups} 天预测偏涨 / {downs} 天偏跌"
                            "（仅是模型按历史算出的方向，<b>非预言、非建议</b>）。</p>")
                if _TRADE_CAL_CACHE["dates"] is not None:
                    date_note = "日期取自 A 股<b>真实交易日历</b>(已跳过周末与法定节假日)"
                else:
                    date_note = "真实交易日历未取到，日期按<b>工作日近似</b>(遇法定节假日会有偏差)"
                rows.append(f"<p style='color:#c0392b'>⚠ {date_note}；"
                            "每天都是<b>独立直接预测</b>(不是拿前一天预测再滚动)，越往后越不可靠。"
                            "务必对照测试集 DA(方向准确率)判断可信度，绝不构成投资建议。</p>")
            else:
                rows.append("<p style='color:#c0392b'>⚠ 每个点是对应周期的<b>直接</b>预测，中间为插值；"
                            "预测越往后越不可靠，绝不构成投资建议。</p>")
            self.fc_side.setHtml("".join(rows))

        # ---- 9.2.2 数据配置区 ----
        def _build_data_group(self) -> QGroupBox:
            box = QGroupBox("数据配置")
            layout = QGridLayout(box)

            layout.addWidget(QLabel("股票代码:"), 0, 0)
            self.code_edit = QLineEdit("600519")           # 默认贵州茅台，仅作占位示例
            layout.addWidget(self.code_edit, 0, 1)

            layout.addWidget(QLabel("起始日期:"), 1, 0)
            self.start_date = QDateEdit(QDate(2020, 1, 1)); self.start_date.setCalendarPopup(True)
            layout.addWidget(self.start_date, 1, 1)

            layout.addWidget(QLabel("结束日期:"), 2, 0)
            self.end_date = QDateEdit(QDate.currentDate()); self.end_date.setCalendarPopup(True)
            layout.addWidget(self.end_date, 2, 1)

            layout.addWidget(QLabel("数据源:"), 3, 0)
            self.data_source_combo = QComboBox()
            has_real = HAS_AKSHARE or HAS_BAOSTOCK          # 有任一真实源即可用
            if has_real:
                srcs = []
                if HAS_AKSHARE:
                    srcs.append("东方财富")
                if HAS_BAOSTOCK:
                    srcs.append("baostock")
                real_label = "真实数据 (自动: " + "/".join(srcs) + ")"
            else:
                real_label = "真实数据 (未装 akshare/baostock，不可用)"
            self.data_source_combo.addItems([real_label, "合成数据 (离线演示/测试用)"])
            if not has_real:
                self.data_source_combo.setCurrentIndex(1)   # 没有真实源时默认合成数据，避免点击就报错
            layout.addWidget(self.data_source_combo, 3, 1)

            layout.addWidget(QLabel("预测目标:"), 4, 0)
            self.target_combo = QComboBox()
            self.target_combo.addItems(["涨跌幅 return (推荐,更诚实)", "价格水平 price"])
            self.target_combo.setToolTip("预测涨跌幅更平稳、可避免价格自相关导致的 R² 虚高；"
                                         "无论哪种，评估都会统一还原成价格并给出方向准确率。")
            layout.addWidget(self.target_combo, 4, 1)

            layout.addWidget(QLabel("预测周期:"), 5, 0)
            self.horizon_combo = QComboBox()
            # (显示文本, 对应的交易日数 horizon)
            self._horizon_options = [("第二天 (1日)", 1), ("3日", 3), ("一周 (5日)", 5),
                                     ("一个月 (20日)", 20), ("三个月 (60日)", 60)]
            self.horizon_combo.addItems([t for t, _ in self._horizon_options])
            self.horizon_combo.setToolTip("预测未来多少个交易日后的涨跌。方向准确率/基准对所有周期都已算对。")
            layout.addWidget(self.horizon_combo, 5, 1)

            self.glossary_btn = QPushButton("❓ 名词解释（PE/PB/DA 看不懂就点这）")
            self.glossary_btn.setStyleSheet("padding:5px;")
            self.glossary_btn.clicked.connect(self._show_glossary)
            layout.addWidget(self.glossary_btn, 6, 0, 1, 2)

            return box

        def _show_glossary(self):
            """弹出「名词解释」对话框(大白话解释金融+模型名词)。"""
            dlg = QDialog(self)
            dlg.setWindowTitle("名词解释（大白话）")
            dlg.resize(720, 640)
            lay = QVBoxLayout(dlg)
            view = QTextBrowser()
            view.setHtml(glossary_html())
            lay.addWidget(view)
            self._oplog("查看名词解释。")
            dlg.exec()

        # ---- 9.2.2b 外部数据接入区（真实来源，可勾选；仅真实数据模式生效）----
        def _build_external_group(self) -> QGroupBox:
            box = QGroupBox("外部数据接入 (真实来源，仅真实数据模式生效)")
            layout = QVBoxLayout(box)
            self.chk_weekly = QCheckBox("周/月/季 多周期趋势特征 (由日线计算，始终开启)")
            self.chk_weekly.setChecked(True); self.chk_weekly.setEnabled(False)
            self.chk_val = QCheckBox("财报估值：PE/PB/总市值 (日频 · 乐咕乐股或百度，自动适配)")
            self.chk_val.setChecked(True)
            self.chk_val.setToolTip("PE市盈率=股价是每股年利润的多少倍(越低越便宜)；PB市净率=股价对每股净资产的倍数；"
                                    "总市值=公司总价值。看不懂点下方「名词解释」。")
            self.chk_idx = QCheckBox("大盘环境：沪深300 涨跌/均线偏离 (东方财富)")
            self.chk_idx.setChecked(True)
            self.chk_idx.setToolTip("沪深300是A股最有代表性的300只大盘股指数，代表「大盘」整体强弱。")
            self.chk_mf = QCheckBox("主力资金流向：主力/超大单/大单净流入 + 进场/洗盘代理 (东方财富)")
            self.chk_mf.setChecked(True)
            self.chk_mf.setToolTip("主力=机构/大户的大单。净流入为正=大资金在买入，为负=在卖出。")
            self.chk_us = QCheckBox("隔夜美股：昨夜纳斯达克涨跌 (新浪，日期+1对齐无泄露)")
            self.chk_us.setChecked(True)
            self.chk_us.setToolTip("昨晚美股(纳斯达克)的涨跌。昨夜美股大跌，A股今天开盘常承压。")
            self.chk_nb = QCheckBox("北向资金：沪深股通每日净流入 (东方财富)")
            self.chk_nb.setChecked(True)
            self.chk_nb.setToolTip("通过沪深港通进来的外资，常被称为「聪明钱」，看外资态度。")
            self.chk_news = QCheckBox("个股新闻：真实标题+链接，仅在\"机器学习内部\"页展示 (东方财富)")
            self.chk_news.setChecked(True)
            for c in (self.chk_weekly, self.chk_val, self.chk_idx, self.chk_mf,
                      self.chk_us, self.chk_nb, self.chk_news):
                layout.addWidget(c)
            tip = QLabel("说明：勾选项失败会自动跳过该来源(特征就是没有)，绝不填假数据；新闻只作展示；"
                         "盘口买卖手为实时快照(无免费历史)，故用可回测的\"每日主力净流入\"作其历史等价物。")
            tip.setWordWrap(True); tip.setStyleSheet("color:#888; font-size:11px;")
            layout.addWidget(tip)
            return box

        # ---- 9.2.3 算法多选区（对应截图 B-1.1，按 3 类分组）----
        def _build_algo_group(self) -> QGroupBox:
            box = QGroupBox(f"多模型算法选择 (可多选，共 {len(ALGO_REGISTRY)} 个模型)")
            outer = QVBoxLayout(box)

            select_all_btn = QPushButton("全选/全不选")
            select_all_btn.clicked.connect(self._toggle_all_algos)
            outer.addWidget(select_all_btn)

            # 默认自动勾选几个"快且稳"的算法，开箱即用（含一个线性、一个树、一个梯度提升作对照）
            default_on = {"Lasso", "RF", "GBRT"}
            categories = ["基础与传统模型", "先进与前沿模型", "公式拟合/演化计算", "经典时序模型"]
            for cat in categories:
                cat_box = QGroupBox(cat)
                grid = QGridLayout(cat_box)
                algos_in_cat = [name for name, cls in ALGO_REGISTRY.items() if cls.category == cat]
                for i, name in enumerate(algos_in_cat):
                    cb = QCheckBox(name)
                    explain = ALGO_EXPLAIN.get(name, "")   # 悬停显示全称+大白话
                    if not ALGO_AVAILABILITY.get(name, True):
                        cb.setEnabled(False)
                        cb.setToolTip((explain + "\n" if explain else "")
                                      + "⚠ 缺少对应依赖库，暂不可用（详见文件顶部依赖说明；也可在「名词解释」查全部算法）")
                    else:
                        cb.setToolTip(explain + "（更多见「名词解释」）" if explain else "")
                        if name in default_on:
                            cb.setChecked(True)               # 默认勾选，无需用户从零开始
                    self.algo_checkboxes[name] = cb
                    grid.addWidget(cb, i // 4, i % 4)
                outer.addWidget(cat_box)
            return box

        def _toggle_all_algos(self):
            enabled_boxes = [cb for cb in self.algo_checkboxes.values() if cb.isEnabled()]
            all_checked = all(cb.isChecked() for cb in enabled_boxes) if enabled_boxes else False
            for cb in enabled_boxes:
                cb.setChecked(not all_checked)

        # ---- 9.2.4 功能选择区（对应截图 B-1.2：HPO方式 + 交叉验证）----
        def _build_function_group(self) -> QGroupBox:
            box = QGroupBox("功能选择")
            layout = QVBoxLayout(box)

            layout.addWidget(QLabel("① 启动超参数优化 / Enable HPO:"))
            hpo_row = QHBoxLayout()
            self.hpo_group = QButtonGroup(box)
            self.hpo_radios = {}
            for method in ["PSO", "GA", "FA", "SOA", "BO", "关闭"]:
                rb = QRadioButton(method + ("(Best)" if method == "BO" else ""))
                if method == "BO":
                    rb.setChecked(True)          # 默认推荐 BO，对应截图默认选中项
                self.hpo_group.addButton(rb)
                self.hpo_radios[method] = rb
                hpo_row.addWidget(rb)
            layout.addLayout(hpo_row)

            self.cv_checkbox = QCheckBox("② 启动 5 折交叉验证 CV")
            layout.addWidget(self.cv_checkbox)

            return box

        # ---- 9.2.5 评价指标勾选区（对应截图 B-1.3）----
        def _build_metric_group(self) -> QGroupBox:
            box = QGroupBox("评价指标核心配置 (系统推荐勾选 R², MAE, RMSE, DA方向准确率)")
            grid = QGridLayout(box)
            metric_names = ["R2", "MAE", "RMSE", "DA", "UP_P", "MAPE", "MSE", "CE", "NSE", "R", "KGE", "SMAPE", "WI", "SI"]
            default_on = {"R2", "MAE", "RMSE", "DA", "UP_P"}
            for i, name in enumerate(metric_names):
                cb = QCheckBox(name)
                cb.setChecked(name in default_on)
                cb.setToolTip(METRIC_EXPLAIN.get(name, "") + "（详见「名词解释」）")
                self.metric_checkboxes[name] = cb
                grid.addWidget(cb, i // 4, i % 4)
            return box

        # ---- 9.2.6 训练测试比例选择区（对应截图 B-1.4）----
        def _build_split_group(self) -> QGroupBox:
            box = QGroupBox("训练集比例 (剩余部分平分为 验证集/测试集，默认 7 : 1.5 : 1.5)")
            layout = QHBoxLayout(box)
            self.split_group = QButtonGroup(box)
            self.split_radios = {}
            for ratio_label, ratio_val in [("5:5", 0.5), ("6:4", 0.6), ("7:3", 0.7), ("8:2", 0.8)]:
                rb = QRadioButton(ratio_label)
                if ratio_val == 0.7:
                    rb.setChecked(True)
                self.split_group.addButton(rb)
                self.split_radios[ratio_val] = rb
                layout.addWidget(rb)
            return box

        # ---- 9.2.7 运行区 ----
        def _build_run_group(self) -> QGroupBox:
            box = QGroupBox()
            layout = QVBoxLayout(box)
            self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 0); self.progress_bar.hide()
            layout.addWidget(self.progress_bar)
            run_btn = QPushButton("运行选中配置模型 / Execute Combined Framework")
            run_btn.setStyleSheet("font-weight:bold; padding:8px; background:#2c6fbb; color:white;")
            run_btn.clicked.connect(self._on_run_clicked)
            layout.addWidget(run_btn)
            return box

        # ---- 9.3 点击"运行"后的完整处理流程 ----
        def _on_run_clicked(self):
            selected_algos = [name for name, cb in self.algo_checkboxes.items() if cb.isChecked()]
            if not selected_algos:
                QMessageBox.warning(self, "提示", "请至少勾选一个算法。")
                return

            # ---- 9.3.1 准备数据 ----
            try:
                use_synthetic = self.data_source_combo.currentIndex() == 1
                if use_synthetic:
                    self.raw_df = StockDataFetcher.generate_synthetic_data()
                    self._log("使用合成数据（离线演示模式）")
                else:
                    fetcher = StockDataFetcher()
                    code = self.code_edit.text().strip()
                    start = self.start_date.date().toString("yyyyMMdd")
                    end = self.end_date.date().toString("yyyyMMdd")
                    self._log(f"正在从 akshare 拉取 {code} [{start} ~ {end}] 的行情数据 ...")
                    self.raw_df = fetcher.fetch(code, start, end)
                    # 接入外部真实数据（财报估值 / 大盘环境 / 主力资金），失败自动跳过、绝不造假
                    if self.chk_val.isChecked() or self.chk_idx.isChecked() or self.chk_mf.isChecked():
                        self.raw_df, st = StockDataFetcher.enrich(
                            self.raw_df, code, self.chk_val.isChecked(),
                            self.chk_idx.isChecked(), self.chk_mf.isChecked(),
                            self.chk_us.isChecked(), self.chk_nb.isChecked())
                        for k, v in st.items():
                            self._log(f"[外部数据·{k}] {v}")
            except Exception as e:
                QMessageBox.critical(self, "数据获取失败", str(e))
                return

            # ---- 9.3.2 组装训练配置 ----
            train_ratio = next(r for r, rb in self.split_radios.items() if rb.isChecked())
            hpo_method = next(m for m, rb in self.hpo_radios.items() if rb.isChecked())
            metrics = [name for name, cb in self.metric_checkboxes.items() if cb.isChecked()] or ["R2", "MAE", "RMSE"]

            target_mode = "return" if self.target_combo.currentIndex() == 0 else "price"
            horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
            self._last_horizon = horizon              # 供"策略回测"页使用
            config = TrainConfig(
                window_size=20, horizon=horizon, train_ratio=train_ratio, target_mode=target_mode,
                use_cv=self.cv_checkbox.isChecked(), cv_folds=5,
                hpo_method=hpo_method, hpo_trials=15, metrics=metrics
            )

            # ---- 9.3.3 后台线程跑训练，避免界面卡死 ----
            src = "合成数据" if use_synthetic else f"真实数据 {self.code_edit.text().strip()}"
            ext = [n for n, cb in [("财报估值", self.chk_val), ("大盘", self.chk_idx),
                                   ("主力资金", self.chk_mf)] if cb.isChecked()]
            self._oplog(f"点击【运行】：数据={src}；预测目标={target_mode}；预测周期={horizon}日；"
                        f"训练比例={train_ratio}；HPO={hpo_method}；外部数据={'+'.join(ext) or '无'}；"
                        f"算法({len(selected_algos)})={', '.join(selected_algos)}")
            self.progress_bar.show()
            self._prog_open(f"⏳ 正在训练 {len(selected_algos)} 个模型，请稍候 ……")
            self._log(f"开始训练，共 {len(selected_algos)} 个模型：{', '.join(selected_algos)}")
            self.worker = TrainingWorker(config, self.raw_df, selected_algos)
            self.worker.progress_signal.connect(self._log)
            self.worker.finished_signal.connect(self._on_training_finished)
            self.worker.error_signal.connect(self._on_training_error)
            self.worker.start()

        # ---- 9.4 训练完成回调：更新表格 + 图表 ----
        def _on_training_finished(self, results: List[ModelResult]):
            self.progress_bar.hide(); self._prog_close()
            self.results = results
            self._log("全部模型训练完成。")
            self._update_result_table()
            self._update_chart()
            # 用成功训练的模型填充"策略回测"页的模型下拉（基准行也可回测，作为对照）
            self.bt_model_combo.clear()
            self.bt_model_combo.addItems([r.algo_name for r in results if not r.error])
            ok = sum(1 for r in results if not r.error)
            self._html_reco = self._build_recommendation_html(results)   # 供报告复用
            self.reco_view.setHtml(self._html_reco)
            self._oplog(f"训练完成：{ok} 个模型成功；结果见「指标结果表格」(每格=训练/验证/测试)。")
            self.tabs.setCurrentIndex(2)          # 直接跳到「指标结果表格」看误差对比

        # 常见特征的中文可读名（找不到就用原名）
        _FEATURE_LABELS = {
            "close": "收盘价", "open": "开盘", "high": "最高", "low": "最低", "volume": "成交量",
            "amount": "成交额", "turnover": "换手率", "ma5": "5日均线", "ma10": "10日均线",
            "ma20": "20日均线", "ma60": "60日均线", "macd_dif": "MACD-DIF", "macd_dea": "MACD-DEA",
            "macd_hist": "MACD柱", "rsi14": "RSI", "kdj_k": "KDJ-K", "kdj_d": "KDJ-D", "kdj_j": "KDJ-J",
            "boll_upper": "布林上轨", "boll_lower": "布林下轨", "return_1d": "日收益率",
            "volatility_10d": "10日波动率", "vol_ratio": "量比", "amplitude_feat": "日内振幅",
            "updown_streak": "连涨跌天数", "pos_in_range20": "20日区间位置",
            "ret_w1": "周涨跌", "ret_m1": "月涨跌", "ret_q1": "季涨跌",
            "val_pe_ttm": "市盈率TTM", "val_pb": "市净率", "val_total_mv": "总市值",
            "idx_ret_1d": "沪深300涨跌", "idx_madev20": "大盘均线偏离",
            "us_ret": "隔夜纳斯达克涨跌", "nb_net": "北向资金净流入",
            "mf_main_net": "主力净流入", "mf_main_pct": "主力净占比", "mf_xl_net": "超大单净额",
            "mf_l_net": "大单净额", "mf_streak": "主力连续进出天数", "mf_cum5": "主力5日累计",
            "mf_price_div": "资金-价格背离",
        }

        def _build_recommendation_html(self, results):
            base = next((r for r in results if r.algo_name == "总是涨(方向基准)"), None)
            up_rate = base.metrics.get("DA") if (base and base.metrics) else None
            models = [r for r in results if not r.error
                      and r.algo_name not in ("Naive(前值)", "总是涨(方向基准)") and r.metrics]
            if not models:
                return "<p>没有成功训练的模型。</p>"
            best = max(models, key=lambda r: (r.metrics.get("DA") or 0))
            da = best.metrics.get("DA"); up_p = best.metrics.get("UP_P")
            n_te = int(len(best.y_test_true)) if best.y_test_true is not None else 0
            sig = da_significance(da, n_te)        # 最佳 DA 是否统计显著>50%(而非运气/挑出来的)
            n_models = len(models)
            # 判定：DA 要①明显高于"总是涨"基准(+3%且>52%) ②统计显著，才推荐；否则诚实劝退
            good = (up_rate is not None and da is not None and da >= up_rate + 3.0
                    and da >= 52.0 and sig["sig"])
            # 多重比较偏差：一次比了 n_models 个模型再挑最好的，最好那个的 DA 天然偏乐观
            mc_note = ""
            if n_models >= 3:
                mc_note = (f"<p style='color:#c0392b;font-size:13px'>⚠ <b>多重比较偏差提醒</b>："
                           f"本次同时比了 {n_models} 个模型再挑出 DA 最高的——"
                           f"「挑出来的最好」本身就偏乐观(像掷一堆骰子只报最大点)。"
                           f"务必用「预测跟踪」页做<b>样本外</b>验证，别只信这个回测数字。</p>")
            if good:
                verdict = (f"<div style='background:#f2fbf2;border:2px solid #1e8449;padding:8px'>"
                           f"<b style='color:#1e8449'>🎯 建议用 {best.algo_name} 做本次预测</b>　"
                           f"（测试方向准确率 DA <b>{da}%</b>，明显高于「总是涨」基准 {up_rate}%、"
                           f"且{sig['text']}；预测上涨时精确率 UP_P {up_p}%）。仍仅供参考，不构成投资建议。{mc_note}</div>")
            else:
                why_bad = (f"最高的 {best.algo_name} DA={da}%" +
                           (f"，虽高于基准但{sig['text']}" if (up_rate and da and da >= up_rate + 3)
                            else f"，未明显超基准 {up_rate}%"))
                verdict = (f"<div style='background:#fff4f4;border:2px solid #c0392b;padding:8px'>"
                           f"<b style='color:#c0392b'>⚠️ 本次没有模型可靠地优于「总是涨」基准</b>"
                           f"（{why_bad}）——<b>不建议据此预测，别信任何单一预测。</b>"
                           f"换股票/换周期/多试几个模型再看。{mc_note}</div>")
            # 预测依据①：模型最看重的真实输入
            fi = best.feature_importance
            if fi:
                lab = lambda k: self._FEATURE_LABELS.get(k, k)
                imp = "、".join(f"{lab(k)} {v}%" for k, v in fi[:6])
                why1 = (f"<p><b>📌 {best.algo_name} 最看重的输入（占比）：</b>{imp}。"
                        "<span style='color:#888'>这是模型<b>依据</b>的输入，不是涨跌的真正原因——"
                        "市场多因素驱动，无法单一归因。</span></p>")
            else:
                why1 = ("<p><b>📌 预测依据：</b>该模型(如 SVR-rbf / 深度学习)无标准特征重要性，"
                        "无法给出输入权重；想看依据可用 <b>RF / Lasso / XGBoost</b> 等模型。</p>")
            # 预测依据②：当前真实数据现状 + 来源
            why2 = self._data_context_html()
            return verdict + why1 + why2

        def _data_context_html(self):
            df = getattr(self, "raw_df", None)
            if df is None or "close" not in df.columns:
                return ""
            code = self.code_edit.text().strip()
            def last(col):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    return float(s.iloc[-1]) if len(s) else None
                return None
            bits = []
            close = pd.to_numeric(df["close"], errors="coerce")
            if len(close) > 21:
                bits.append(f"近1月涨跌 {close.iloc[-1]/close.iloc[-21]*100-100:+.1f}%")
            pe, mf, idxr = last("val_pe_ttm"), last("mf_main_net"), last("idx_ret_1d")
            if pe is not None:
                bits.append(f"市盈率TTM {pe:.1f}" if pe > 0 else "市盈率 亏损")
            if mf is not None:
                bits.append(f"最新主力净流入 {mf/1e4:.0f}万" + ("(流入)" if mf >= 0 else "(流出)"))
            if idxr is not None:
                bits.append(f"沪深300最新 {idxr*100:+.2f}%")
            if not bits:
                return ""
            links = self._source_links_html(code) if code else ""
            return (f"<p><b>📊 当前真实数据现状（模型环境）：</b>" + "；".join(bits) +
                    "。<span style='color:#888'>数据来源见下方溯源链接，可逐一核对。</span></p>" +
                    ("<details><summary>展开数据溯源链接</summary>" + links + "</details>" if links else ""))

        def _on_training_error(self, msg: str):
            self.progress_bar.hide(); self._prog_close()
            QMessageBox.critical(self, "训练出错", msg)

        # ---- 9.5 结果表格 ----
        def _update_result_table(self):
            valid_results = [r for r in self.results if not r.error]
            metric_names = list(valid_results[0].metrics.keys()) if valid_results else []

            # 每个指标一列，单元格里显示"训练/验证/测试"三个值，直观看出是否过拟合
            self.result_table.setColumnCount(1 + len(metric_names))
            self.result_table.setHorizontalHeaderLabels(
                ["算法"] + [f"{m}(训/验/测)" for m in metric_names])
            self.result_table.setRowCount(len(self.results))

            def _fmt(d, k):
                if not d or k not in d:
                    return "-"
                v = d[k]
                return f"{v:.3f}" if isinstance(v, float) else str(v)

            for row, r in enumerate(self.results):
                self.result_table.setItem(row, 0, QTableWidgetItem(r.algo_name))
                if r.error:
                    self.result_table.setItem(row, 1, QTableWidgetItem(f"失败: {r.error}"))
                    continue
                for col, m in enumerate(metric_names, start=1):
                    cell = f"{_fmt(r.metrics_train, m)} / {_fmt(r.metrics_val, m)} / {_fmt(r.metrics, m)}"
                    self.result_table.setItem(row, col, QTableWidgetItem(cell))
            self.result_table.resizeColumnsToContents()

        # ---- 9.6 预测对比图：把所有成功的模型画在同一张图上，横轴是日期，纵轴是收盘价 ----
        def _update_chart(self):
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            valid_results = [r for r in self.results if not r.error]
            if not valid_results:
                self.canvas.draw()
                return

            # 真实值只需要画一次（所有模型的测试集真实值是一样的）
            first = valid_results[0]
            ax.plot(first.test_dates, first.y_test_true, label="真实值 (Actual)",
                     color="black", linewidth=2)

            for r in valid_results:
                ax.plot(r.test_dates, r.y_test_pred, label=f"{r.algo_name} 预测", alpha=0.8)

            ax.set_xlabel("日期"); ax.set_ylabel("收盘价")
            ax.set_title("测试集：预测值 vs 真实值（这条线『很贴合』是假象，别被骗！）")
            ax.legend(loc="best", fontsize=8)
            # 关键诚实提示：价格预测的线总是紧贴真实值，是因为"明天价≈今天价"，连Naive基准都贴合，
            # 并不代表模型厉害。真本事看 DA 方向准确率，不是看这条线贴不贴。
            ax.text(0.5, -0.22,
                    "⚠️ 别被这张图骗了：明天价≈今天价，所以连『Naive前值』基准都紧贴真实值——线贴合≠预测准。\n"
                    "真本事请看『指标结果表格』的 DA 方向准确率(≈50%就是没用)，和『策略回测』能不能赚钱。",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#c0392b")
            self.figure.subplots_adjust(bottom=0.30)
            self.figure.autofmt_xdate()
            self.canvas.draw()

        # ---- 9.7 日志输出 ----
        def _log(self, msg: str):
            self.log_box.append(msg)

        def _oplog(self, msg: str):
            """记录用户操作到"操作日志"页，带时间戳。"""
            ts = dt.datetime.now().strftime("%H:%M:%S")
            self.oplog_box.append(f"[{ts}] {msg}")

        @contextlib.contextmanager
        def _busy(self, label=None, msg="⏳ 正在运行，请稍候 ...", btn=None):
            """统一的"加载中"提示：等待光标 + 该页提示文字 + 按钮临时禁用；结束自动恢复。
            用法： with self._busy(self.kline_hint, "正在拉取行情 ...", self.kline_btn): ...heavy..."""
            old = label.text() if label is not None else None
            if label is not None:
                label.setText(msg)
            if btn is not None:
                btn.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._prog_open(msg)                      # 明显的模态"进程界面"
            QApplication.processEvents()             # 先把"加载中"画出来再干活
            try:
                yield
            finally:
                QApplication.restoreOverrideCursor()
                self._prog_close()
                if btn is not None:
                    btn.setEnabled(True)
                # 提示文字不强制还原(各处会自己设完成后的文案)；仅当调用方没改时保留原文
                if label is not None and label.text() == msg and old is not None:
                    label.setText(old)

        # ---- 明显可见的"进程界面"(模态忙碌弹窗) ----
        def _prog_open(self, msg="⏳ 正在运行，请稍候 ……"):
            """弹出一个居中的模态"运行中"窗口，比等待光标醒目得多。
            range=(0,0) 表示忙碌指示(来回滚动的进度条)。同步耗时操作里它至少会"弹出来"，
            后台线程操作里它还会持续滚动。重复调用只更新文字，不叠开多个。"""
            try:
                dlg = getattr(self, "_prog_dlg", None)
                if dlg is None:
                    dlg = QProgressDialog(msg, None, 0, 0, self)   # 无取消按钮
                    dlg.setWindowTitle("请稍候")
                    dlg.setWindowModality(Qt.ApplicationModal)
                    dlg.setCancelButton(None)
                    dlg.setMinimumDuration(0)       # 立刻显示，不等 4 秒
                    dlg.setAutoClose(False); dlg.setAutoReset(False)
                    dlg.setMinimumWidth(320)
                    self._prog_dlg = dlg
                dlg.setLabelText(msg)
                dlg.show(); dlg.raise_()
                QApplication.processEvents()
            except Exception:
                pass                                # 进程窗只是提示，绝不因它中断主流程

        def _prog_close(self):
            try:
                dlg = getattr(self, "_prog_dlg", None)
                if dlg is not None:
                    dlg.reset(); dlg.hide()
                QApplication.processEvents()
            except Exception:
                pass


# ==================== 第十部分：可编程 API（供脚本 / 其它 AI 调用） ====================
def run_experiment(code: str = "600519",
                   start: str = "20200101",
                   end: Optional[str] = None,
                   algos: Optional[List[str]] = None,
                   synthetic: bool = False,
                   hpo: str = "关闭",
                   hpo_trials: int = 15,
                   train_ratio: float = 0.7,
                   window: int = 20,
                   horizon: int = 1,
                   target_mode: str = "return",
                   use_cv: bool = False,
                   metrics: Optional[List[str]] = None,
                   add_naive_baseline: bool = True,
                   forecast: bool = False,
                   use_valuation: bool = True,
                   use_index: bool = True,
                   use_fundflow: bool = True,
                   backtest: bool = False,
                   cost_bps: float = 0.2,
                   progress_cb: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    一个"无界面、纯函数式"的完整实验入口，专为脚本化 / 被其它程序或 AI 调用而设计。
    只需传参数进来，就返回一个可直接 json.dumps 的结果字典，不依赖任何 GUI。

    返回结构（示例）：
        {
          "config": {...实际使用的配置...},
          "n_samples": {"train": 500, "test": 200},
          "results": [{"algo": "RF", "metrics": {...}, "error": null}, ...],
          "forecast": [{"algo": "RF", "last_date": "...", "pred_close": ...}, ...] 或 None
        }

    典型用法（其它 AI / 脚本可直接照抄）：
        from stock_predictor import run_experiment
        out = run_experiment(code="600519", start="20200101", algos=["RF", "XGBoost"], forecast=True)
        print(out["results"])
    """
    log = progress_cb or (lambda msg: None)
    algos = algos or ["RF", "SVR", "GBRT"]
    metrics = metrics or ["R2", "MAE", "RMSE", "DA", "UP_P"]
    end = end or dt.date.today().strftime("%Y%m%d")

    # ---- 1) 取数据：真实(东方财富/baostock 自动容错) 或 合成 ----
    if synthetic or not (HAS_AKSHARE or HAS_BAOSTOCK):
        if not synthetic:
            log("[提示] 未安装 akshare/baostock，自动改用合成数据。"
                "要用真实数据请先 pip install akshare baostock 并去掉 --synthetic。")
        raw_df = StockDataFetcher.generate_synthetic_data()
        data_source = "synthetic"
    else:
        log(f"正在拉取真实行情 {code} [{start}~{end}]（东方财富优先，失败自动切 baostock）...")
        raw_df = StockDataFetcher().fetch(code, start, end)
        data_source = f"akshare:{code}"
        if use_valuation or use_index or use_fundflow:
            raw_df, est = StockDataFetcher.enrich(raw_df, code, use_valuation, use_index, use_fundflow)
            for k, v in est.items():
                log(f"[外部数据·{k}] {v}")

    # ---- 2) 组装配置并训练 ----
    config = TrainConfig(window_size=window, horizon=horizon, train_ratio=train_ratio,
                         target_mode=target_mode, use_cv=use_cv, hpo_method=hpo,
                         hpo_trials=hpo_trials, metrics=metrics,
                         add_naive_baseline=add_naive_baseline)
    pipeline = TrainingPipeline(config)
    results = pipeline.run_batch(algos, raw_df, progress_cb=log)

    # ---- 3) 可选：用最新窗口向前预测"未来" ----
    forecasts = None
    if forecast:
        forecasts = []
        for algo in algos:
            try:
                forecasts.append(pipeline.predict_next(algo, raw_df, progress_cb=log))
            except Exception as e:
                forecasts.append({"algo": algo, "error": str(e)})

    n_train, n_test = 0, 0
    for r in results:
        if not r.error:
            n_test = len(r.y_test_true)
            break

    # ---- 4) 可选：方向性收益回测（扣手续费，和买入持有对比）----
    def _bt(r):
        if not backtest or r.error or r.prev_close is None:
            return None
        b = backtest_directional(r.prev_close, r.y_test_true, r.y_test_pred,
                                 r.test_dates, horizon=horizon, cost_bps=cost_bps,
                                 bar_info=r.bar_info, code=(None if synthetic else code))
        return {k: v for k, v in b.items() if k not in ("eq_strat", "eq_bh", "eq_bench", "seg_dates")}

    return {
        "data_source": data_source,
        "config": {
            "code": code, "start": start, "end": end, "algos": algos,
            "hpo": hpo, "hpo_trials": hpo_trials, "train_ratio": train_ratio,
            "window": window, "horizon": horizon, "target_mode": target_mode,
            "use_cv": use_cv, "metrics": metrics, "add_naive_baseline": add_naive_baseline,
        },
        "n_samples": {"test": n_test},
        "results": [
            {"algo": r.algo_name, "metrics": r.metrics,
             "metrics_train": r.metrics_train, "metrics_val": r.metrics_val,
             "formula": r.formula, "cv_metrics": r.cv_metrics,
             "error": r.error, "backtest": _bt(r)}
            for r in results
        ],
        "forecast": forecasts,
    }


# ==================== 第十一部分：程序入口（GUI / 命令行 双模式） ====================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="一键式 A 股预测研究软件。不带参数 = 启动图形界面；带 --cli = 无界面运行并输出 JSON。")
    p.add_argument("--cli", action="store_true", help="强制无界面(命令行)模式运行")
    p.add_argument("--gui", action="store_true", help="强制启动图形界面")
    p.add_argument("--code", default="600519", help="股票代码，如 600519(贵州茅台)")
    p.add_argument("--start", default="20200101", help="起始日期 YYYYMMDD")
    p.add_argument("--end", default=None, help="结束日期 YYYYMMDD，默认今天")
    p.add_argument("--synthetic", action="store_true", help="使用离线合成数据(不联网、无需 akshare)")
    p.add_argument("--algos", default="RF,SVR,GBRT", help="逗号分隔的算法名，如 RF,XGBoost,LSTM")
    p.add_argument("--hpo", default="关闭", choices=["PSO", "GA", "FA", "SOA", "BO", "关闭"],
                   help="超参数优化方式，默认关闭(最快)")
    p.add_argument("--hpo-trials", type=int, default=15, help="超参数寻优迭代次数")
    p.add_argument("--ratio", type=float, default=0.7, help="训练集比例，默认 0.7")
    p.add_argument("--window", type=int, default=20, help="滑动窗口天数")
    p.add_argument("--horizon", type=int, default=1, help="预测未来第几天(方向准确率/朴素基准要求为 1)")
    p.add_argument("--target", default="return", choices=["return", "price"],
                   help="预测目标: return=涨跌幅(推荐,更诚实) / price=价格水平")
    p.add_argument("--cv", action="store_true", help="启用 5 折交叉验证")
    p.add_argument("--metrics", default="R2,MAE,RMSE,DA,UP_P", help="逗号分隔的评价指标")
    p.add_argument("--no-naive", action="store_true", help="不加入朴素基准(不建议)")
    p.add_argument("--no-valuation", action="store_true", help="不接入财报估值(PE/PB/PS/市值)")
    p.add_argument("--no-index", action="store_true", help="不接入大盘环境(沪深300)")
    p.add_argument("--no-fundflow", action="store_true", help="不接入主力资金流向(主力/大单净流入+进场/洗盘代理)")
    p.add_argument("--forecast", action="store_true", help="额外输出用最新窗口预测的未来收盘价")
    p.add_argument("--week", action="store_true",
                   help="额外输出『未来一周·逐日』：下1~5交易日(约周一~周五)每天各一个直接预测")
    p.add_argument("--backtest", action="store_true", help="方向性收益回测(按预测涨跌做多/空仓，扣手续费)")
    p.add_argument("--cost", type=float, default=0.2, help="回测往返成本(%%)：佣金+印花税+滑点，默认 0.2")
    p.add_argument("--json", default=None, help="把结果 JSON 写入指定文件路径")
    p.add_argument("--snapshot", default=None,
                   help="无界面抓取一次实时盘口并记录到 monitor_log/(供 Windows 任务计划定时调用)，如 --snapshot 002418")
    p.add_argument("--verify", action="store_true",
                   help="无界面自动验证到期预测(拉真实股价对比)，供定时任务每日调用")
    return p


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    """命令行模式：跑一次实验，打印精简结果表，并按需输出 JSON。"""
    out = run_experiment(
        code=args.code, start=args.start, end=args.end,
        algos=[a.strip() for a in args.algos.split(",") if a.strip()],
        synthetic=args.synthetic, hpo=args.hpo, hpo_trials=args.hpo_trials,
        train_ratio=args.ratio, window=args.window, horizon=args.horizon,
        target_mode=args.target,
        use_cv=args.cv, metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
        add_naive_baseline=not args.no_naive, forecast=args.forecast,
        use_valuation=not args.no_valuation, use_index=not args.no_index,
        use_fundflow=not args.no_fundflow,
        backtest=args.backtest, cost_bps=args.cost,
        progress_cb=print,
    )

    # ---- 打印一张精简的结果表 ----
    print("\n" + "=" * 78)
    print(f"数据源: {out['data_source']}    测试样本数: {out['n_samples']['test']}")
    print("-" * 78)
    metric_cols = out["config"]["metrics"]
    header = f"{'算法':<14}" + "".join(f"{m:>10}" for m in metric_cols)
    print(header)
    print("-" * 78)
    for r in out["results"]:
        if r["error"]:
            print(f"{r['algo']:<14}  失败: {r['error']}")
            continue
        row = f"{r['algo']:<14}" + "".join(f"{r['metrics'].get(m, ''):>10}" for m in metric_cols)
        print(row)
    print("=" * 78)
    print("提示：任何模型都应明显优于 Naive(前值) 才算真的有用；DA(方向准确率) 需显著 > 50%。")

    if out["forecast"]:
        print("\n【向前预测：最新窗口 -> 下一交易日收盘价】")
        for f in out["forecast"]:
            if f.get("error"):
                print(f"  {f['algo']:<12} 失败: {f['error']}")
            else:
                print(f"  {f['algo']:<12} 最后({f['last_date']})收盘={f['last_close']} "
                      f"-> 预测={f['pred_close']}  ({f['pred_change_pct']:+}%)")
        print("  ⚠ 单日单模型的点预测仅供研究参考，绝不构成任何投资建议。")

    if getattr(args, "week", False):
        print("\n【未来一周·逐日预测：下 1~5 个交易日(约周一~周五)每天各一个直接预测】")
        if args.synthetic:
            wdf = StockDataFetcher.generate_synthetic_data()
        else:
            wdf = StockDataFetcher().fetch(args.code, args.start, args.end)
        algo1 = [a.strip() for a in args.algos.split(",") if a.strip()][0]
        wfc = forecast_curve(algo1, wdf, target_mode=args.target, horizons=(1, 2, 3, 4, 5),
                             progress_cb=print)
        last_d = pd.to_datetime(wfc["last_date"])
        wfut = next_trading_days(last_d, 5, progress_cb=print)
        cal_ok = _TRADE_CAL_CACHE["dates"] is not None
        print(f"  模型={algo1}  最新收盘({wfc['last_date']})={wfc['last_close']}")
        for pt in wfc["points"]:
            if pt.get("error"):
                print(f"    下第{pt['horizon']}日  失败: {pt['error']}"); continue
            fd = wfut[pt["horizon"] - 1]
            wd = "周" + "一二三四五六日"[fd.weekday()]
            print(f"    下{('一二三四五')[pt['horizon']-1]}({wd} {fd.strftime('%Y-%m-%d')})  "
                  f"预测={pt['pred_close']}  ({pt['change_pct']:+}%)")
        cal_msg = "日期取自真实交易日历(已跳过周末+节假日)" if cal_ok else "真实日历未取到，日期按工作日近似(节假日可能不准)"
        print(f"  ⚠ {cal_msg}；每天独立直接预测(非递归)，越往后越不可靠，绝不构成投资建议。")

    if args.backtest:
        print(f"\n【方向性收益回测：按预测涨跌做多/空仓，往返成本 {args.cost}%，含A股涨跌停/停牌约束】")
        print(f"{'算法':<14}{'策略收益%':>10}{'年化%':>9}{'买入持有%':>10}{'超额%':>9}"
              f"{'最大回撤%':>10}{'胜率%':>8}{'夏普':>8}{'涨停挡':>7}{'停牌':>6}{'跌停锁':>7}")
        for r in out["results"]:
            b = r.get("backtest")
            if not b:
                continue
            print(f"{r['algo']:<14}{b['total_return_pct']:>10}{b['annual_return_pct']:>9}"
                  f"{b['buyhold_return_pct']:>10}{b['excess_vs_buyhold_pct']:>9}"
                  f"{b['max_drawdown_pct']:>10}{b['win_rate_pct']:>8}{b['sharpe']:>8}"
                  f"{b.get('n_block_buy',0):>7}{b.get('n_suspend',0):>6}{b.get('n_stuck_sell',0):>7}")
        print("  → 只有策略明显跑赢\"买入持有\"(超额为正)，才说明预测涨跌真的能赚钱。仍不构成投资建议。")
        print("  说明：涨停挡=预测该买但一字涨停买不进的段数；停牌=基准日停牌买不进；跌停锁=目标日一字跌停卖不出(收益偏乐观)。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n结果 JSON 已写入: {args.json}")
    return out


def main():
    """
    程序主入口，三种情形：
        1) 带任意命令行参数(如 --cli/--code/...)         -> 无界面命令行模式，输出结果表/JSON
        2) 不带参数且已安装 PySide6                       -> 启动图形界面
        3) 不带参数且未装 PySide6                         -> 退化为一次合成数据的最小演示
    """
    # Windows 默认控制台是 GBK 编码，打印中文/emoji(如 ⚠) 会抛 UnicodeEncodeError，
    # 这里统一把标准输出/错误切到 UTF-8，避免命令行模式下因为一个字符崩掉。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")     # Python 3.7+
        except Exception:
            pass

    args = _build_arg_parser().parse_args()

    # --snapshot：无界面抓一次盘口就退出（供 Windows 任务计划定时调用，不需开 GUI）
    if args.snapshot:
        code = args.snapshot.strip()
        try:
            q = StockDataFetcher.fetch_realtime(code)
            path = write_snapshot(code, q)
            trading, status = is_trading_now()
            print(f"[snapshot] {code} 现价={q.get('price')} 涨跌={q.get('pct')}% [{status}] -> {path}")
        except Exception as e:
            print(f"[snapshot] {code} 抓取失败: {e}")
        return

    # --verify：无界面自动验证到期预测（供定时任务每日调用）
    if args.verify:
        try:
            n = verify_predictions(progress_cb=print)
            print(f"[verify] 本次验证 {n} 条到期预测。")
        except Exception as e:
            print(f"[verify] 失败: {e}")
        return

    # 判断是否走命令行模式：显式 --cli，或用户传了任何非默认业务参数而没要求 --gui
    only_defaults = (len(sys.argv) == 1)
    if args.gui or (only_defaults and HAS_PYSIDE6 and not args.cli):
        if not HAS_PYSIDE6:
            print("未检测到 PySide6，无法启动图形界面。请执行: pip install PySide6")
            return
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    if not only_defaults or args.cli:
        run_cli(args)
        return

    # 情形 3：没参数、没 GUI，跑个最小演示，方便先验证逻辑
    print("=" * 70)
    print("未检测到 PySide6，且未提供命令行参数。")
    print("下面演示一次命令行版最小流程（离线合成数据）。")
    print("要看全部用法： python stock_predictor.py -h")
    print("=" * 70)
    args.synthetic = True
    args.algos = "RF,SVR,GBRT"
    run_cli(args)


if __name__ == "__main__":
    main()
