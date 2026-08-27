#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
                    一键式 A 股预测研究软件  |  stock_predictor.py
========================================================================================

【软件说明】
    本软件是一个单文件（single-file）的 A 股股票走势预测与可视化平台，功能包括：
        1. 数据获取层   —— 通过 akshare（免费开源库）拉取 A 股历史行情数据，支持本地缓存
        2. 特征工程层   —— 常用技术指标计算 + 滑动窗口样本构造 + 时间序列切分
        3. 模型层       —— 34 种机器学习/深度学习/公式拟合算法，统一接口，可插拔注册表
        4. 超参数优化层 —— PSO / GA / FA / SOA / 贝叶斯优化(BO，默认推荐) 五种寻优方式
        5. 训练评估层   —— 训练/测试集切分、5折交叉验证、10 种评价指标
        6. 可视化 GUI 层 —— 基于 PySide6，仿照科研平台的多算法勾选式操作界面

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
    34 个算法中，绝大多数（RF/XGBoost/LightGBM/SVR/GPR/LSTM/Transformer 等）都有
    公认的标准 Python 实现，本文件直接调用对应库，结果是"真实"的。
    但其中 3 个算法（KStar、M5Rules、GEP、MEP）在 Python 生态中没有和 WEKA / 专业
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
        QTabWidget, QProgressBar, QMessageBox, QScrollArea, QSplitter, QTextEdit
    )
    from PySide6.QtCore import Qt, QThread, Signal, QDate
    from PySide6.QtGui import QFont
    matplotlib.use("QtAgg")     # 让 matplotlib 使用 Qt 后端，方便嵌入 PySide6 界面
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
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
    train_ratio: float = 0.7         # 训练集比例，默认 7:3（对应截图默认选中项）
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
        if use_cache and os.path.exists(cache_path):
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
                status["valuation"] = f"已接入(乐咕乐股, {len(v)}行)"
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
        return df.reset_index(drop=True), status

    @staticmethod
    def _fetch_fundflow(code: str, bypass_proxy: bool = True) -> pd.DataFrame:
        """
        真实主力资金流向(日频)：akshare stock_individual_fund_flow，数据源=东方财富。
        这是"主力/大单是买是卖"的**历史可回测**版本，也是"买卖手力量"的历史等价物
        （实时盘口委买委卖是快照、无免费历史，不能用于回测）。
        取：主力净流入净额/净占比、超大单/大单/中单/小单净额。
        """
        market = "sh" if code.startswith("6") else ("bj" if code.startswith(("4", "8")) else "sz")
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            f = ak.stock_individual_fund_flow(stock=code, market=market)
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

    @staticmethod
    def _fetch_valuation(code: str, bypass_proxy: bool = True) -> pd.DataFrame:
        """真实财报估值(日频)：akshare stock_a_indicator_lg，数据源=乐咕乐股 legulegu.com。
        字段：pe/pe_ttm/pb/ps/ps_ttm/dv_ratio/dv_ttm/total_mv。取其中有代表性的几项。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            v = ak.stock_a_indicator_lg(symbol=code)
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

    @staticmethod
    def _fetch_index(bypass_proxy: bool = True, symbol: str = "sh000300") -> pd.DataFrame:
        """真实大盘环境：akshare stock_zh_index_daily(沪深300)。产出当日涨跌与相对20日均线偏离。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            ix = ak.stock_zh_index_daily(symbol=symbol)
        ix["date"] = pd.to_datetime(ix["date"]).astype("datetime64[ns]")
        ix = ix.sort_values("date")
        ix["idx_ret_1d"] = ix["close"].pct_change()
        ix["idx_madev20"] = ix["close"] / (ix["close"].rolling(20).mean() + 1e-9) - 1
        return ix[["date", "idx_ret_1d", "idx_madev20"]].dropna()

    @staticmethod
    def fetch_news(code: str, limit: int = 15, bypass_proxy: bool = True) -> pd.DataFrame:
        """真实个股新闻(仅供展示/近期参考)：akshare stock_news_em，数据源=东方财富。
        注意：新闻接口只覆盖近期，且把文本转成可靠的历史情绪分数需要 NLP 模型；为避免臆造，
        本软件**不**把新闻情绪当作历史训练特征，只在界面展示真实标题与链接。"""
        ctx = _no_proxy() if bypass_proxy else contextlib.nullcontext()
        with ctx:
            nd = ak.stock_news_em(symbol=code)
        nd = nd.rename(columns={"新闻标题": "title", "发布时间": "time",
                                "文章来源": "source", "新闻链接": "url"})
        cols = [c for c in ["time", "title", "source", "url"] if c in nd.columns]
        return nd[cols].head(limit)

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
        df = df.dropna().reset_index(drop=True)      # 技术指标开头会有 NaN（如 MA60 需要60天），一并丢弃

        if feature_cols is None:
            # 默认使用除日期外的全部数值列作为特征
            feature_cols = [c for c in df.columns if c != "date"]
        self.feature_cols = feature_cols

        values = df[feature_cols].values
        close = df["close"].values
        dates = df["date"]

        X, y, sample_dates, prev_close, close_target = [], [], [], [], []
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

        self.prev_close_ = np.array(prev_close)
        self.close_target_ = np.array(close_target)
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


@dataclass
class PreparedData:
    """一次数据准备的产物。y 处于"目标空间"(price 或 return)，未缩放；
    评估/画图统一用价格空间：真实值=close_*，预测值由各模型输出还原得到。"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray                # 目标空间(price 或 return)，未缩放
    y_test: np.ndarray
    d_train: pd.Series
    d_test: pd.Series
    prev_close_train: np.ndarray       # 每个样本"基准日"收盘价
    prev_close_test: np.ndarray
    close_train: np.ndarray            # 每个样本"真实未来"收盘价(价格空间)
    close_test: np.ndarray


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

    # ---------- 8.1 数据准备：技术指标 -> 样本 -> 标准化 -> 时间序列切分 ----------
    def prepare_data(self, raw_df: pd.DataFrame) -> PreparedData:
        X, y, dates = self.fe.build_supervised_samples(raw_df)
        pc, ct = self.fe.prev_close_, self.fe.close_target_
        n = len(X)
        split = int(n * self.config.train_ratio)
        self.fe.fit_scale(X[:split], y[:split])                  # 只在训练集上 fit，防止数据泄漏
        return PreparedData(
            X_train=self.fe.transform(X[:split]), X_test=self.fe.transform(X[split:]),
            y_train=y[:split], y_test=y[split:],
            d_train=dates[:split], d_test=dates[split:].reset_index(drop=True),
            prev_close_train=pc[:split], prev_close_test=pc[split:],
            close_train=ct[:split], close_test=ct[split:],
        )

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

            # ---- 8.2.4 测试集预测 -> 逆缩放回目标空间 -> 统一还原成价格 ----
            pred_target = self.fe.inverse_y(model.predict(X_test))
            pred_price = self._to_price(pred_target, pdata.prev_close_test)
            metrics = Metrics.calc_all(close_test, pred_price, self.config.metrics)
            self._apply_dir(metrics, close_test, pred_price, pdata.prev_close_test)

            # ---- 8.2.5 可选：5折交叉验证（衡量稳定性；仅算非方向类指标，方向类需价格还原故略）----
            cv_metrics = None
            if self.config.use_cv:
                log(f"[{algo_name}] 正在做 {self.config.cv_folds} 折交叉验证 ...")
                cv_metrics = self._cross_validate(model_cls, best_params, X_train, y_train_s,
                                                  y_train, mk)

            formula = model.get_formula() if hasattr(model, "get_formula") else None

            return ModelResult(algo_name=algo_name, metrics=metrics, y_test_true=close_test,
                                y_test_pred=pred_price, test_dates=pdata.d_test,
                                cv_metrics=cv_metrics, formula=formula)
        except Exception as e:
            log(f"[{algo_name}] 训练失败：{e}")
            return ModelResult(algo_name=algo_name, metrics={}, y_test_true=close_test,
                                y_test_pred=np.full_like(close_test, np.nan, dtype=float),
                                test_dates=pdata.d_test, error=str(e))

    # ---------- 8.3 K 折交叉验证（时间序列友好版：仍然按时间顺序滚动切分，不随机打乱）----------
    def _cross_validate(self, model_cls, best_params, X, y_scaled, y_raw,
                        model_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        model_kwargs = model_kwargs or {}
        # 方向类指标(DA/UP_P)需要"价格还原"才有意义，而 CV 在目标空间(price/return)内评估，故此处略去
        cv_metric_names = [m for m in self.config.metrics if m not in ("DA", "UP_P")]
        n = len(X)
        fold_size = n // self.config.cv_folds
        fold_metrics = []
        for k in range(self.config.cv_folds):
            start, end = k * fold_size, (k + 1) * fold_size if k < self.config.cv_folds - 1 else n
            if start >= end - 1:
                continue
            X_tr = np.concatenate([X[:start], X[end:]], axis=0)
            y_tr = np.concatenate([y_scaled[:start], y_scaled[end:]], axis=0)
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
        naive_price = pdata.prev_close_test.astype(float)
        metrics = Metrics.calc_all(pdata.close_test, naive_price, self.config.metrics)
        self._apply_dir(metrics, pdata.close_test, naive_price, pdata.prev_close_test)
        return ModelResult(algo_name="Naive(前值)", metrics=metrics, y_test_true=pdata.close_test,
                           y_test_pred=naive_price, test_dates=pdata.d_test)

    def _alwaysup_baseline(self, pdata: PreparedData) -> ModelResult:
        """
        "总是猜涨"方向基准：每天都预测上涨。它的方向准确率(DA)与上涨精确率(UP_P) 就等于
        测试集里"上涨日占比"。**任何模型的 DA/UP_P 只有明显高于这一行，才算真的有涨跌判别力。**
        （价格类指标对本行无意义，请只看 DA / UP_P。）
        """
        up_price = pdata.prev_close_test.astype(float) * 1.001    # 恒定"涨一点"，方向恒为涨
        metrics = Metrics.calc_all(pdata.close_test, up_price, self.config.metrics)
        self._apply_dir(metrics, pdata.close_test, up_price, pdata.prev_close_test)
        return ModelResult(algo_name="总是涨(方向基准)", metrics=metrics,
                           y_test_true=pdata.close_test, y_test_pred=up_price,
                           test_dates=pdata.d_test)

    # ---------- 8.5 批量运行多个算法 ----------
    def run_batch(self, algo_names: List[str], raw_df: pd.DataFrame,
                  progress_cb: Optional[Callable[[str], None]] = None) -> List[ModelResult]:
        log = progress_cb or (lambda msg: None)
        pdata = self.prepare_data(raw_df)
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

        # ---- 9.2.1 整体布局搭建 ----
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)

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

            self.result_table = QTableWidget()
            self.tabs.addTab(self.result_table, "指标结果表格")

            self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
            self.tabs.addTab(self.log_box, "运行日志")

            # 最后一个标签页：机器学习内部（真实趋势 + 输入来源 + 训练/测试数据集，透明可查）
            self.tabs.addTab(self._build_mldata_tab(), "机器学习内部")

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
            self.kline_btn = QPushButton("📈 获取/刷新 K 线图")
            self.kline_btn.setStyleSheet("font-weight:bold; padding:6px;")
            self.kline_btn.clicked.connect(self._on_fetch_kline)
            top.addWidget(self.kline_btn)
            self.kline_hint = QLabel("提示：先在左上角填好股票代码/日期/数据源，再点这里。红涨绿跌。")
            self.kline_hint.setStyleSheet("color:#666;")
            top.addWidget(self.kline_hint, stretch=1)
            layout.addLayout(top)

            self.kline_figure = Figure(figsize=(8, 5))
            self.kline_canvas = FigureCanvas(self.kline_figure)
            layout.addWidget(self.kline_canvas, stretch=1)
            return panel

        # ---- 9.2.1c 点击"获取 K 线"后：拉数据并画图 ----
        def _on_fetch_kline(self):
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
                self._draw_kline(df, title)
                self.kline_hint.setText(f"共 {len(df)} 个交易日。红涨绿跌，含 MA5/10/20 与成交量。")
            except Exception as e:
                QMessageBox.critical(self, "K 线获取失败", str(e))
                self.kline_hint.setText("获取失败，详见弹窗。若是代理/网络问题，可先用合成数据。")

        # ---- 9.2.1d 画蜡烛图（价格 + 均线 + 成交量）----
        def _draw_kline(self, df: pd.DataFrame, title: str):
            self.kline_figure.clear()
            gs = self.kline_figure.add_gridspec(4, 1, hspace=0.08)
            ax_p = self.kline_figure.add_subplot(gs[0:3, 0])
            ax_v = self.kline_figure.add_subplot(gs[3, 0], sharex=ax_p)

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
            self.kline_canvas.draw()

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
            # 中：输入的真实来源说明
            self.mldata_srcbox = QTextEdit(); self.mldata_srcbox.setReadOnly(True)
            split.addWidget(self.mldata_srcbox)
            # 下：数据集预览表
            self.mldata_table = QTableWidget()
            split.addWidget(self.mldata_table)
            split.setSizes([300, 240, 300])
            layout.addWidget(split, stretch=1)
            return panel

        def _on_build_mldata(self):
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
                            self.chk_idx.isChecked(), self.chk_mf.isChecked())
                    if self.chk_news.isChecked():
                        try:
                            news = StockDataFetcher.fetch_news(code)
                        except Exception as e:
                            enrich_status["news"] = f"跳过(取数失败: {e})"

                target_mode = "return" if self.target_combo.currentIndex() == 0 else "price"
                horizon = self._horizon_options[self.horizon_combo.currentIndex()][1]
                train_ratio = next(r for r, rb in self.split_radios.items() if rb.isChecked())
                fe = FeatureEngineer(window_size=20, horizon=horizon, target_mode=target_mode)
                X, y, sample_dates = fe.build_supervised_samples(df)
                if len(X) < 10:
                    QMessageBox.warning(self, "提示", "样本太少，无法透视，请拉长日期区间。")
                    return
                split_idx = int(len(X) * train_ratio)

                self._draw_mldata_trend(code, sample_dates, fe.close_target_, split_idx, target_mode)
                self._fill_mldata_source(fe, target_mode, len(X), split_idx, enrich_status, news)
                self._fill_mldata_table(fe, y, sample_dates, split_idx, target_mode)
                self.mldata_hint.setText(
                    f"{code}：预测周期 {horizon}日；共 {len(X)} 条样本，训练 {split_idx} / 测试 {len(X)-split_idx}，"
                    f"分界日期 {pd.to_datetime(sample_dates.iloc[split_idx]).strftime('%Y-%m-%d')}。")
            except Exception as e:
                QMessageBox.critical(self, "数据透视失败", str(e))
                self.mldata_hint.setText("失败，详见弹窗。可先用合成数据看效果。")

        def _draw_mldata_trend(self, code, sample_dates, close_target, split_idx, target_mode):
            self.mldata_figure.clear()
            ax = self.mldata_figure.add_subplot(111)
            x = np.arange(len(close_target))
            ax.plot(x, close_target, color="#333", linewidth=1.0, label="真实收盘价")
            # 训练区(蓝)/测试区(橙)背景，直观证明"训练在前、测试在后，不泄露"
            ax.axvspan(0, split_idx, color="#4a90d9", alpha=0.10)
            ax.axvspan(split_idx, len(x), color="#e08a2c", alpha=0.12)
            ax.axvline(split_idx, color="#c0392b", linestyle="--", linewidth=1.2)
            ax.text(split_idx / 2, ax.get_ylim()[1], "训练集", ha="center", va="top",
                    color="#2c6fbb", fontsize=10)
            ax.text((split_idx + len(x)) / 2, ax.get_ylim()[1], "测试集", ha="center", va="top",
                    color="#d35400", fontsize=10)
            idx = np.linspace(0, len(x) - 1, min(8, len(x))).astype(int)
            ax.set_xticks(x[idx])
            ax.set_xticklabels([pd.to_datetime(sample_dates.iloc[i]).strftime("%y-%m-%d") for i in idx],
                               rotation=30, fontsize=8)
            ax.set_ylabel("收盘价")
            ax.set_title(f"{code} 真实趋势与训练/测试划分（预测目标：{'涨跌幅' if target_mode=='return' else '价格'}）")
            ax.grid(True, alpha=0.25)
            self.mldata_figure.tight_layout()
            self.mldata_canvas.draw()

        def _fill_mldata_source(self, fe, target_mode, n_samples, split_idx,
                                enrich_status=None, news=None):
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
                (has("val_"), "财报估值(日频)", f"乐咕乐股 legulegu.com · {enrich_status.get('valuation','')}",
                 "val_pe_ttm/val_pb/val_ps_ttm/val_dv_ttm/val_total_mv"),
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
                ns = enrich_status.get("news", "未获取")
                news_html = (f"<p><b>个股新闻：</b>{ns}。真实来源=东方财富 stock_news_em；"
                             "仅展示、不作历史训练特征(避免臆造情绪分)。</p>")

            html = f"""
            <h3>输入的真实来源（当前共 {n_feat} 个特征 × 窗口20天 = 每条样本 {n_feat*20} 维）</h3>
            <p>下表状态按"特征列里是否真有对应列"如实判定；⛔ 表示该来源本次未接入(取数失败/未勾选)，
            <b>软件绝不用假数据填充</b>。</p>
            <table border=1 cellpadding=4 cellspacing=0 width=100%>
              <tr bgcolor=#eef><th>输入类别</th><th>真实来源</th><th>具体特征</th></tr>{rows_html}
            </table>
            {news_html}
            <p><b>关于"主力意图"与"买卖手"（诚实说明）：</b>
            "主力进场/退场/洗盘"没有权威标签，本软件不臆造"主力预言"，而是接入<b>真实的每日主力资金净流入</b>
            并派生代理特征(连续净流入天数、资金-价格背离)喂给模型，由模型自己学。
            实时盘口"买1-5/卖1-5(买卖手)"是<b>快照、无免费历史序列</b>，用于历史回测会造成泄露或造假，
            因此用可回测的"每日主力/大单净额"作为其历史等价物。</p>
            <p style="color:#2c6fbb"><b>为什么不会泄露未来数据：</b>
            ① 所有特征(含估值/大盘)都按日期"向后对齐"，只用当天及以前的已知值；
            ② 严格按时间顺序切分——训练集在前 {split_idx} 条、测试集在后 {n_samples-split_idx} 条，绝不打乱；
            ③ 标准化器只在训练集上 fit，再套用到测试集。</p>
            """
            self.mldata_srcbox.setHtml(html)

        def _fill_mldata_table(self, fe, y, sample_dates, split_idx, target_mode):
            n = len(y)
            tgt_label = "目标(涨跌幅%)" if target_mode == "return" else "目标(收盘价)"
            headers = ["序号", "目标日期", "基准日收盘", tgt_label, "所属集合"]
            self.mldata_table.setColumnCount(len(headers))
            self.mldata_table.setHorizontalHeaderLabels(headers)
            # 样本可能上千条：为流畅只展示前 60 + 后 60，并在中间放一行省略提示
            if n <= 130:
                show_idx = list(range(n))
            else:
                show_idx = list(range(60)) + [-1] + list(range(n - 60, n))
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
                grp = "训练集" if i < split_idx else "测试集"
                vals = [str(i), d, f"{pc:.2f}", f"{tv:.3f}", grp]
                for c, v in enumerate(vals):
                    self.mldata_table.setItem(row, c, QTableWidgetItem(v))
            self.mldata_table.resizeColumnsToContents()

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

            return box

        # ---- 9.2.2b 外部数据接入区（真实来源，可勾选；仅真实数据模式生效）----
        def _build_external_group(self) -> QGroupBox:
            box = QGroupBox("外部数据接入 (真实来源，仅真实数据模式生效)")
            layout = QVBoxLayout(box)
            self.chk_weekly = QCheckBox("周/月/季 多周期趋势特征 (由日线计算，始终开启)")
            self.chk_weekly.setChecked(True); self.chk_weekly.setEnabled(False)
            self.chk_val = QCheckBox("财报估值：PE/PB/PS/股息率/总市值 (日频 · 乐咕乐股 legulegu.com)")
            self.chk_val.setChecked(True)
            self.chk_idx = QCheckBox("大盘环境：沪深300 涨跌/均线偏离 (东方财富)")
            self.chk_idx.setChecked(True)
            self.chk_mf = QCheckBox("主力资金流向：主力/超大单/大单净流入 + 进场/洗盘代理 (东方财富)")
            self.chk_mf.setChecked(True)
            self.chk_news = QCheckBox("个股新闻：真实标题+链接，仅在\"机器学习内部\"页展示 (东方财富)")
            self.chk_news.setChecked(True)
            for c in (self.chk_weekly, self.chk_val, self.chk_idx, self.chk_mf, self.chk_news):
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

            categories = ["基础与传统模型", "先进与前沿模型", "公式拟合/演化计算", "经典时序模型"]
            for cat in categories:
                cat_box = QGroupBox(cat)
                grid = QGridLayout(cat_box)
                algos_in_cat = [name for name, cls in ALGO_REGISTRY.items() if cls.category == cat]
                for i, name in enumerate(algos_in_cat):
                    cb = QCheckBox(name)
                    if not ALGO_AVAILABILITY.get(name, True):
                        cb.setEnabled(False)
                        cb.setToolTip("缺少对应依赖库，暂不可用（详见文件顶部依赖说明）")
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
                self.metric_checkboxes[name] = cb
                grid.addWidget(cb, i // 4, i % 4)
            return box

        # ---- 9.2.6 训练测试比例选择区（对应截图 B-1.4）----
        def _build_split_group(self) -> QGroupBox:
            box = QGroupBox("训练集与测试集比例选择 (默认 7:3)")
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
                            self.chk_idx.isChecked(), self.chk_mf.isChecked())
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
            config = TrainConfig(
                window_size=20, horizon=horizon, train_ratio=train_ratio, target_mode=target_mode,
                use_cv=self.cv_checkbox.isChecked(), cv_folds=5,
                hpo_method=hpo_method, hpo_trials=15, metrics=metrics
            )

            # ---- 9.3.3 后台线程跑训练，避免界面卡死 ----
            self.progress_bar.show()
            self._log(f"开始训练，共 {len(selected_algos)} 个模型：{', '.join(selected_algos)}")
            self.worker = TrainingWorker(config, self.raw_df, selected_algos)
            self.worker.progress_signal.connect(self._log)
            self.worker.finished_signal.connect(self._on_training_finished)
            self.worker.error_signal.connect(self._on_training_error)
            self.worker.start()

        # ---- 9.4 训练完成回调：更新表格 + 图表 ----
        def _on_training_finished(self, results: List[ModelResult]):
            self.progress_bar.hide()
            self.results = results
            self._log("全部模型训练完成。")
            self._update_result_table()
            self._update_chart()
            self.tabs.setCurrentIndex(0)

        def _on_training_error(self, msg: str):
            self.progress_bar.hide()
            QMessageBox.critical(self, "训练出错", msg)

        # ---- 9.5 结果表格 ----
        def _update_result_table(self):
            valid_results = [r for r in self.results if not r.error]
            metric_names = list(valid_results[0].metrics.keys()) if valid_results else []

            self.result_table.setColumnCount(1 + len(metric_names))
            self.result_table.setHorizontalHeaderLabels(["算法"] + metric_names)
            self.result_table.setRowCount(len(self.results))

            for row, r in enumerate(self.results):
                self.result_table.setItem(row, 0, QTableWidgetItem(r.algo_name))
                if r.error:
                    self.result_table.setItem(row, 1, QTableWidgetItem(f"失败: {r.error}"))
                    continue
                for col, m in enumerate(metric_names, start=1):
                    self.result_table.setItem(row, col, QTableWidgetItem(str(r.metrics.get(m, ""))))
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
            ax.set_title("测试集：预测值 vs 真实值")
            ax.legend(loc="best", fontsize=8)
            self.figure.autofmt_xdate()
            self.canvas.draw()

        # ---- 9.7 日志输出 ----
        def _log(self, msg: str):
            self.log_box.append(msg)


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
            {"algo": r.algo_name, "metrics": r.metrics, "formula": r.formula,
             "cv_metrics": r.cv_metrics, "error": r.error}
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
    p.add_argument("--json", default=None, help="把结果 JSON 写入指定文件路径")
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
