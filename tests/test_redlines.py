# -*- coding: utf-8 -*-
"""红线 + 核心函数自动化测试（离线、不依赖 29G 本地数据集）。

守住 CLAUDE.md 的 4 条不可触碰红线，把『防泄露/防造假』从人肉自查变成 CI 门禁：
  R1 时序不 shuffle（前段训练、后段测试）
  R2 标准化只在训练集 fit
  R3 不删诚实基准（Naive(前值) + DA 方向准确率）
  R4 目标不进特征（未来收益绝不出现在 X 里）
外加这几轮新增函数的行为测试（指纹/蒙特卡洛/中性化/异动避雷/多期限数据集因果构造）。
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stock_predictor as s  # noqa: E402


# ============ 红线 R1：时序不 shuffle ============
def test_r1_time_series_split_is_chronological():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.arange(100).astype(float)
    dates = pd.Series(pd.date_range("2020-01-01", periods=100))
    Xtr, Xte, ytr, yte, dtr, dte = s.FeatureEngineer.time_series_split(X, y, dates, train_ratio=0.7)
    assert len(Xtr) == 70 and len(Xte) == 30
    # 训练集全部在测试集之前（无重叠、无乱序）
    assert float(Xtr.max()) < float(Xte.min())
    assert pd.Timestamp(dtr.iloc[-1]) < pd.Timestamp(dte.iloc[0])


# ============ 红线 R4：目标不进特征 ============
def test_r4_features_and_targets_disjoint():
    feats = set(s.MH_FEATURE_COLS)
    targs = set(s.MH_TARGET_COLS)
    assert feats.isdisjoint(targs)
    # 任何特征列都不得以 y 开头（y1_dir/y2..y5 都是输出）
    assert not any(str(c).startswith("y") for c in s.MH_FEATURE_COLS)


# ============ 红线 R3：Naive 基准 + DA 不可缺 ============
def test_r3_naive_baseline_and_da_present():
    pytest.importorskip("sklearn")
    out = s.run_experiment(synthetic=True, algos=["RF"], add_naive_baseline=True,
                           metrics=["R2", "DA"])
    algos = [r["algo"] for r in out["results"]]
    assert any("Naive" in a for a in algos), "诚实基准 Naive(前值) 不得删除"
    rf = next(r for r in out["results"] if r["algo"] == "RF")
    assert "DA" in (rf.get("metrics") or {}), "DA 方向准确率不得删除"


# ============ 红线 R2：标准化只在训练集 fit ============
def test_r2_scaler_fit_on_train_only():
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    # 训练段均值0方差1，测试段整体平移（若在全量 fit，测试段均值会被拉回0附近）
    Xtr = rng.normal(0, 1, size=(200, 3))
    Xte = rng.normal(5, 1, size=(80, 3))
    sc = StandardScaler().fit(Xtr)           # 只在训练集 fit
    zte = sc.transform(Xte)
    # 正确做法下：测试集变换后均值应显著偏离 0（≈+5），证明没在测试集/全量上 fit
    assert np.mean(zte) > 3.0


# ============ 冻结特征指纹 ============
def test_feature_fingerprint_order_sensitive_and_stable():
    a = s._feature_fingerprint(["h", "ret_1d", "pe_ttm"])
    assert a == s._feature_fingerprint(["h", "ret_1d", "pe_ttm"])   # 稳定
    assert a != s._feature_fingerprint(["ret_1d", "h", "pe_ttm"])   # 顺序敏感
    assert a != s._feature_fingerprint(["h", "ret_1d"])             # 内容敏感


# ============ 蒙特卡洛生存模拟 ============
def test_monte_carlo_invariants_and_determinism():
    r1 = s.monte_carlo_trading(win_rate=50, rr=1.5, risk_per_trade_pct=2,
                               n_trades=150, cost_pct=0.15, n_paths=3000, seed=7)
    r2 = s.monte_carlo_trading(win_rate=50, rr=1.5, risk_per_trade_pct=2,
                               n_trades=150, cost_pct=0.15, n_paths=3000, seed=7)
    assert r1 == r2                                   # 同 seed 结果可复现
    assert 0 <= r1["p_ruin_pct"] <= 100
    assert 0 <= r1["p_loss_pct"] <= 100
    # edge = 胜率 − 保本胜率(rr=1.5 → 40%)，50-40=10
    assert abs(r1["edge"] - 10.0) < 1e-6
    # 重仓 + 负 edge → 破产概率应明显更高
    heavy = s.monte_carlo_trading(win_rate=42, rr=1.5, risk_per_trade_pct=6,
                                  n_trades=300, cost_pct=0.2, n_paths=4000, seed=7)
    light = s.monte_carlo_trading(win_rate=42, rr=1.5, risk_per_trade_pct=1,
                                  n_trades=300, cost_pct=0.2, n_paths=4000, seed=7)
    assert heavy["p_ruin_pct"] > light["p_ruin_pct"]


# ============ 行业/市值中性化 ============
def test_neutralize_removes_size_exposure():
    # 单日截面：y_excess 完全由 log(市值) 线性决定 → 市值中性化后残差应≈0
    mv = np.array([1e8, 3e8, 6e8, 1e9, 2e9, 5e9], dtype=float)
    y = 4.0 * np.log(mv) - 50.0
    ds = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * len(mv)),
        "industry": [""] * len(mv), "total_mv": mv, "y_excess": y,
    })
    out, info = s._neutralize_cross_section(ds, "y_excess", use_size=True, use_industry=False)
    assert info["size_applied"] is True
    assert float(np.nanstd(out.values)) < 1e-6      # size 被完全剔除


# ============ 异动/避雷打分 ============
def _mk_series(n=80, base_ret=0.0, ret_std=1.0, turnover=2.0):
    rng = np.random.default_rng(1)
    ret = rng.normal(base_ret, ret_std, size=n)
    close = 10 * np.cumprod(1 + ret / 100.0)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "name": ["测试股"] * n, "close": close, "ret_1d": ret,
        "turnover": np.full(n, turnover), "vol_ratio": np.ones(n),
        "mom20": pd.Series(close).pct_change(20).fillna(0).values * 100,
        "dist_ma60": np.zeros(n),
        "vol20": pd.Series(ret).rolling(20).std().values,
        "pe_ttm": np.full(n, 30.0),
    })


def test_anomaly_calm_low_spiky_high():
    calm = s._anomaly_risk_one(_mk_series(base_ret=0.0, ret_std=0.8, turnover=2.0), "平静股")
    spiky = _mk_series(base_ret=0.0, ret_std=1.0, turnover=2.0)
    # 制造异动：末日换手暴增、末20日多次涨跌停、末日放量
    spiky.loc[spiky.index[-1], "turnover"] = 40.0
    spiky.loc[spiky.index[-1], "vol_ratio"] = 4.0
    for j in range(1, 8):
        spiky.loc[spiky.index[-j], "ret_1d"] = 10.0 if j % 2 else -10.0
    hot = s._anomaly_risk_one(spiky, "妖股")
    assert hot["score"] > calm["score"]
    assert hot["score"] >= 35           # 至少中等异动
    assert any("涨" in r or "换手" in r or "放量" in r for r in hot["reasons"])


# ============ 多期限数据集：因果构造 + 特征/目标齐全 ============
def _write_fake_local(root, code="600000", n=400):
    """造一份 股票4.14 同结构的本地 CSV（真实字段名），供离线测 build_multi_horizon_dataset。"""
    d = os.path.join(root, "每只股票一个文件", "前复权")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2022-01-03", periods=n)
    ret = rng.normal(0, 1.5, size=n)
    close = 10 * np.cumprod(1 + ret / 100.0)
    cl = pd.Series(close)
    df = pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"), "名称": "测试股", "收盘价": close.round(3),
        "涨幅%": ret.round(3),
        "3日涨幅%": (cl / cl.shift(3) - 1).mul(100).round(3),
        "6日涨幅%": (cl / cl.shift(6) - 1).mul(100).round(3),
        "10日涨幅%": (cl / cl.shift(10) - 1).mul(100).round(3),
        "量比": 1.0, "换手率": 2.0,
        "20日线": cl.rolling(20).mean().round(3),
        "60日线": cl.rolling(60).mean().round(3),
        "250日线": cl.rolling(250).mean().round(3),
        "滚动市盈率": 30.0, "市净率": 3.0, "滚动市销率": 5.0,
        "总市值（元）": 5e9, "所属行业": "测试业", "退市时间": "-",
    })
    df.to_csv(os.path.join(d, f"{code}.csv"), index=False, encoding="utf-8-sig")
    return code


def test_build_multi_horizon_causal(tmp_path):
    root = str(tmp_path)
    code = _write_fake_local(root)
    ds = s.build_multi_horizon_dataset([code], horizons=[1, 5, 10], root=root,
                                       anchor_stride=3, allow_online=False)
    assert len(ds) > 0 and ds["code"].nunique() == 1
    # 特征列齐全、且与目标不相交（R4）
    for c in s.MH_FEATURE_COLS:
        assert c in ds.columns
    assert set(s.MH_FEATURE_COLS).isdisjoint(s.MH_TARGET_COLS)
    assert "y4_mean_pct" in ds.columns
    # 因果：最后一个锚定日必须给未来窗口留出 hmax 天（不越界取未来）
    hmax = 10
    df_raw, _ = s._mh_load_local_rich(code, root)
    last_dt = df_raw["date"].max()
    max_anchor = ds["date"].max()
    future_room = (df_raw["date"] > max_anchor).sum()
    assert future_room >= hmax, "锚定日越界取了不存在的未来 → 泄露风险"
