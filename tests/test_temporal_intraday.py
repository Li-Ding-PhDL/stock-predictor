"""时间边界、调参和盘中快照回归测试；全部离线，数据均为合成。"""
import csv
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stock_predictor as s


def test_purged_split_removes_unfinished_labels_and_keeps_same_day_together():
    ds = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"]),
        "label_end_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08", "2024-01-05"]),
    })
    tr, te = s._purged_date_split(ds, "2024-01-04")
    assert list(tr.index) == [0]
    assert list(te.index) == [3]
    with pytest.raises(ValueError, match="日期"):
        s._purged_date_split(ds.assign(label_end_date=pd.NaT), "2024-01-04")


def test_hpo_scalers_fit_internal_train_only_and_group_dates():
    dates = pd.bdate_range("2024-01-01", periods=20)
    anchors = pd.Series(dates[:15].repeat(2))
    ends = pd.Series(dates[3:18].repeat(2))
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = X.ravel().copy()
    cutoff = dates[12]
    expected = np.flatnonzero(((anchors < cutoff) & (ends < cutoff)).to_numpy())
    X[anchors >= cutoff] += 1000
    y[anchors >= cutoff] += 1000
    xt, yt, xv, yv = s._temporal_hpo_data(X, y, anchors, ends, scale_y=True)
    assert len(xt) == len(expected) == 18
    assert len(xv) == 6
    assert np.allclose(xt.mean(axis=0), 0)
    assert np.isclose(yt.mean(), 0)
    assert xv.mean() > 100 and yv.mean() > 100


def _prepared_pipeline(monkeypatch, horizon=5):
    pipe = s.TrainingPipeline(s.TrainConfig(horizon=horizon, train_ratio=0.6, val_ratio=0.2))
    calendar = pd.bdate_range("2024-01-01", periods=100 + horizon)
    anchors = pd.Series(calendar[:100])
    ends = pd.Series(calendar[horizon:100 + horizon])
    X = np.arange(200, dtype=float).reshape(100, 2)
    y = np.arange(100, dtype=float) / 10000

    def build(_df):
        pipe.fe.anchor_dates_ = anchors
        pipe.fe.prev_close_ = np.full(100, 10.0)
        pipe.fe.close_target_ = 10 * (1 + y)
        pipe.fe.bar_info_ = {"base_vol": np.arange(100)}
        return X, y, ends

    monkeypatch.setattr(pipe.fe, "build_supervised_samples", build)
    return pipe, X, y, anchors, ends


def test_prepare_data_fits_actual_pipeline_on_purged_train(monkeypatch):
    pipe, X, y, anchors, ends = _prepared_pipeline(monkeypatch)
    data = pipe.prepare_data(pd.DataFrame())
    expected = np.flatnonzero((ends[:60] < anchors.iloc[60]).to_numpy())
    assert np.allclose(pipe.fe.scaler_x.mean_, X[expected].mean(axis=0))
    assert np.allclose(pipe.fe.scaler_y.mean_, y[expected].mean())
    assert data.d_train.max() < anchors.iloc[60]
    assert data.d_val.max() < anchors.iloc[80]
    assert len(data.X_train) == 55 and len(data.X_val) == 15 and len(data.X_test) == 20
    assert np.array_equal(data.bar_info_test["base_vol"], np.arange(80, 100))


def test_run_single_hpo_uses_separate_scalers(monkeypatch):
    pipe, X, y, anchors, ends = _prepared_pipeline(monkeypatch)
    data = pipe.prepare_data(pd.DataFrame())
    mean_before = pipe.fe.scaler_x.mean_.copy()
    pipe.config.hpo_method = "BO"
    captured = []

    def hpo(method, cls, xt, yt, xv, yv, **kwargs):
        captured.append((xt, yt, xv, yv))
        return {}

    monkeypatch.setattr(s, "run_hpo", hpo)
    result = pipe.run_single("RidgeReg", data)
    assert result.error is None
    assert len(captured) == 1
    xt, yt, xv, yv = captured[0]
    assert np.allclose(xt.mean(axis=0), 0)
    assert np.isclose(yt.mean(), 0)
    assert xv.mean() > 1 and yv.mean() > 1
    assert np.array_equal(pipe.fe.scaler_x.mean_, mean_before)


def test_cv_purges_labels_at_each_validation_anchor(monkeypatch):
    pipe, X, y, anchors, ends = _prepared_pipeline(monkeypatch)
    pipe.fe.fit_scale(X, y)
    pipe.config.cv_folds = 3
    pipe.config.hpo_method = "关闭"
    seen = []
    params_seen = []

    class SpyModel:
        def __init__(self, **kwargs):
            params_seen.append(kwargs)

        def fit(self, x, target):
            seen.append(len(x))

        def predict(self, x):
            return np.zeros(len(x))

    result = pipe._cross_validate(SpyModel, {}, pipe.fe.transform(X), y, y,
                                  anchor_dates=anchors, label_end_dates=ends)
    assert seen == [20, 45, 70]
    assert result
    # 整段HPO选出的参数不能传入早期折；折内重新选择只能看到该折训练历史。
    pipe.config.hpo_method = "BO"
    hpo_seen = []
    def hpo(method, cls, xt, yt, xv, yv, **kwargs):
        hpo_seen.append(len(xt))
        return {"past_only": True}
    monkeypatch.setattr(s, "run_hpo", hpo)
    params_seen.clear()
    result = pipe._cross_validate(SpyModel, {"future_choice": True}, pipe.fe.transform(X), y, y,
                                  anchor_dates=anchors, label_end_dates=ends)
    assert result
    assert hpo_seen == [11, 31, 51]
    assert params_seen == [{"past_only": True}] * 3


def test_direction_ties_match_single_stock_pipeline():
    actual = np.array([-1, 0, 1, 1])
    pred = np.zeros(4)
    assert s._mh_direction_da(pred, actual) == 50
    assert s._mh_direction_da(pred, actual) == s.TrainingPipeline._dir_metrics(
        10 + actual, 10 + pred, np.full(4, 10))["DA"]
    assert s._mh_direction_da(np.zeros(4), np.ones(4)) == 0


def test_walk_forward_purges_training_labels(monkeypatch):
    pipe, X, y, anchors, ends = _prepared_pipeline(monkeypatch)
    seen = []

    class SpyModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, target):
            seen.append(len(x))

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(s, "TrainingPipeline", lambda cfg: pipe)
    monkeypatch.setitem(s.ALGO_REGISTRY, "TemporalSpy", SpyModel)
    out = s.walk_forward_eval("TemporalSpy", pd.DataFrame(), horizon=5, n_folds=3)
    assert seen == [45, 70]
    assert out["n_oos"] == 50


def _pooled_dataset():
    dates = pd.bdate_range("2023-01-02", periods=90)
    rows = []
    for i in range(70):
        for code in range(6):
            for h in (1, 5):
                row = {f: float((i + code) % 7 + 1) for f in s.MH_FEATURE_COLS}
                target = float(code - 2)  # 包含零超额/零涨跌，验证实际多数类基准。
                row.update({t: target for t in s.MH_TARGET_COLS})
                row.update(date=dates[i], label_end_date=dates[i+h], code=str(code), h=h)
                rows.append(row)
    return pd.DataFrame(rows), dates


def test_global_training_integration_uses_purged_splits_and_real_baselines(monkeypatch):
    ds, dates = _pooled_dataset()
    monkeypatch.setattr(s, "build_multi_horizon_dataset", lambda *a, **k: ds.copy())
    monkeypatch.setattr(s, "HAS_STATSMODELS", False)
    calls = []
    hpo_calls = []
    def hpo(method, cls, xt, yt, xv, yv, **kwargs):
        assert np.allclose(xt.mean(axis=0), 0)
        assert len(xt) and len(xv)
        hpo_calls.append(len(xt))
        return {}
    monkeypatch.setattr(s, "run_hpo", hpo)
    real_split = s._purged_date_split

    def split(frame, cutoff):
        tr, te = real_split(frame, cutoff)
        assert (tr["label_end_date"] < pd.Timestamp(cutoff)).all()
        assert set(tr["date"]).isdisjoint(te["date"])
        calls.append((tr, te))
        return tr, te

    monkeypatch.setattr(s, "_purged_date_split", split)
    out = s.train_global_pooled(codes=["0"], algos=["RidgeReg"],
                                 split_date=str(dates[35].date()), hpo_method="BO", freeze=False)
    assert len(calls) == 2
    assert len(hpo_calls) == len(s.MH_TARGET_COLS)
    te = calls[-1][1]
    expected = (te["y4_mean_pct"] <= 0).mean() * 100
    assert out["results"][0]["DA_dir_pct"] == round(expected, 2)
    assert {r["algo"] for r in out["results"]} >= {"RidgeReg", "总是涨(方向基准)", s.ALWAYS_DOWN_NAME}
    assert out["results"][-1].get("DA_dir_pct") is not None


def test_cross_sectional_training_uses_actual_majority_baseline(monkeypatch):
    ds, dates = _pooled_dataset()
    # 同日相同标签的截面：所有超额都为0，基准应为100%，不能写死50%。
    ds["y4_mean_pct"] = 1.0
    monkeypatch.setattr(s, "build_multi_horizon_dataset", lambda *a, **k: ds.copy())
    out = s.train_global_cross_sectional(codes=["0"], algos=["RidgeReg"],
                                          split_date=str(dates[35].date()), hpo_method="关闭")
    assert out["base_rate_pct"] == 100
    assert out["direction_baselines"]["Naive(涨跌=0)"] == 100
    assert out["results"][0]["xs_da_pct"] == 100


def _quote():
    return {"price": 11, "avg": 10, "high": 12, "low": 8,
            "bids": [(10.99 - i * .01, 30) for i in range(5)],
            "asks": [(11.01 + i * .01, 10) for i in range(5)]}


def test_intraday_describes_real_fields_without_inventing_data():
    quote = _quote()
    result = s.analyze_intraday_snapshot(quote)
    assert result["book_imbalance_pct"] == 50
    assert result["vs_avg_pct"] == pytest.approx(10)
    assert result["range_position_pct"] == 75
    assert result["spread_bps"] == pytest.approx(.02 / 11 * 10000)
    assert result["source_time_verified"] is False
    quote["bids"] = quote["bids"][:1]
    quote["avg"] = float("nan")
    quote["price"] = float("inf")
    result = s.analyze_intraday_snapshot(quote)
    assert result["book_imbalance_pct"] is None
    assert result["vs_avg_pct"] is None
    assert result["range_position_pct"] is None
    assert result["warnings"]
    assert "非投资建议" in s.intraday_snapshot_html(quote)


def test_snapshot_preserves_legacy_rows_and_missing_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "MONITOR_DIR", str(tmp_path))
    day = s.dt.datetime.now().strftime("%Y%m%d")
    path = tmp_path / f"monitor_600000_{day}.csv"
    path.write_text("time,price,buy1_p,buy1_v\n2024-01-01 10:00:00,10,9.99,5\n", encoding="utf-8-sig")
    quote = _quote()
    s.write_snapshot("600000", quote)
    s.write_snapshot("600000", {"price": 12})
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["price"] == "10" and rows[0]["buy1_v"] == "5"
    assert rows[1]["buy5_v"] == "30" and rows[1]["avg"] == "10"
    assert rows[2]["price"] == "12" and rows[2]["buy1_p"] == ""
    assert all(None not in row for row in rows)


def test_collector_keeps_quote_when_fundflow_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "MONITOR_DIR", str(tmp_path))
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: _quote())
    def failed_flow(code):
        raise OSError("offline test")
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", failed_flow)
    monkeypatch.setattr(s, "is_trading_now", lambda: (False, "合成测试：休市"))
    out = s.collect_intraday_snapshot("600000")
    assert out["quote"]["price"] == 11
    assert out["fundflow"]["main_net"] is None
    assert out["analysis"]["warnings"]
    assert out["quote"]["received_at"] and out["quote"]["fundflow_received_at"]
    with open(out["snapshot_path"], encoding="utf-8-sig", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["price"] == "11" and row["main_net"] == ""
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: {"price": "-"})
    with pytest.raises(RuntimeError, match="现价"):
        s.collect_intraday_snapshot("600000")
    assert len(pd.read_csv(out["snapshot_path"])) == 1


def test_realtime_transport_has_timeout_and_preserves_hand_units(monkeypatch):
    import requests
    calls = []
    data = {"f57": "600000", "f43": 11, "f71": 10, "f47": 500,
            "f19": 10.99, "f20": 30, "f39": 11.01, "f40": 10, "f170": "-"}
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": data}
    def get(url, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(s.StockDataFetcher, "_retry", lambda fn, **kwargs: fn())
    quote = s.StockDataFetcher.fetch_realtime("600000", bypass_proxy=False)
    assert calls[0]["timeout"] == 8
    assert calls[0]["params"]["secid"] == "1.600000"
    assert "f530" in calls[0]["params"]["fields"].split(",")  # 五档聚合字段，不能仅请求逐档字段。
    assert quote["bids"][0] == (10.99, 30)
    assert quote["asks"][0] == (11.01, 10)
    assert quote["vol"] == 500 and quote["book_volume_unit"] == "手"
    assert quote["pct"] is None
    data["f57"] = "000001"
    with pytest.raises(RuntimeError, match="不匹配"):
        s.StockDataFetcher.fetch_realtime("600000", bypass_proxy=False)


def test_order_comparison_excludes_legacy_unknown_units(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "MONITOR_DIR", str(tmp_path))
    old = _quote()
    s.write_snapshot("600000", old)
    assert "error" in s.analyze_order_sincerity("600000")
    fresh = dict(old, book_volume_unit="手")
    s.write_snapshot("600000", fresh)
    s.write_snapshot("600000", fresh)
    out = s.analyze_order_sincerity("600000")
    assert out["excluded_unverified_rows"] == 1
    assert out["n_snap"] == 2 and out["book_volume_unit"] == "手"
    assert out["buy_keep_pct"] == 100


class _NoWait:
    def __init__(self):
        self.waits = []
    def is_set(self):
        return False
    def wait(self, seconds):
        self.waits.append(seconds)
        return False


def test_monitor_continues_after_failure_without_overlapping_requests(monkeypatch):
    attempts = []
    def collect(code):
        attempts.append(code)
        if len(attempts) == 2:
            raise OSError("offline interruption")
        return {"code": code, "analysis": {"price": 10 + len(attempts)}}
    monkeypatch.setattr(s, "collect_intraday_snapshot", collect)
    updates, event = [], _NoWait()
    out = s.monitor_intraday("600000", interval=3, max_updates=3,
                              on_update=updates.append, stop_event=event)
    assert out == {"attempts": 3, "succeeded": 2, "failed": 1, "stopped": False}
    assert len(updates) == 3 and "error" in updates[1]
    assert updates[-1]["analysis"]["price"] == 13
    assert event.waits == [3, 3]
    stopped = threading.Event()
    stopped.set()
    assert s.monitor_intraday("600000", stop_event=stopped)["attempts"] == 0
    for interval in (0, 2, float("nan")):
        with pytest.raises(ValueError):
            s.monitor_intraday("600000", interval=interval, max_updates=1)
    with pytest.raises(ValueError):
        s.monitor_intraday("invalid", max_updates=1)


def test_monitor_cli_keeps_last_success_json_on_network_error(tmp_path, monkeypatch, capsys):
    calls = []
    def collect(code):
        calls.append(code)
        if len(calls) == 2:
            raise OSError("offline interruption")
        return {"code": code, "status": "合成测试", "snapshot_path": "synthetic.csv",
                "analysis": {"price": 11, "received_at": "test"}, "fundflow": {"main_net": None}}
    monkeypatch.setattr(s, "collect_intraday_snapshot", collect)
    real_monitor = s.monitor_intraday
    def monitor(code, interval, count, callback):
        return real_monitor(code, interval, count, callback, stop_event=_NoWait())
    monkeypatch.setattr(s, "monitor_intraday", monitor)
    output = tmp_path / "latest.json"
    monkeypatch.setattr(sys, "argv", ["stock_predictor.py", "--monitor", "600000",
                                      "--monitor-count", "2", "--json", str(output)])
    result = s.main()
    assert result["succeeded"] == 1 and result["failed"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["analysis"]["price"] == 11
    assert "offline interruption" in capsys.readouterr().out


@pytest.mark.skipif(not s.HAS_PYSIDE6, reason="可选PySide6未安装")
def test_gui_monitor_is_async_and_discards_stopped_response(monkeypatch):
    from PySide6.QtCore import QCoreApplication, QObject, QTimer
    app = QCoreApplication.instance() or QCoreApplication([])
    gate, started = threading.Event(), threading.Event()
    records = []
    def collect(code, persist):
        started.set()
        assert gate.wait(2)
        return {"code": code, "quote": _quote(), "fundflow": {}, "trading": True, "status": "合成测试"}
    monkeypatch.setattr(s, "collect_intraday_snapshot", collect)

    class Harness(QObject):
        _refresh_monitor = s.MainWindow._refresh_monitor
        _on_monitor_result = s.MainWindow._on_monitor_result
        _on_monitor_worker_finished = s.MainWindow._on_monitor_worker_finished
        def _on_monitor_error(self, result):
            records.append(result)
        def _record_snapshot(self, code, quote):
            records.append(code)
        def _render_monitor(self, *args):
            pass

    harness = Harness()
    harness._monitor_timer = QTimer(harness)
    harness._monitor_timer.start(10000)
    harness._monitor_code, harness._monitor_generation, harness._monitor_worker = "600000", 1, None
    t0 = time.monotonic()
    harness._refresh_monitor()
    assert time.monotonic() - t0 < .5
    worker = harness._monitor_worker
    try:
        assert started.wait(1)
        harness._refresh_monitor()
        assert harness._monitor_worker is worker  # 慢请求期间不启动第二个线程。
        harness._monitor_timer.stop()
        harness._monitor_generation += 1
        gate.set()
        deadline = time.monotonic() + 3
        while harness._monitor_worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.001)
        assert harness._monitor_worker is None
        assert records == []  # 停止后到达的响应不显示、不写快照。
    finally:
        gate.set()
        if harness._monitor_worker is not None:
            harness._monitor_worker.wait(3000)
            app.processEvents()


def test_ranking_returns_use_exit_price_not_future_average(monkeypatch):
    ds, dates = _pooled_dataset()
    ds = ds[ds["h"] == 5].copy()
    ds["close"] = 10.0
    ds["end_close_yuan"] = 10.0
    monkeypatch.setattr(s, "build_multi_horizon_dataset", lambda *a, **k: ds.copy())
    out = s.rank_stocks_cross_sectional(codes=["0"], horizon=5, algo="RidgeReg",
        split_date=str(dates[35].date()), neutralize=False)
    assert out["top_cagr_gross_pct"] == 0
    assert out["ls_spread_pct"] == 0
    years = (dates[70] - dates[35]).days / 365.25
    expected = ((1 - 0.006) ** (out["n_rebalance"] / years) - 1) * 100
    assert out["top_cagr_net_pct"] == pytest.approx(round(expected, 2))



def test_return_mode_excludes_raw_price_levels_and_keeps_stationary_features():
    raw = s.StockDataFetcher.generate_synthetic_data(n_days=220)
    fe_ret = s.FeatureEngineer(window_size=10, horizon=1, target_mode="return")
    X_ret, _, _ = fe_ret.build_supervised_samples(raw)
    assert X_ret.shape[1] == 10 * len(fe_ret.feature_cols)
    assert set(fe_ret.feature_cols).isdisjoint(s.FeatureEngineer._RAW_LEVEL_COLS)
    for col in ("gap_ret", "intraday_ret", "natr14", "ret_5d", "macd_hist_norm", "obv_slope10"):
        assert col in fe_ret.feature_cols

    fe_px = s.FeatureEngineer(window_size=10, horizon=1, target_mode="price")
    fe_px.build_supervised_samples(raw)
    assert "close" in fe_px.feature_cols and "ma20" in fe_px.feature_cols


def test_ensemble_model_and_tabular_slicing_run_end_to_end():
    out = s.run_experiment(
        synthetic=True, algos=["Ensemble", "RidgeReg", "ELM"],
        hpo="关闭", forecast=True, add_naive_baseline=True, metrics=["R2", "RMSE", "DA"],
    )
    rows = {r["algo"]: r for r in out["results"]}
    assert "Naive(前值)" in rows
    for name in ("Ensemble", "RidgeReg", "ELM"):
        assert rows[name]["error"] is None
        assert np.isfinite(rows[name]["metrics"]["R2"])
        assert 0.0 <= rows[name]["metrics"]["DA"] <= 100.0
    assert out["forecast"] and len(out["forecast"]) == 3
    assert all(np.isfinite(fc["pred_close"]) for fc in out["forecast"])


def _synthetic_intraday_bars(n=180, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-11-10 09:35:00", periods=n, freq="5min")
    rets = rng.normal(0.0002, 0.004, size=n)
    close = 10.0 * np.cumprod(1.0 + rets)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.001, 0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.001, 0.001, size=n)))
    vol = rng.integers(500, 5000, size=n).astype(float)
    return pd.DataFrame({
        "datetime": ts, "open": open_, "close": close, "high": high, "low": low,
        "volume": vol, "amount": close * vol * 100.0,
    })


def test_intraday_bar_features_are_causal_and_predict_purged_oos():
    bars = _synthetic_intraday_bars(n=180)
    f1 = s.build_intraday_bar_features(bars)
    bars_mut = bars.copy()
    bars_mut.loc[150:, ["close", "high", "low", "volume"]] *= 3.0
    f2 = s.build_intraday_bar_features(bars_mut)
    for col in s.INTRADAY_BAR_FEATURE_COLS:
        s1 = f1[col].iloc[:150].fillna(-999)
        s2 = f2[col].iloc[:150].fillna(-999)
        assert np.allclose(s1, s2), f"{col} 受到未来K线篡改影响"

    short_fc = s.predict_intraday_bars(bars.iloc[:30], horizon_bars=3)
    assert short_fc["available"] is False

    fc = s.predict_intraday_bars(bars, horizon_bars=3, live_price=float(bars["close"].iloc[-1]) * 1.003)
    assert fc["available"] is True
    assert 20.0 <= fc["p_up_pct"] <= 80.0
    assert fc["p_up_pct"] + fc["p_down_pct"] == pytest.approx(100.0)
    assert fc["pred_lo"] <= fc["pred_price"] <= fc["pred_hi"]
    assert 0.0 <= fc["oos_da_pct"] <= 100.0
    assert 50.0 <= fc["majority_baseline_pct"] <= 100.0


def test_compute_intraday_realtime_prediction_and_tracker_verification():
    bull_q = {
        "price": 10.50, "avg": 10.10, "high": 10.55, "low": 9.95, "vol": 2000, "amount": 2.02e7,
        "limit_up": 11.00, "limit_down": 9.00,
        "bids": [(10.49 - i * 0.01, 800) for i in range(5)],
        "asks": [(10.51 + i * 0.01, 100) for i in range(5)],
    }
    prev_q = dict(bull_q, price=10.35, vol=1200, main_net=1e6,
                  bids=[(10.34 - i * 0.01, 300) for i in range(5)],
                  asks=[(10.36 + i * 0.01, 400) for i in range(5)])
    bull_pred = s.compute_intraday_realtime_prediction(
        bull_q, snapshots=[prev_q, bull_q], fundflow={"main_net": 5e6, "main_pct": 8.0},
        bar_forecast={"available": True, "horizon_bars": 3, "p_up_pct": 66.0,
                      "pred_ret_pct": 0.4, "pred_price": 10.54, "pred_lo": 10.45,
                      "pred_hi": 10.63, "band_pct": 0.85, "oos_da_pct": 58.0,
                      "majority_baseline_pct": 52.0, "oos_excess_da_pct": 6.0, "oos_n": 30},
    )
    assert bull_pred["p_up_pct"] > 55.0
    assert "偏多" in bull_pred["direction"]
    assert bull_pred["micro_price"] > 10.50
    assert "非投资建议" in bull_pred["disclaimer"] or "投资建议" in bull_pred["disclaimer"]

    bear_q = {
        "price": 9.60, "avg": 10.00, "high": 10.10, "low": 9.58, "limit_up": 11.00, "limit_down": 9.60,
        "bids": [(9.59 - i * 0.01, 50) for i in range(5)],
        "asks": [(9.61 + i * 0.01, 900) for i in range(5)],
    }
    bear_pred = s.compute_intraday_realtime_prediction(bear_q)
    assert bear_pred["p_up_pct"] < 45.0
    assert "偏空" in bear_pred["direction"]
    assert "跌停" in bear_pred["limit_state"]
    assert "⚡ 场内实时涨跌预测计算" in s.intraday_realtime_html(bull_pred)

    tracker = s.IntradayPredictionTracker(horizon_sec=30.0)
    st1 = tracker.update("600000", 10.0, 62.0, ts=1000.0)
    assert st1["total_recorded"] == 1 and st1["verified_n"] == 0
    st2 = tracker.update("600000", 10.2, 40.0, ts=1035.0)
    assert st2["verified_n"] == 1 and st2["hit_n"] == 1 and st2["hit_rate_pct"] == 100.0


def test_fetch_intraday_bars_transport_and_intraday_predict_cli(tmp_path, monkeypatch):
    import requests
    calls = []
    bars = _synthetic_intraday_bars(n=100)
    klines = [
        f"{r.datetime.strftime('%Y-%m-%d %H:%M')},{r.open:.3f},{r.close:.3f},{r.high:.3f},{r.low:.3f},{int(r.volume)},{r.amount:.1f}"
        for r in bars.itertuples()
    ]
    payload = {"data": {"code": "600000", "klines": klines}}

    class Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return payload

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    df = s.StockDataFetcher.fetch_intraday_bars("600000", period="5", bypass_proxy=False)
    assert calls[0]["timeout"] == 8
    assert calls[0]["params"]["klt"] == "5"
    assert len(df) == 100

    payload["data"]["code"] = "000001"
    with pytest.raises(RuntimeError, match="不匹配"):
        s.StockDataFetcher.fetch_intraday_bars("600000", period="5", bypass_proxy=False)

    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: _quote())
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", lambda code: {"main_net": 2e6, "main_pct": 4.5})
    monkeypatch.setattr(s.StockDataFetcher, "fetch_intraday_bars", lambda code, period="5": bars)
    monkeypatch.setattr(s, "is_trading_now", lambda: (True, "盘中交易"))
    out_path = tmp_path / "intraday.json"
    monkeypatch.setattr(sys, "argv", ["stock_predictor.py", "--intraday-predict", "600000", "--json", str(out_path)])
    res = s.main()
    assert res["intraday_prediction"]["bar_forecast"]["available"] is True
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["code"] == "600000" and "p_up_pct" in saved["intraday_prediction"]



def test_generate_ths_formula_and_export_cli(tmp_path, monkeypatch):
    ths = s.generate_ths_formula("MY_THS")
    assert ths["name"] == "MY_THS"
    assert "MA(CLOSE, 5)" in ths["formula"]
    assert "量化多空分" in ths["formula"]
    assert "非买卖信号" in ths["formula"]
    assert len(ths["guide"]) >= 4

    out_txt = tmp_path / "ths.txt"
    monkeypatch.setattr(sys, "argv", ["stock_predictor.py", "--export-ths", str(out_txt)])
    res = s.main()
    assert res["name"] == "SP_QUANT"
    content = out_txt.read_text(encoding="utf-8")
    assert "量化多空分" in content and "云同步" in content


def test_mobile_web_server_and_auto_analysis_service(monkeypatch):
    import urllib.request
    import urllib.error

    bars = _synthetic_intraday_bars(n=100)
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: _quote())
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", lambda code: {"main_net": 1.5e6, "main_pct": 3.2})
    monkeypatch.setattr(s.StockDataFetcher, "fetch_intraday_bars", lambda code, period="5": bars)
    monkeypatch.setattr(s, "is_trading_now", lambda: (True, "盘中交易"))

    svc = s.AutoAnalysisService(codes=["600519", "000001"], interval_sec=10.0)
    entry = svc.analyze_code_once("600519")
    assert entry["code"] == "600519"
    assert entry["intraday_prediction"]["bar_forecast"]["available"] is True

    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0, auth_token="secret42")
    urls = srv.start(background=True)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        # 未带 token 应拒绝访问
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/api/state", timeout=3)
        assert exc_info.value.code == 403

        # 携带 token 读取手机主页、浮窗页、状态 JSON 与同花顺公式
        html_main = urllib.request.urlopen(f"{base}/?token=secret42", timeout=3).read().decode("utf-8")
        assert "A股场内实时预测" in html_main and "/widget?token=secret42" in html_main

        html_widget = urllib.request.urlopen(f"{base}/widget?token=secret42", timeout=3).read().decode("utf-8")
        assert "同花顺悬浮预测条" in html_widget

        state = json.loads(urllib.request.urlopen(f"{base}/api/state?token=secret42", timeout=3).read().decode("utf-8"))
        assert state["active_code"] == "600519"
        assert state["active"]["intraday_prediction"]["p_up_pct"] > 0

        ths_api = json.loads(urllib.request.urlopen(f"{base}/api/ths_formula?token=secret42", timeout=3).read().decode("utf-8"))
        assert "量化多空分" in ths_api["formula"]

        # POST /api/watch 动态切换/添加监控代码
        req = urllib.request.Request(
            f"{base}/api/watch?token=secret42",
            data=json.dumps({"code": "002418", "bar_period": "1", "horizon_bars": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        post_res = json.loads(urllib.request.urlopen(req, timeout=3).read().decode("utf-8"))
        assert post_res["ok"] is True and post_res["active_code"] == "002418"
        assert "002418" in svc.codes and svc.bar_period == "1" and svc.horizon_bars == 5
        assert "token=secret42" in urls["widget_url"]
    finally:
        srv.stop()
