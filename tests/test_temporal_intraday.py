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


def test_fetch_realtime_tencent_fallback_on_eastmoney_disconnect(monkeypatch):
    import requests

    parts = [""] * 55
    parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] = "贵州茅台", "600519", "1453.90", "1435.26", "1426.30", "39967"
    parts[9], parts[10], parts[11], parts[12] = "1453.50", "1", "1453.00", "5"
    parts[19], parts[20], parts[21], parts[22] = "1453.90", "2", "1453.96", "3"
    parts[31], parts[32], parts[33], parts[34] = "18.64", "1.30", "1454.93", "1425.00"
    parts[35], parts[38], parts[47], parts[48], parts[49], parts[51] = "1453.90/39967/5767305714", "0.32", "1578.79", "1291.73", "1.23", "1443.02"
    tx_body = ('v_sh600519="' + "~".join(parts) + '";').encode("gbk")

    class Resp:
        content = tx_body
        text = tx_body.decode("gbk")
        def raise_for_status(self):
            pass

    def fake_get(url, **kwargs):
        if "eastmoney.com" in url:
            raise requests.ConnectionError("RemoteDisconnected")
        return Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(s.StockDataFetcher, "_retry", lambda fn, **kwargs: fn())
    q = s.StockDataFetcher.fetch_realtime("600519", bypass_proxy=False)
    assert q["source"] == "tencent"
    assert q["name"] == "贵州茅台"
    assert q["price"] == 1453.90 and q["avg"] == 1443.02
    assert q["book_volume_unit"] == "手" and q["vol"] == 39967
    assert q["bids"][0] == (1453.50, 1.0) and q["asks"][0] == (1453.90, 2.0)

    parts[2] = "000001"
    tx_bad = ('v_sh600519="' + "~".join(parts) + '";').encode("gbk")
    Resp.content = tx_bad
    with pytest.raises(RuntimeError, match="不匹配"):
        s.StockDataFetcher.fetch_realtime("600519", bypass_proxy=False)


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
    assert "涨倾向" in ths["formula"]
    assert "盘中涨分" in ths["intraday_formula"]
    assert "POINTDOT" in ths["formula"]
    assert "非买卖信号" in ths["formula"]
    assert len(ths["guide"]) >= 4

    # 带入历史行情时进行样本外校验并输出数据校准权重
    raw = s.StockDataFetcher.generate_synthetic_data(n_days=220)
    ths_fit = s.generate_ths_formula("FIT_THS", df=raw, horizon=1)
    assert ths_fit["oos_eval"] is not None
    assert 0.0 <= ths_fit["oos_eval"]["oos_da_pct"] <= 100.0
    assert "样本外DA=" in ths_fit["formula"]

    out_txt = tmp_path / "ths.txt"
    monkeypatch.setattr(sys, "argv", ["stock_predictor.py", "--export-ths", str(out_txt)])
    res = s.main()
    assert res["name"] == "SP_QUANT"
    content = out_txt.read_text(encoding="utf-8")
    assert "涨倾向" in content and "盘中涨分" in content and "云同步" in content


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
        assert "涨倾向" in ths_api["formula"] and "盘中涨分" in ths_api["intraday_formula"]

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



def test_intraday_tracker_persistence_and_calibration_summary(tmp_path):
    tracker = s.IntradayPredictionTracker(horizon_sec=20.0, persist_dir=str(tmp_path))
    tracker.update("600519", 100.0, 62.0, ts=1000.0)
    tracker.update("600519", 102.0, 38.0, ts=1025.0)
    st = tracker.update("600519", 101.0, 50.0, ts=1050.0)
    assert st["verified_n"] == 2 and st["hit_n"] == 2 and st["hit_rate_pct"] == 100.0
    assert st["hist_verified_n"] == 2 and st["hist_hit_rate_pct"] == 100.0

    cal = s.intraday_calibration_summary(code="600519", persist_dir=str(tmp_path))
    assert cal["verified_n"] == 2 and cal["hit_rate_pct"] == 100.0
    bucket_names = {b["bucket"] for b in cal["buckets"]}
    assert "高看涨(>=58%)" in bucket_names and "高看跌(<=42%)" in bucket_names


def test_auto_service_async_daily_and_name_propagation(tmp_path, monkeypatch):
    import urllib.request

    bars = _synthetic_intraday_bars(n=100)
    q_named = dict(_quote(), name="贵州茅台")
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: q_named)
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", lambda code: {"main_net": 1.0e6, "main_pct": 2.0})
    monkeypatch.setattr(s.StockDataFetcher, "fetch_intraday_bars", lambda code, period="5": bars)
    monkeypatch.setattr(s, "is_trading_now", lambda: (False, "休市"))
    monkeypatch.setattr(s, "run_experiment", lambda **kw: {
        "results": [{"algo": "Ensemble", "metrics": {"DA": 60.0, "UP_P": 62.0, "RMSE": 1.1}, "error": None}],
        "forecast": [{"algo": "Ensemble", "last_date": "2026-11-16", "last_close": 100.0, "pred_close": 101.0}],
        "trust_gate": {"n_passed": 1, "n_models": 1},
    })

    svc = s.AutoAnalysisService(codes=["600519"], interval_sec=5.0, persist_dir=str(tmp_path))
    entry = svc.analyze_code_once("600519")
    assert entry["name"] == "贵州茅台"

    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        req = urllib.request.Request(
            f"{base}/api/daily_analyze",
            data=json.dumps({"code": "600519"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 202
            ack = json.loads(resp.read().decode("utf-8"))
            assert ack["ok"] is True

        deadline = time.monotonic() + 3.0
        state = {}
        while time.monotonic() < deadline:
            state = json.loads(urllib.request.urlopen(f"{base}/api/state", timeout=3).read().decode("utf-8"))
            if state.get("daily") and not state.get("daily_running"):
                break
            time.sleep(0.05)
        assert state["daily"]["models"][0]["algo"] == "Ensemble"
        assert state["watchlist_summary"][0]["name"] == "贵州茅台"

        cal = json.loads(urllib.request.urlopen(f"{base}/api/intraday_calibration?code=600519", timeout=3).read().decode("utf-8"))
        assert cal["verified_n"] == 0
    finally:
        srv.stop()


def _write_synthetic_monitor_csv(path, rows):
    import csv

    fields = [
        "time", "price", "pct", "vol_ratio", "turnover",
        "buy1_p", "buy1_v", "buy2_p", "buy2_v", "buy3_p", "buy3_v", "buy4_p", "buy4_v", "buy5_p", "buy5_v",
        "sell1_p", "sell1_v", "sell2_p", "sell2_v", "sell3_p", "sell3_v", "sell4_p", "sell4_v", "sell5_p", "sell5_v",
        "avg", "vol", "amount", "main_net", "main_pct", "book_volume_unit",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def test_evaluate_orderbook_accuracy_causal_and_cli(tmp_path, monkeypatch):
    import datetime as dt

    csv_path = tmp_path / "monitor_600519_20261116.csv"
    t0 = dt.datetime(2026, 11, 16, 10, 0, 0)
    rows = []
    # 先插入一条缺少 book_volume_unit="手" 的旧记录，验证会被过滤
    rows.append({
        "time": (t0 - dt.timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S"),
        "price": 100.0, "buy1_p": 99.9, "buy1_v": 900, "sell1_p": 100.0, "sell1_v": 10,
        "book_volume_unit": "",
    })
    prices = [100.0, 100.1, 100.2, 100.3, 100.2, 100.1, 100.0, 100.2, 100.4, 100.6, 100.5, 100.3]
    for i, px in enumerate(prices):
        fut_up = (i + 2 < len(prices)) and (prices[i + 2] > px)
        bv = 600.0 if fut_up else 30.0
        sv = 30.0 if fut_up else 600.0
        rows.append({
            "time": (t0 + dt.timedelta(seconds=30 * i)).strftime("%Y-%m-%d %H:%M:%S"),
            "price": px,
            "avg": px - 0.1 if fut_up else px + 0.1,
            "buy1_p": px - 0.01, "buy1_v": bv, "buy2_p": px - 0.02, "buy2_v": bv,
            "sell1_p": px + 0.01, "sell1_v": sv, "sell2_p": px + 0.02, "sell2_v": sv,
            "vol": 1000 + i * 100,
            "amount": 1e6 + i * 1e5,
            "book_volume_unit": "手",
        })
    _write_synthetic_monitor_csv(csv_path, rows)

    ev = s.evaluate_orderbook_accuracy(codes=["600519"], horizon_sec=60.0, monitor_dir=str(tmp_path))
    assert ev["n_files"] == 1
    assert ev["excluded_unverified_rows"] == 1
    assert ev["n_snapshots"] == len(prices)
    assert ev["n_pairs"] == len(prices) - 2
    assert ev["flat_n"] == 1 and ev["move_n"] == len(prices) - 3
    sig_map = {x["signal"]: x for x in ev["signals"]}
    wobi_sig = sig_map["买手卖手五档失衡(W-OBI)"]
    assert wobi_sig["n_move"] == ev["move_n"]
    assert wobi_sig["da_pct"] > wobi_sig["majority_baseline_pct"]
    assert {b["bucket"] for b in ev["wobi_buckets"]} >= {"强买盘(W-OBI>=+25%)", "强卖盘(W-OBI<=-25%)"}

    # 因果防泄露：prefix 文件只有前5条有效快照（rows[0]为无效旧格式），与追加未来行情截断回放保持一致
    prefix_dir = tmp_path / "prefix_only"
    prefix_dir.mkdir()
    _write_synthetic_monitor_csv(prefix_dir / "monitor_600519_20261116.csv", rows[:6])
    ev_prefix = s.evaluate_orderbook_accuracy(codes=["600519"], horizon_sec=30.0, monitor_dir=str(prefix_dir))
    assert ev_prefix["n_snapshots"] == 5 and ev_prefix["n_pairs"] == 4

    # 完整 CSV 配 --eval-orderbook CLI + --json 输出
    monkeypatch.setattr(s, "MONITOR_DIR", str(tmp_path))
    out_json = tmp_path / "eval.json"
    monkeypatch.setattr(s.sys, "argv", ["stock_predictor.py", "--eval-orderbook", "--eval-horizon", "60", "--json", str(out_json)])
    cli_res = s.main()
    assert cli_res["n_pairs"] == len(prices) - 2
    assert out_json.exists() and json.loads(out_json.read_text(encoding="utf-8"))["n_pairs"] == len(prices) - 2
    assert "#ffffff" in s.orderbook_eval_report_html(cli_res, light=True)


def test_auto_service_persists_orderbook_and_eval_endpoints(tmp_path, monkeypatch):
    import urllib.request

    bars = _synthetic_intraday_bars(n=80)
    q = dict(_quote(), name="贵州茅台", bids=[(100.0, 420.0)], asks=[(100.1, 40.0)], book_volume_unit="手")
    monkeypatch.setattr(s, "MONITOR_DIR", str(tmp_path))
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: dict(q))
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", lambda code: None)
    monkeypatch.setattr(s.StockDataFetcher, "fetch_intraday_bars", lambda code, period="5": bars)
    monkeypatch.setattr(s, "is_trading_now", lambda: (True, "盘中"))

    svc = s.AutoAnalysisService(codes=["600519"], interval_sec=5.0, persist_dir=str(tmp_path), persist_snapshots=True)
    svc.analyze_code_once("600519")
    svc.analyze_code_once("600519")
    csv_files = list(tmp_path.glob("monitor_600519_*.csv"))
    assert len(csv_files) == 1
    lines_trading = [ln for ln in csv_files[0].read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    assert len(lines_trading) == 3  # header + 2 trading snapshots
    assert "手" in lines_trading[1]

    # 休市期间同代码多次轮询只保留 1 条收盘快照，避免夜间重复写入静态行
    monkeypatch.setattr(s, "is_trading_now", lambda: (False, "休市"))
    svc_off = s.AutoAnalysisService(codes=["000858"], interval_sec=5.0, persist_dir=str(tmp_path), persist_snapshots=True)
    svc_off.analyze_code_once("000858")
    svc_off.analyze_code_once("000858")
    off_files = list(tmp_path.glob("monitor_000858_*.csv"))
    assert len(off_files) == 1
    lines_off = [ln for ln in off_files[0].read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    assert len(lines_off) == 2  # header + 1 offhours snapshot only

    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        ev = json.loads(urllib.request.urlopen(f"{base}/api/orderbook_eval?code=600519", timeout=3).read().decode("utf-8"))
        assert ev["horizon_sec"] == 60.0
        html = urllib.request.urlopen(f"{base}/orderbook_eval?code=600519", timeout=3).read().decode("utf-8")
        assert "盘口回放" in html
        assert "#ffffff" in s.orderbook_eval_report_html(ev, light=True)
    finally:
        srv.stop()


def test_market_sentiment_snapshot_and_backtest_daily(tmp_path, monkeypatch):
    import pandas as pd
    import urllib.request

    zt_df = pd.DataFrame([
        {"代码": "600001", "连板数": 3, "所属行业": "军工"},
        {"代码": "600002", "连板数": 2, "所属行业": "军工"},
        {"代码": "600003", "连板数": 1, "所属行业": "半导体"},
        {"代码": "600004", "连板数": 1, "所属行业": "半导体"},
    ])
    zb_df = pd.DataFrame([{"代码": "600005"}])
    pv_df = pd.DataFrame([{"涨跌幅": 3.5}, {"涨跌幅": -1.5}])

    monkeypatch.setattr(s, "HAS_AKSHARE", True)
    monkeypatch.setattr(s, "_MARKET_MOOD_CACHE", {"ts": 0.0, "day": "", "data": None})
    monkeypatch.setattr(s.ak, "stock_zt_pool_em", lambda date: zt_df)
    monkeypatch.setattr(s.ak, "stock_zt_pool_zbgc_em", lambda date: zb_df)
    monkeypatch.setattr(s.ak, "stock_zt_pool_previous_em", lambda date: pv_df)

    mood = s.fetch_market_sentiment_snapshot(day="20261116", bypass_proxy=False)
    assert mood["zt_count"] == 4 and mood["zb_count"] == 1
    assert mood["seal_rate_pct"] == 80.0
    assert mood["max_lb"] == 3 and mood["ladder"]["3板及以上"] == 1
    assert mood["prev_zt_avg_pct"] == 1.0 and mood["prev_zt_up_pct"] == 50.0
    assert {x["industry"] for x in mood["hot_industries"]} == {"军工", "半导体"}

    out_json = tmp_path / "mood.json"
    monkeypatch.setattr(s.sys, "argv", ["stock_predictor.py", "--market-mood", "--json", str(out_json)])
    cli_mood = s.main()
    assert cli_mood["zt_count"] == 4 and out_json.exists()

    bars = _synthetic_intraday_bars(n=80)
    q = dict(_quote(), name="贵州茅台", book_volume_unit="手")
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime", lambda code: dict(q))
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_fundflow", lambda code: None)
    monkeypatch.setattr(s.StockDataFetcher, "fetch_intraday_bars", lambda code, period="5": bars)
    monkeypatch.setattr(s, "run_experiment", lambda **kw: {
        "results": [{
            "algo": "Ensemble",
            "metrics": {"DA": 58.0, "UP_P": 60.0, "RMSE": 1.2},
            "backtest": {
                "total_return_pct": 12.5, "buyhold_return_pct": 4.2,
                "excess_vs_buyhold_pct": 8.3, "max_drawdown_pct": -6.1,
                "n_segments": 3, "cost_bps": 0.2,
                "eq_strat": [1.0, 1.05, 1.125], "eq_bh": [1.0, 1.02, 1.042],
            },
            "error": None,
        }],
        "forecast": [], "trust_gate": {"n_passed": 1, "n_models": 1},
    })
    svc = s.AutoAnalysisService(codes=["600519"], persist_dir=str(tmp_path), persist_snapshots=False)
    daily = svc.run_daily_analysis("600519")
    assert daily["backtest"]["excess_vs_buyhold_pct"] == 8.3
    assert len(daily["backtest"]["eq_strat"]) == 3

    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        m_resp = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/market_mood", timeout=3).read().decode("utf-8"))
        assert m_resp["seal_rate_pct"] == 80.0
    finally:
        srv.stop()


def test_exchange_routing_watchlist_and_batch_scan(monkeypatch):
    import requests
    import urllib.request

    # 沪ETF(588940)与科创板(688037)走 sh/1，创业板(300750)与深ETF(159155)走 sz/0，北交所走 bj/0
    assert s.StockDataFetcher._exchange_market("588940") == ("1", "sh")
    assert s.StockDataFetcher._exchange_market("688037") == ("1", "sh")
    assert s.StockDataFetcher._exchange_market("300750") == ("0", "sz")
    assert s.StockDataFetcher._exchange_market("159155") == ("0", "sz")
    assert s.StockDataFetcher._exchange_market("002594") == ("0", "sz")
    assert s.StockDataFetcher._exchange_market("830001") == ("0", "bj")

    # 补齐同花顺4张自选图里的40只个股 + 新增603466 + 2只场内ETF
    for code in ("002531", "002086", "000980", "688037", "600715", "300324", "002594", "300750", "000678", "603466"):
        assert code in s.USER_WATCHLIST_CODES
    assert len(s.USER_WATCHLIST_CODES) == 41 and len(set(s.USER_WATCHLIST_CODES)) == 41
    assert s.USER_ETF_CODES == ["588940", "159155"]

    def _tx_line(var_prefix, name, code, price, pct):
        p = [""] * 55
        p[1], p[2], p[3], p[4], p[5], p[6] = name, code, str(price), str(price), str(price), "1000"
        p[9], p[10], p[19], p[20] = str(price), "200", str(price + 0.01), "50"
        p[31], p[32], p[33], p[34], p[51] = "0.10", str(pct), str(price), str(price), str(price)
        return f'{var_prefix}="{ "~".join(p) }";'

    body = "\n".join([
        _tx_line("v_sh000001", "上证指数", "000001", 3602.57, -0.87),
        _tx_line("v_sz399001", "深证成指", "399001", 10690.78, -2.53),
        _tx_line("v_sz399006", "创业板指", "399006", 2320.02, -4.37),
        _tx_line("v_sh588940", "科创50ETF富国", "588940", 0.630, -5.41),
        _tx_line("v_sz002594", "比亚迪", "002594", 80.49, 0.63),
    ]).encode("gbk")

    class Resp:
        content = body
        def raise_for_status(self):
            pass

    seen_urls = []
    monkeypatch.setattr(requests, "get", lambda url, **kw: (seen_urls.append(url), Resp())[1])
    batch = s.StockDataFetcher.fetch_realtime_batch(["588940", "002594"], include_indices=True, bypass_proxy=False)
    assert "sh588940" in seen_urls[0] and "sz002594" in seen_urls[0]
    assert len(batch["indices"]) == 3
    assert batch["quotes"]["588940"]["price"] == 0.630 and batch["quotes"]["002594"]["name"] == "比亚迪"

    svc = s.AutoAnalysisService(codes=["588940", "002594"], persist_dir=None, persist_snapshots=False)
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_batch", lambda codes, include_indices=True: batch)
    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/api/scan_watchlist", data=b"{}", method="POST")
        scan_res = json.loads(urllib.request.urlopen(req, timeout=3).read().decode("utf-8"))
        assert scan_res["ok"] is True and scan_res["count"] == 2
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/state", timeout=3).read().decode("utf-8"))
        assert len(st["indices"]) == 3 and st["watchlist_summary"][0]["price"] == 0.630
        w_html = urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/widget", timeout=3).read().decode("utf-8")
        assert 'id="wsel"' in w_html and 'id="bookGrid"' in w_html and 'id="widx"' in w_html
    finally:
        srv.stop()


def test_real_orderbook_wall_trap_seal_ratio_and_outer_inner():
    # 1) 买一单档大托单(>58%)但内盘抛压重(外盘<48.5%) -> 触发诱多托单警示，折算资金金额与委比准确
    q_trap = {
        "price": 8.32, "open": 8.85, "high": 9.03, "low": 8.25, "prev": 9.14, "avg": 8.53,
        "vol": 322365.0, "amount": 2.75e8, "outer_vol": 150000.0, "inner_vol": 172365.0,
        "limit_up": 10.05, "limit_down": 8.23, "book_volume_unit": "手",
        "bids": [(8.31, 3213), (8.30, 120), (8.29, 80), (8.28, 70), (8.27, 60)],
        "asks": [(8.32, 199), (8.33, 299), (8.34, 68), (8.35, 110), (8.36, 23)],
    }
    res_trap = s.compute_intraday_realtime_prediction(q_trap)
    assert res_trap["weibi_pct"] > 50.0
    assert res_trap["buy_total_wan"] > 280.0 and res_trap["sell_total_wan"] > 50.0
    assert res_trap["active_buy_pct"] < 48.0 and res_trap["bid_wall_ratio_pct"] > 80.0
    assert res_trap["price_pos_pct"] < 15.0
    assert res_trap["book_alert"] and "诱多" in res_trap["book_alert"]

    # 2) 涨停封单占成交额 < 2% -> 提醒封板偏弱防开板
    q_zt = {
        "price": 3.74, "open": 3.65, "high": 3.74, "low": 3.50, "prev": 3.40, "avg": 3.66,
        "vol": 10825081.0, "amount": 39.57e8, "outer_vol": 5423747.0, "inner_vol": 5401333.0,
        "limit_up": 3.74, "limit_down": 3.06, "book_volume_unit": "手",
        "bids": [(3.74, 77032), (3.73, 10912), (3.72, 2210), (3.71, 924), (3.70, 5648)],
        "asks": [(0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)],
    }
    res_zt = s.compute_intraday_realtime_prediction(q_zt)
    assert "涨停" in res_zt["limit_state"]
    assert 2800.0 <= res_zt["seal_amount_wan"] <= 2900.0
    assert 0.5 <= res_zt["seal_to_amount_pct"] <= 1.0
    assert res_zt["book_alert"] and "防开板" in res_zt["book_alert"]

    # 3) 连续快照增量内外盘 Δouter > Δinner 增强偏多评分
    q_prev = dict(q_trap, outer_vol=150000.0, inner_vol=172000.0)
    q_next = dict(q_trap, outer_vol=158000.0, inner_vol=172200.0, price=8.36)
    res_mom = s.compute_intraday_realtime_prediction(q_next, snapshots=[q_prev, q_next])
    names = {f["name"]: f for f in res_mom["factors"]}
    assert "内外盘主动买卖比" in names and names["内外盘主动买卖比"]["bias"] == "偏多"


def test_stale_local_csv_refreshes_online_and_falls_back(tmp_path, monkeypatch):
    """本地离线 CSV 明显滞后于目标 end_date(>10天) 时，auto 模式优先联网拉最新；联网失败才回退本地。"""
    csv_dir = tmp_path / "每只股票一个文件" / "前复权"
    csv_dir.mkdir(parents=True)
    local_csv = csv_dir / "601212.csv"
    local_csv.write_text(
        "日期,开盘价,最高价,最低价,收盘价,成交量（股）,成交额（元）\n"
        "2026-04-09,7.90,8.10,7.85,8.00,1000000,8000000\n"
        "2026-04-10,8.00,8.12,7.95,8.04,1100000,8844000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_LOCAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(s.StockDataFetcher, "_cache_path", staticmethod(lambda *a, **kw: str(tmp_path / "cache.parquet")))
    monkeypatch.setattr(s, "HAS_AKSHARE", True)
    monkeypatch.setattr(s, "HAS_BAOSTOCK", False)

    online_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-04-10", "2026-11-17"]),
        "open": [8.00, 4.41], "high": [8.12, 4.61],
        "low": [7.95, 4.32], "close": [8.04, 4.49], "volume": [10000, 11087],
    })
    monkeypatch.setattr(s.StockDataFetcher, "_fetch_akshare", staticmethod(lambda *a, **kw: online_df))
    monkeypatch.setattr(s._AK_THROTTLE, "wait", lambda: None)

    got_auto = s.StockDataFetcher().fetch("601212", "20260401", "20261117", use_cache=False, source="auto")
    assert pd.to_datetime(got_auto["date"]).max().strftime("%Y-%m-%d") == "2026-11-17"
    assert float(got_auto["close"].iloc[-1]) == 4.49

    # 指定 source="local" 时强制仅用本地
    got_local = s.StockDataFetcher().fetch("601212", "20260401", "20261117", source="local")
    assert pd.to_datetime(got_local["date"]).max().strftime("%Y-%m-%d") == "2026-04-10"

    # 若联网失败，则自动回退 stale 本地 CSV
    def _boom(*a, **kw):
        raise RuntimeError("offline")
    monkeypatch.setattr(s.StockDataFetcher, "_fetch_akshare", staticmethod(_boom))
    got_fb = s.StockDataFetcher().fetch("601212", "20260401", "20261117", use_cache=False, source="auto", retries=1)
    assert pd.to_datetime(got_fb["date"]).max().strftime("%Y-%m-%d") == "2026-04-10"


def test_rank_stocks_scores_latest_live_date_and_watchlist_health_api(tmp_path, monkeypatch):
    """验证 rank_stocks_cross_sectional 实时打分使用最新交易日(而非 horizon 日前)，以及 /api/watchlist_health 接口与看板风控提醒。"""
    csv_dir = tmp_path / "每只股票一个文件" / "前复权"
    csv_dir.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=320)
    codes = ["000001", "000002", "000003", "000004", "000005", "000006"]
    rng = np.random.default_rng(7)
    for idx, c in enumerate(codes):
        ret = rng.normal(0.0005 * (idx - 2), 0.02, len(dates))
        close = 10.0 * np.cumprod(1.0 + ret)
        df_c = pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "代码": c,
            "名称": f"测试股{idx}",
            "所属行业": "电子" if idx % 2 == 0 else "医药",
            "开盘价": np.round(close * 0.995, 3),
            "最高价": np.round(close * 1.015, 3),
            "最低价": np.round(close * 0.985, 3),
            "收盘价": np.round(close, 3),
            "成交量（股）": rng.integers(100000, 500000, len(dates)),
            "成交额（元）": rng.integers(1000000, 5000000, len(dates)),
            "换手率": np.round(rng.uniform(1.0, 5.0, len(dates)), 2),
            "涨幅%": np.round(ret * 100.0, 2),
            "滚动市盈率": np.round(rng.uniform(15.0, 35.0, len(dates)), 2),
            "市净率": np.round(rng.uniform(1.2, 3.5, len(dates)), 2),
            "滚动市销率": np.round(rng.uniform(1.0, 4.0, len(dates)), 2),
            "退市时间": ["-"] * len(dates),
        })
        df_c.to_csv(csv_dir / f"{c}.csv", index=False, encoding="utf-8-sig")

    monkeypatch.setenv("STOCK_LOCAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(s, "UPDATED_DATA_ROOT", str(tmp_path))

    rk = s.rank_stocks_cross_sectional(codes=codes, horizon=10, split_date="2024-01-01",
                                       start="20230101", root=str(tmp_path), out_png=None)
    # 必须按最新一天打分，而不是倒退 10 个交易日
    assert rk["as_of"] == dates[-1].strftime("%Y-%m-%d")
    assert len(rk["ranked"]) == len(codes)
    assert all(item["close"] is not None for item in rk["ranked"])

    rep = s.watchlist_health_report(codes=codes, root=str(tmp_path), include_rank=False)
    assert rep["n_codes"] == len(codes)
    assert rep["as_of"] == dates[-1].strftime("%Y-%m-%d")
    assert len(rep["dip_strict"]) == 2
    assert "avoid_summary" in rep and "temperature_summary" in rep
    sample_df, _ = s._mh_load_local_rich(codes[0], str(tmp_path))
    rec = s._dip_record_from_df(sample_df, codes[0])
    assert rec is not None and len(rec[3]) == len(sample_df)

    svc = s.AutoAnalysisService(codes=codes, interval_sec=60.0)
    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        import urllib.request
        with urllib.request.urlopen(base + "/api/watchlist_health", timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["n_codes"] == len(codes)
            assert "dip_strict" in payload
        with urllib.request.urlopen(base + "/", timeout=10) as resp:
            html_txt = resp.read().decode("utf-8")
            assert "loadHealth(" in html_txt
            assert "短线风控提醒" in html_txt
    finally:
        srv.stop()


def test_watchlist_price_percentiles_and_web_pointer(tmp_path, monkeypatch):
    """验证 2020~2026 每日收盘价从低到高排序排位百分比、指针 SVG、实时报价覆写及 Web /price_pos 接口。"""
    csv_dir = tmp_path / "每只股票一个文件" / "前复权"
    csv_dir.mkdir(parents=True)
    # 构造含 2019(应被2020起点过滤掉) 与 2020~2026 收盘价 [10.0, 20.0, 30.0, 40.0, 50.0] 的 2 只测试股
    dates_2020 = pd.date_range("2020-01-02", periods=10, freq="B")
    closes_a = [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 30.0]
    closes_b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    for code, nm, cl_arr in [("000001", "低位样", closes_a), ("000002", "高位样", closes_b)]:
        df = pd.DataFrame({
            "日期": ["2019-12-31"] + list(dates_2020.strftime("%Y-%m-%d")),
            "代码": code,
            "名称": nm,
            "收盘价": [999.0] + cl_arr,  # 2019年的999必须被 start="20200101" 排除
            "退市时间": ["-"] * 11,
        })
        df.to_csv(csv_dir / f"{code}.csv", index=False, encoding="utf-8-sig")

    monkeypatch.setenv("STOCK_LOCAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(s, "UPDATED_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(s.StockDataFetcher, "fetch_realtime_batch", staticmethod(
        lambda codes, include_indices=False: {
            "quotes": {
                "000001": {"code": "000001", "name": "低位样", "price": 15.0, "pct": 1.2, "received_at": "2026-11-20 11:00"},
                "000002": {"code": "000002", "name": "高位样", "price": 14.0, "pct": 2.5, "received_at": "2026-11-20 11:00"},
            }
        }
    ))

    rep = s.watchlist_price_percentiles(codes=["000001", "000002"], start="20200101", root=str(tmp_path), use_live=True)
    assert rep["summary"]["n_scored"] == 2
    assert rep["summary"]["live_count"] == 2
    rows_map = {r["code"]: r for r in rep["rows"]}
    # closes_a sorted = [10,15,20,25,30,30,35,40,45,50], price=15 -> left=1, right=2 -> mid=1.5 -> 15.0%
    r1 = rows_map["000001"]
    assert r1["min_price"] == 10.0 and r1["max_price"] == 50.0
    assert r1["rank_pct"] == 15.0
    assert r1["bucket"] == "15%~20%"
    assert r1["range_pct"] == 12.5
    assert all(b >= a for a, b in zip(r1["sorted_curve"], r1["sorted_curve"][1:]))
    # closes_b sorted = [5..14], price=14 -> left=9, right=10 -> mid=9.5 -> 95.0%
    r2 = rows_map["000002"]
    assert r2["rank_pct"] == 95.0 and r2["bucket"] == "90%~100%" and r2["range_pct"] == 100.0
    assert len(rep["summary"]["buckets"]) == 12
    assert rep["summary"]["buckets"][0]["label"] == "0%~5%"
    assert rep["summary"]["buckets"][4]["label"] == "20%~30%"
    assert rep["summary"]["lt10_count"] == 0
    assert rep["rows"][0]["code"] == "000001"  # 默认按排位升序排列
    md = s.format_price_pos_markdown(rep)
    assert "<10%" in md and "0%~5%" in md and "15%~20%" in md and "90%~100%" in md
    assert "从低到高" in md and "#1" in md and "#2" in md
    assert md.index("000001") < md.index("000002")
    assert s.StockDataFetcher._is_etf_code("588940")
    assert s.StockDataFetcher._is_etf_code("159155")
    assert not s.StockDataFetcher._is_etf_code("688037")

    svc = s.AutoAnalysisService(codes=["000001", "000002"], interval_sec=60.0)
    srv = s.MobileWebServer(svc, host="127.0.0.1", port=0)
    srv.start(background=True)
    try:
        base = f"http://127.0.0.1:{srv.port}"
        import urllib.request
        with urllib.request.urlopen(base + "/api/price_positions", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["summary"]["n_scored"] == 2
            assert data["rows"][0]["rank_pct"] == 15.0
        with urllib.request.urlopen(base + "/price_pos", timeout=10) as resp:
            phtml = resp.read().decode("utf-8")
            assert "<svg" in phtml and "15.0%" in phtml and "95.0%" in phtml
        with urllib.request.urlopen(base + "/widget", timeout=10) as resp:
            whtml = resp.read().decode("utf-8")
            assert "2020~2026排序" in whtml
    finally:
        srv.stop()
