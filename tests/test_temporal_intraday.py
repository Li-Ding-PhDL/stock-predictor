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
