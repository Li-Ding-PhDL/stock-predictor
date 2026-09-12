# -*- coding: utf-8 -*-
"""数据层护栏测试：全局限流闸门 + 会话内 TTL 缓存（纯逻辑、离线、秒级）。
这是『先补护栏再小步改』的护栏——接入热路径前先锁住这两个原语的行为。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stock_predictor as s  # noqa: E402


def test_rate_limiter_enforces_min_interval():
    rl = s._RateLimiter(min_interval=0.05)
    rl.wait()                       # 第一次立即返回并记时
    t0 = time.monotonic()
    rl.wait()                       # 第二次必须等够 min_interval
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045, f"限流未生效，间隔仅 {elapsed:.3f}s"


def test_rate_limiter_no_wait_when_spaced_out():
    rl = s._RateLimiter(min_interval=0.05)
    rl.wait()
    time.sleep(0.06)                # 已自然间隔超过 min_interval
    t0 = time.monotonic()
    rl.wait()                       # 不应再额外睡
    assert (time.monotonic() - t0) < 0.02


def test_ttl_cache_hit_miss_expire_clear():
    c = s._TTLCache()
    assert c.get("k") is None                    # 未命中
    c.set("k", {"v": 1}, ttl=0.2)
    assert c.get("k") == {"v": 1}                # 命中
    time.sleep(0.25)
    assert c.get("k") is None                    # 过期
    c.set("k2", 42, ttl=5)
    c.clear()
    assert c.get("k2") is None                   # clear 后清空


def test_ttl_cache_independent_keys():
    c = s._TTLCache()
    c.set("a", 1, ttl=5); c.set("b", 2, ttl=5)
    assert c.get("a") == 1 and c.get("b") == 2


def test_session_cache_and_throttle_singletons_exist():
    # 模块级单例存在且类型正确（接入热路径时依赖它们）
    assert isinstance(s._AK_THROTTLE, s._RateLimiter)
    assert isinstance(s._SESSION_CACHE, s._TTLCache)
