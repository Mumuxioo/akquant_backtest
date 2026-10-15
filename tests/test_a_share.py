"""A 股 SQLite 同步测试。"""

import sqlite3
import threading
import time

import pandas as pd

from src.data import a_share


def test_sync_all_continues_after_single_stock_failure(tmp_path, monkeypatch):
    stocks = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "name": ["平安银行", "万科A"],
        }
    )
    bars = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.2],
            "最高": [10.5, 10.4],
            "最低": [9.9, 10.0],
            "收盘": [10.2, 10.1],
            "成交量": [100, 120],
        }
    )

    monkeypatch.setattr(a_share, "fetch_stock_list", lambda: stocks)

    def fake_fetch(symbol: str, _start: str, _end: str) -> pd.DataFrame:
        if symbol == "000002":
            raise RuntimeError("模拟单只股票拉取失败")
        return bars

    monkeypatch.setattr(a_share, "fetch_daily", fake_fetch)
    db_path = tmp_path / "a_share.db"

    result = a_share.sync_all(
        db_path,
        start_date="20240101",
        end_date="20240131",
        delay=0,
    )

    assert result.total == 2
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.rows == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT symbol, date, close, adjust FROM daily_bars ORDER BY date"
        ).fetchall()
    assert rows == [
        ("000001", "2024-01-02", 10.2, "qfq"),
        ("000001", "2024-01-03", 10.1, "qfq"),
    ]


def test_incremental_overlap_and_new_stock_backfill(tmp_path, monkeypatch):
    db_path = tmp_path / "a_share.db"
    existing = pd.DataFrame(
        {
            "date": ["2024-06-15"],
            "symbol": ["000001"],
            "open": [9.0],
            "high": [9.2],
            "low": [8.8],
            "close": [9.1],
            "volume": [100],
            "amount": [1.0],
            "amplitude": [1.0],
            "pct_change": [1.0],
            "change": [0.1],
            "turnover": [0.1],
        }
    )
    conn = a_share.connect(db_path)
    try:
        a_share._replace_daily(conn, "000001", existing, "20240601", "20240620")
    finally:
        conn.close()

    stocks = pd.DataFrame(
        {"symbol": ["000001", "000002"], "name": ["平安银行", "万科A"]}
    )
    calls: list[tuple[str, str]] = []

    def fake_fetch(symbol: str, start: str, _end: str) -> pd.DataFrame:
        calls.append((symbol, start))
        return pd.DataFrame(
            {
                "日期": ["2024-06-20"],
                "开盘": [10.0],
                "最高": [10.5],
                "最低": [9.9],
                "收盘": [10.2],
                "成交量": [100],
            }
        )

    monkeypatch.setattr(a_share, "fetch_stock_list", lambda: stocks)
    monkeypatch.setattr(a_share, "fetch_daily", fake_fetch)

    result = a_share.sync_all(
        db_path,
        start_date="20200101",
        end_date="20240620",
        delay=0,
        incremental=True,
        overlap_days=10,
    )

    assert result.succeeded == 2
    assert calls == [("000001", "20240605"), ("000002", "20200101")]
    assert a_share._sync_start("2020-01-05", "20200101", "20240620", 30) == "20200101"


def test_interrupt_keeps_committed_rows_and_unlocks(tmp_path, monkeypatch):
    stocks = pd.DataFrame(
        {"symbol": ["000001", "000002"], "name": ["平安银行", "万科A"]}
    )
    bars = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "开盘": [10.0],
            "最高": [10.5],
            "最低": [9.9],
            "收盘": [10.2],
            "成交量": [100],
        }
    )
    calls = {"count": 0}

    def fake_fetch(symbol: str, _start: str, _end: str) -> pd.DataFrame:
        calls["count"] += 1
        if symbol == "000002":
            raise KeyboardInterrupt
        return bars

    monkeypatch.setattr(a_share, "fetch_stock_list", lambda: stocks)
    monkeypatch.setattr(a_share, "fetch_daily", fake_fetch)
    db_path = tmp_path / "a_share.db"

    try:
        a_share.sync_all(db_path, start_date="20240101", end_date="20240131", delay=0)
        raise AssertionError("中断后应该抛出 SyncInterrupted")
    except a_share.SyncInterrupted:
        pass

    other = sqlite3.connect(db_path, timeout=1)
    try:
        rows = other.execute("SELECT symbol, close FROM daily_bars ORDER BY symbol").fetchall()
        other.execute(
            "INSERT INTO stocks(symbol, name, updated_at) VALUES ('999999', 'probe', 't')"
        )
        other.commit()
    finally:
        other.close()
    assert rows == [("000001", 10.2)]
    assert calls["count"] == 2


def test_load_daily_accepts_exchange_prefix(tmp_path):
    db_path = tmp_path / "a_share.db"
    bars = pd.DataFrame(
        {
            "date": ["2025-02-12", "2025-02-13"],
            "symbol": ["600000", "600000"],
            "open": [10.0, 10.4],
            "high": [10.6, 10.8],
            "low": [9.8, 10.2],
            "close": [10.5, 10.3],
            "volume": [1000, None],
            "amount": [1.0, 1.0],
            "amplitude": [1.0, 1.0],
            "pct_change": [1.0, -1.0],
            "change": [0.1, -0.2],
            "turnover": [0.1, 0.1],
        }
    )
    conn = a_share.connect(db_path)
    try:
        a_share._replace_daily(conn, "600000", bars, "20250201", "20260212")
    finally:
        conn.close()

    df = a_share.load_daily("sh600000", "20250212", "20260212", db_path)

    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[0] == 10.5
    assert df["volume"].iloc[1] == 0.0


def test_sync_symbol_writes_bars(tmp_path, monkeypatch):
    """单票同步应写入本地库，供后续 load_daily 直接读取。"""
    bars = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.2],
            "最高": [10.5, 10.4],
            "最低": [9.9, 10.0],
            "收盘": [10.2, 10.1],
            "成交量": [100, 120],
        }
    )
    monkeypatch.setattr(
        a_share,
        "fetch_daily",
        lambda symbol, start, end: bars,
    )
    db_path = tmp_path / "one.db"

    rows = a_share.sync_symbol(
        "000001",
        db_path=db_path,
        start_date="20240101",
        end_date="20240131",
        name="平安银行",
    )
    assert rows == 2

    loaded = a_share.load_daily("000001", "20240101", "20240131", db_path)
    assert len(loaded) == 2
    assert loaded["close"].iloc[-1] == 10.1


def test_sync_all_workers_overlap_and_ignore_delay(tmp_path, monkeypatch):
    """并发拉取应重叠，单只失败不拖垮整批，且不再按只 sleep。"""
    stocks = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "000004"],
            "name": ["平安银行", "万科A", "国农科技", "国华网安"],
        }
    )
    bars = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "开盘": [10.0],
            "最高": [10.5],
            "最低": [9.9],
            "收盘": [10.2],
            "成交量": [100],
        }
    )
    state = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_fetch(symbol: str, _start: str, _end: str) -> pd.DataFrame:
        with lock:
            state["n"] += 1
            state["max"] = max(state["max"], state["n"])
        try:
            time.sleep(0.15)
            if symbol == "000002":
                raise RuntimeError("模拟单只股票拉取失败")
            return bars
        finally:
            with lock:
                state["n"] -= 1

    monkeypatch.setattr(a_share, "fetch_stock_list", lambda: stocks)
    monkeypatch.setattr(a_share, "fetch_daily", fake_fetch)
    db_path = tmp_path / "a_share.db"
    started = time.perf_counter()

    # delay=5 若仍串行，光睡觉就要 15 秒。
    result = a_share.sync_all(
        db_path,
        start_date="20240101",
        end_date="20240131",
        delay=5,
        workers=4,
    )

    assert time.perf_counter() - started < 3
    assert state["max"] >= 2
    assert result.total == 4
    assert result.succeeded == 3
    assert result.failed == 1
    assert result.rows == 3
    with sqlite3.connect(db_path) as conn:
        symbols = [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol")]
    assert symbols == ["000001", "000003", "000004"]
