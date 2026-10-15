"""全 A 股前复权日线同步到本地 SQLite。"""

from __future__ import annotations

import signal
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from ..utils.logger import get_logger

log = get_logger("data.a_share")

_DAILY_COLUMNS = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}
_VALUE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]
# 2026-10-14 本机探测：腾讯日线（主路径）16 并发连续 300 只全成功，约 67 次/秒；
# 24 也全成功，但单次变慢。东财 4 并发全成功，8 并发开始被对端掐连接。
# 北交所和腾讯失败回退才走东财，所以默认用腾讯的稳定档。
DEFAULT_SYNC_WORKERS = 16


@dataclass(frozen=True)
class SyncResult:
    """一次全市场同步的结果。"""

    total: int
    succeeded: int
    failed: int
    rows: int


class SyncInterrupted(Exception):
    """同步被中断。未提交的写入已回滚，数据库锁已释放。"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """连接 SQLite，并初始化表结构。"""
    if db_path is None or not str(db_path).strip():
        raise ValueError("db_path 不能为空")

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    # journal_mode 会改库头。取回结果并立刻提交，避免隐式事务一直占着写锁。
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.commit()
    if mode is None or mode[0] is None or str(mode[0]).lower() != "wal":
        log.warning("SQLite 日志模式是 %s，中断时可能留下锁定日志", mode)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            amplitude REAL,
            pct_change REAL,
            change REAL,
            turnover REAL,
            adjust TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (symbol, date)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date);
        """
    )
    conn.commit()
    # 上次进程被强杀时，WAL 会留在磁盘上。这里能合并就合并，打不开也不影响本次连接。
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        log.warning("打开数据库时合并 WAL 失败，可能仍有其他进程占用")
    return conn


def release_db(conn: sqlite3.Connection | None) -> None:
    """回滚未提交事务、合并 WAL 并关闭连接。中断后数据库不再保持锁定。"""
    if conn is None:
        return
    try:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        try:
            # 把已提交的数据合并回主库，并删掉 -wal/-shm，避免库一直处于锁定状态。
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _raise_interrupt(signum: int, _frame: object) -> None:
    """把 Ctrl+C 和进程终止都变成可清理的中断。"""
    if signum is None:
        return
    raise KeyboardInterrupt


def _install_stop_signals() -> dict | None:
    if threading.current_thread() is not threading.main_thread():
        return None
    return {
        sig: signal.signal(sig, _raise_interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }


def _restore_stop_signals(previous: dict | None) -> None:
    if not previous:
        return
    for sig, handler in previous.items():
        if sig is None or handler is None:
            continue
        signal.signal(sig, handler)


def fetch_stock_list() -> pd.DataFrame:
    """获取当前全 A 股代码与名称。"""
    raw = None
    for attempt in range(3):
        try:
            raw = ak.stock_info_a_code_name()
            break
        except Exception:
            if attempt == 2:
                raise
            wait = 2**attempt
            log.warning("股票清单获取失败，%d 秒后重试（%d/3）", wait, attempt + 1)
            time.sleep(wait)
    if raw is None or raw.empty:
        raise RuntimeError("AkShare 未返回 A 股股票清单")
    if not {"code", "name"}.issubset(raw.columns):
        raise ValueError(f"股票清单缺少 code/name 列，实际列: {list(raw.columns)}")

    stocks = raw[["code", "name"]].copy()
    stocks["symbol"] = stocks["code"].astype(str).str.strip().str.zfill(6)
    stocks["name"] = stocks["name"].fillna("").astype(str).str.strip()
    stocks = stocks[
        stocks["symbol"].str.fullmatch(r"\d{6}", na=False) & stocks["name"].ne("")
    ]
    return stocks[["symbol", "name"]].drop_duplicates("symbol").reset_index(drop=True)


def fetch_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取单只股票的前复权日线。"""
    if not symbol or not symbol.isdigit() or len(symbol) != 6:
        raise ValueError(f"股票代码必须是 6 位数字，实际为 {symbol!r}")

    # 腾讯源对沪深市场更稳定；北交所不受支持，直接使用东方财富源。
    is_beijing = symbol.startswith(("4", "8", "92"))
    exchange_symbol = f"{'sh' if symbol.startswith('6') else 'sz'}{symbol}"
    providers = []
    if not is_beijing:
        providers.append(
            (
                "腾讯",
                lambda: ak.stock_zh_a_hist_tx(
                    symbol=exchange_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                ),
            )
        )
    providers.append(
        (
            "东方财富",
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ),
        )
    )

    last_error: Exception | None = None
    for provider_name, request in providers:
        for attempt in range(3):
            try:
                raw = request()
                if raw is None:
                    raise RuntimeError(f"{provider_name}接口返回 None")
                return raw
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    wait = 2**attempt
                    log.warning(
                        "%s %s源失败，%d 秒后重试（%d/3）",
                        symbol,
                        provider_name,
                        wait,
                        attempt + 1,
                    )
                    time.sleep(wait)
        log.warning("%s %s源不可用，切换备用源", symbol, provider_name)

    raise RuntimeError(f"{symbol} 所有日线数据源均不可用") from last_error


def normalize_daily(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把 AkShare 中文列名转换为稳定的本地字段。"""
    if raw is None:
        raise ValueError(f"{symbol} 日线数据为 None")
    if raw.empty:
        return pd.DataFrame(columns=["date", "symbol", *_VALUE_COLUMNS])

    out = raw.rename(columns=_DAILY_COLUMNS).copy()
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"{symbol} 日线缺少列 {sorted(missing)}，实际列: {list(raw.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = symbol
    for column in _VALUE_COLUMNS:
        if column not in out.columns:
            out[column] = None
        out[column] = pd.to_numeric(out[column], errors="coerce")

    # 核心行情为空时无法用于研究，直接剔除，避免脏数据进入数据库。
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return (
        out[["date", "symbol", *_VALUE_COLUMNS]]
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _save_stocks(conn: sqlite3.Connection, stocks: pd.DataFrame) -> None:
    if conn is None:
        raise ValueError("SQLite 连接不能为空")
    if stocks is None or stocks.empty:
        raise ValueError("股票清单不能为空")

    now = datetime.now().isoformat(timespec="seconds")
    records = [(row.symbol, row.name, now) for row in stocks.itertuples(index=False)]
    with conn:
        conn.executemany(
            """
            INSERT INTO stocks(symbol, name, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                name=excluded.name,
                updated_at=excluded.updated_at
            """,
            records,
        )


def _replace_daily(
    conn: sqlite3.Connection,
    symbol: str,
    bars: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> int:
    if conn is None:
        raise ValueError("SQLite 连接不能为空")
    if bars is None or bars.empty:
        return 0

    now = datetime.now().isoformat(timespec="seconds")

    def nullable(value: object) -> object | None:
        return value if pd.notna(value) else None

    records = [
        (
            row.symbol,
            row.date,
            *(nullable(getattr(row, column)) for column in _VALUE_COLUMNS),
            "qfq",
            now,
        )
        for row in bars.itertuples(index=False)
    ]

    # 前复权因子会随除权事件改变，所以必须原子替换整个目标区间，不能只追加新日期。
    with conn:
        conn.execute(
            "DELETE FROM daily_bars WHERE symbol=? AND date BETWEEN ? AND ?",
            (symbol, _iso_date(start_date), _iso_date(end_date)),
        )
        conn.executemany(
            """
            INSERT INTO daily_bars(
                symbol, date, open, high, low, close, volume, amount,
                amplitude, pct_change, change, turnover, adjust, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    return len(records)


def _iso_date(value: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError("日期不能为空")
    return datetime.strptime(str(value).replace("-", ""), "%Y%m%d").strftime("%Y-%m-%d")


def _latest_bar_dates(conn: sqlite3.Connection) -> dict[str, str]:
    """每只股票本地已有的最后交易日。没有日线的股票不会出现在结果里。"""
    if conn is None:
        raise ValueError("SQLite 连接不能为空")
    rows = conn.execute("SELECT symbol, MAX(date) FROM daily_bars GROUP BY symbol").fetchall()
    return {symbol: day for symbol, day in rows if symbol and day}


def _sync_start(last_date: str | None, floor: str, end: str, overlap_days: int) -> str:
    """增量起点：已有数据从最后日期回退 overlap 天，新股从 floor 补全。

    前复权会改写最近的历史价格，所以要重叠回补，不能只追加最后一天之后的日期。
    # ponytail: 重叠窗口之外的旧前复权价格不会更新；除权漂移用一次不带 incremental 的全量同步校正。
    """
    if overlap_days < 0:
        raise ValueError("overlap_days 不能小于 0")
    floor_day = datetime.strptime(_iso_date(floor), "%Y-%m-%d").date()
    end_day = datetime.strptime(_iso_date(end), "%Y-%m-%d").date()
    if last_date is None or not str(last_date).strip():
        start = floor_day
    else:
        last = datetime.strptime(str(last_date)[:10], "%Y-%m-%d").date()
        start = last - timedelta(days=overlap_days)
        if start < floor_day:
            start = floor_day
    if start > end_day:
        start = end_day
    return start.strftime("%Y%m%d")


def _bare_symbol(symbol: str) -> str:
    """把 sh600000 / 600000 统一成库里的 6 位代码。"""
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    text = str(symbol).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if not text.isdigit() or len(text) != 6:
        raise ValueError(f"股票代码必须是 6 位数字或带 sh/sz/bj 前缀，实际为 {symbol!r}")
    return text


def sync_symbol(
    symbol: str,
    db_path: str | Path = "data/a_share.db",
    start_date: str = "19900101",
    end_date: str | None = None,
    name: str | None = None,
) -> int:
    """同步单只股票前复权日线到本地库，返回写入行数。"""
    if db_path is None or not str(db_path).strip():
        raise ValueError("db_path 不能为空")
    code = _bare_symbol(symbol)
    end = end_date or date.today().strftime("%Y%m%d")
    _iso_date(start_date)
    _iso_date(end)

    stock_name = str(name).strip() if name is not None and str(name).strip() else code
    conn = None
    try:
        conn = connect(db_path)
        _save_stocks(
            conn,
            pd.DataFrame([{"symbol": code, "name": stock_name}]),
        )
        raw = fetch_daily(code, start_date, end)
        bars = normalize_daily(raw, code)
        count = _replace_daily(conn, code, bars, start_date, end)
        log.info("%s：单票同步 %s ~ %s，%d 行", code, start_date, end, count)
        return count
    finally:
        release_db(conn)


def load_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    db_path: str | Path = "data/a_share.db",
) -> pd.DataFrame:
    """从本地 SQLite 读取单只股票前复权日线，列名与 AkQuant 直接兼容。"""
    if db_path is None or not str(db_path).strip():
        raise ValueError("db_path 不能为空")
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"本地数据库不存在：{path}，请先运行 scripts/sync_a_share.py")

    code = _bare_symbol(symbol)
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    conn = connect(path)
    try:
        df = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, volume
            FROM daily_bars
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            conn,
            params=(code, start, end),
        )
    finally:
        conn.close()

    if df is None or df.empty:
        raise LookupError(f"{code} 在 {start} ~ {end} 没有本地日线，请先同步该区间")

    # AkQuant 用 date 列识别时间，数值列必须是 float。
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    if df.empty:
        raise LookupError(f"{code} 在 {start} ~ {end} 的本地日线缺少有效 OHLC")
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """线程里只拉行情。SQLite 连接不能跨线程使用。"""
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    if start is None or not str(start).strip() or end is None or not str(end).strip():
        raise ValueError(f"{symbol} 同步区间不能为空")
    code = str(symbol).strip()
    raw = fetch_daily(code, str(start).strip(), str(end).strip())
    if raw is None:
        raise RuntimeError(f"{code} 日线数据为 None")
    return normalize_daily(raw, code)


def _silence_tx_progress():
    """腾讯源按年包了 tqdm，多线程时进度条会互相覆盖。"""
    try:
        import akshare.stock_feature.stock_hist_tx as tx_hist
    except Exception:
        log.warning("无法关闭腾讯进度条，并发日志可能被进度条打断")
        return None
    if tx_hist is None or not hasattr(tx_hist, "get_tqdm"):
        return None
    previous = tx_hist.get_tqdm
    tx_hist.get_tqdm = lambda enable=True: (lambda iterable, *args, **kwargs: iterable)
    return tx_hist, previous


def _restore_tx_progress(state) -> None:
    if not state:
        return
    tx_hist, previous = state
    if tx_hist is None or previous is None:
        return
    tx_hist.get_tqdm = previous


def _sync_concurrent(
    conn: sqlite3.Connection,
    stocks: pd.DataFrame,
    latest: dict[str, str],
    start_date: str,
    end: str,
    incremental: bool,
    overlap_days: int,
    workers: int,
) -> tuple[int, int, int]:
    """并发拉日线，主线程串行写库。返回 (成功数, 失败数, 写入行数)。"""
    if conn is None:
        raise RuntimeError("数据库连接为空，无法写入")
    if stocks is None or stocks.empty:
        raise ValueError("股票清单不能为空")
    if workers is None or workers < 1:
        raise ValueError("workers 必须大于 0")

    jobs: list[tuple[str, str, str]] = []
    failed = 0
    for stock in stocks.itertuples(index=False):
        # 空代码没法请求，记失败后继续，避免整批停住。
        if stock is None or not getattr(stock, "symbol", None) or not str(stock.symbol).strip():
            failed += 1
            log.error("股票代码为空，跳过")
            continue
        symbol = str(stock.symbol).strip()
        raw_name = getattr(stock, "name", None)
        name = str(raw_name).strip() if raw_name is not None and str(raw_name).strip() else symbol
        start = (
            _sync_start(latest.get(symbol) if latest is not None else None, start_date, end, overlap_days)
            if incremental
            else start_date
        )
        jobs.append((symbol, name, start))

    total = len(stocks)
    succeeded = rows = 0
    # ponytail: 全市场共用一个线程池。腾讯 16 并发稳定；东财 8 并发开始断连。
    # 回退失败变多时把 workers 降到 8，不要为东财再做一套限流。
    pool = ThreadPoolExecutor(max_workers=workers)
    quiet = _silence_tx_progress()
    try:
        future_map = {
            pool.submit(_load_bars, symbol, start, end): (symbol, name, start)
            for symbol, name, start in jobs
        }
        done = failed
        for fut in as_completed(future_map):
            done += 1
            meta = future_map.get(fut)
            if meta is None:
                failed += 1
                log.error("[%d/%d] 任务结果丢失", done, total)
                continue
            symbol, name, start = meta
            try:
                bars = fut.result()
            except KeyboardInterrupt:
                raise
            except Exception:
                failed += 1
                log.exception("[%d/%d] %s %s：同步失败", done, total, symbol, name)
                continue
            if bars is None:
                failed += 1
                log.error("[%d/%d] %s %s：日线数据为 None", done, total, symbol, name)
                continue
            try:
                count = _replace_daily(conn, symbol, bars, start, end)
            except Exception:
                failed += 1
                log.exception("[%d/%d] %s %s：写入失败", done, total, symbol, name)
                continue
            succeeded += 1
            rows += count
            log.info(
                "[%d/%d] %s %s：%s ~ %s，%d 行",
                done,
                total,
                symbol,
                name,
                start,
                end,
                count,
            )
    except KeyboardInterrupt:
        # 还没开始的请求取消掉；正在飞的请求不写库，已提交的股票保留。
        pool.shutdown(wait=False, cancel_futures=True)
        pool = None
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        _restore_tx_progress(quiet)
    return succeeded, failed, rows


def sync_all(
    db_path: str | Path,
    start_date: str = "19900101",
    end_date: str | None = None,
    delay: float = 0.1,
    limit: int | None = None,
    incremental: bool = False,
    overlap_days: int = 30,
    workers: int = 1,
) -> SyncResult:
    """同步全 A 股前复权日线；单只失败时记录日志并继续。

    incremental 为真时，已有股票只回补最后日期前的 overlap_days 天到今天；
    本地还没有日线的股票仍从 start_date 补全。
    workers 大于 1 时并发拉行情、主线程写库；这时不再按只 sleep，delay 不生效。
    """
    end = end_date or date.today().strftime("%Y%m%d")
    _iso_date(start_date)
    _iso_date(end)
    if delay is None or delay < 0:
        raise ValueError("delay 不能小于 0")
    if overlap_days < 0:
        raise ValueError("overlap_days 不能小于 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")
    # bool 是 int 的子类，True 会被当成 1 个线程，这里直接拒绝。
    if workers is None or isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers 必须是大于 0 的整数")

    conn = None
    interrupted = False
    succeeded = failed = rows = 0
    total = 0
    previous_signals = _install_stop_signals()
    try:
        stocks = fetch_stock_list()
        if stocks is None or stocks.empty:
            raise RuntimeError("股票清单为空，取消同步")
        if limit is not None:
            stocks = stocks.iloc[:limit].copy()

        conn = connect(db_path)
        _save_stocks(conn, stocks)
        # 水位在写股票清单之后读取；清单更新不会改动日线。
        latest = _latest_bar_dates(conn) if incremental else {}
        total = len(stocks)
        if workers == 1:
            # 单线程保留按只间隔，方便把请求压到接口几乎不并发的档位。
            for index, stock in enumerate(stocks.itertuples(index=False), start=1):
                start = (
                    _sync_start(latest.get(stock.symbol), start_date, end, overlap_days)
                    if incremental
                    else start_date
                )
                try:
                    raw = fetch_daily(stock.symbol, start, end)
                    bars = normalize_daily(raw, stock.symbol)
                    count = _replace_daily(conn, stock.symbol, bars, start, end)
                    succeeded += 1
                    rows += count
                    log.info(
                        "[%d/%d] %s %s：%s ~ %s，%d 行",
                        index,
                        total,
                        stock.symbol,
                        stock.name,
                        start,
                        end,
                        count,
                    )
                except Exception:
                    failed += 1
                    log.exception("[%d/%d] %s %s：同步失败", index, total, stock.symbol, stock.name)
                if delay > 0 and index < total:
                    time.sleep(delay)
        else:
            # 并发本身就是节流。再按只 sleep 的话，5000 只 × delay 会把加速吃掉。
            succeeded, failed, rows = _sync_concurrent(
                conn,
                stocks,
                latest,
                start_date,
                end,
                incremental,
                overlap_days,
                workers,
            )
    except KeyboardInterrupt:
        # 已提交的股票保留；正在写入的这只由 release_db 回滚，避免热日志把库锁住。
        interrupted = True
        log.warning("同步被中断，正在回滚当前写入并解除数据库锁定")
    finally:
        # 清理期间忽略第二次中断，确保连接一定被关闭。
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        release_db(conn)
        _restore_stop_signals(previous_signals)

    if interrupted:
        raise SyncInterrupted("同步已中断，已提交的数据已保留，数据库锁已释放")
    return SyncResult(total=total, succeeded=succeeded, failed=failed, rows=rows)
