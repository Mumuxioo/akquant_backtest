"""五线 S2/S3/S4 + 5/10/20/30 EMA 粘合中继突破。只读本地 data/a_share.db。

口径按 config/strategy_bond_break.yaml 和通达信量化公式：

- 阶段只允许 S2/S3/S4。S4 要「五线排列连续两根」，斜率只看当天相对 SLOPE_LAG 根之前。
- 买点全部在收盘确认：粘合、站上均线簇、中长期位置、放量、阳线、MACD 动能。
- 成交在下一根开盘。不用当天之后的任何价格。
- 硬止损：今日收盘和昨日收盘都低于「当天」E55×(1-容差)，下一开盘清仓。
- 止盈二选一：macd = DEA 上穿 DIF；rr = 按成交价的固定止损/止盈。

用法（在项目根目录）：

    python scripts/bond_break.py
    python scripts/bond_break.py backtest 002432 --start 2020-01-01 --exit macd
    python scripts/bond_break.py selfcheck
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Strategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a_share import connect
from src.utils.config_loader import load_config, resolve_path, section

DB_PATH = PROJECT_ROOT / "data" / "a_share.db"
DEFAULT_SYMBOL = "002432"
OUTPUT_ROOT = PROJECT_ROOT / "output"
LOG_NAME = "bond_break_log.txt"
REPORT_NAME = "bond_report.html"
REVIEW_NAME = "bond_review.html"
STREAM = False

# 公式默认值。网页和 config.yaml 的 bond_break 段可以覆盖。
TOL = 0.002
BOND_THRESH = 0.05
VOL_MULTIPLE = 1.5
VOL_MA = 5
RISE_MIN = 1.03
MIN_BAR_COUNT = 260
MIN_PHASE = 2
BOND_LOOKBACK = 1
BREAKOUT_BUFFER = 0.0
HIGH_LOOKBACK = 120
HIGH_POS_CAP = 0.85
YEAR_FLOOR = 0.90
SLOPE_LAG = 5
EMA_FAST = 5
EMA_MID = 13
EMA_TREND = 55
EMA_S3 = 125
EMA_LONG = 250
MA_FAST = 5
MA_10 = 10
MA_20 = 20
MA_30 = 30
MA_60 = 60
MA_120 = 120
MA_250 = 250
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
USE_HALF_YEAR = 1
USE_YEAR_FLOOR = 1
USE_TREND = 1
USE_HIGH_FILTER = 1
USE_VOLUME = 1
USE_BULL = 1
USE_MACD = 1
SINGLE_CAP = 0.15
STOP_PCT = 0.05
TARGET_PCT = 0.10
SLIPPAGE = 0.001
INITIAL_CASH = 1_000_000.0
LOT = 100
EXIT_MODE = "macd"       # macd：DEA 上穿 DIF；rr：固定盈亏比
HIGH_FIELD = "close"     # close=公式 HHV(C)；high=yaml 的 HHV(H)
# ponytail: 单票回测，yaml 的 max_open_count=6 没做。组合回测再在选股结果上分仓。

EMA_COLOR = {
    "e5": "#f5c542",
    "e13": "#e07a3d",
    "e55": "#3d8bfd",
    "e125": "#9b6bff",
    "e250": "#d6456a",
    "ma_high": "#22c58b",
    "ma_low": "#8b97b3",
}

TUNABLE = (
    "TOL", "BOND_THRESH", "VOL_MULTIPLE", "VOL_MA", "RISE_MIN",
    "MIN_BAR_COUNT", "MIN_PHASE", "BOND_LOOKBACK", "BREAKOUT_BUFFER",
    "HIGH_LOOKBACK", "HIGH_POS_CAP", "YEAR_FLOOR", "SLOPE_LAG",
    "EMA_FAST", "EMA_MID", "EMA_TREND", "EMA_S3", "EMA_LONG",
    "MA_FAST", "MA_10", "MA_20", "MA_30", "MA_60", "MA_120", "MA_250",
    "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL",
    "USE_HALF_YEAR", "USE_YEAR_FLOOR", "USE_TREND", "USE_HIGH_FILTER",
    "USE_VOLUME", "USE_BULL", "USE_MACD",
    "SINGLE_CAP", "STOP_PCT", "TARGET_PCT", "SLIPPAGE", "INITIAL_CASH", "LOT",
)
ENTRY_FLAGS = (
    "phase_ok", "bond", "broke", "half_year", "year_ok", "trend_ok",
    "not_high", "vol_ok", "bull", "macd_ok", "bars_ok", "tradable",
)
OPTIONAL_ENTRY_FLAGS = {
    "half_year": "USE_HALF_YEAR",
    "year_ok": "USE_YEAR_FLOOR",
    "trend_ok": "USE_TREND",
    "not_high": "USE_HIGH_FILTER",
    "vol_ok": "USE_VOLUME",
    "bull": "USE_BULL",
    "macd_ok": "USE_MACD",
}
ENGINE_COLUMNS = [
    "phase", "entry", *ENTRY_FLAGS,
    "below2", "cross_down",
    "e5", "e13", "e55", "e125", "e250", "ma_high", "ma_low",
    "dif", "dea", "macd_bar", "bond_width", "bond_age", "prev_close", "vol_ma", "vol",
]


def _finite(value) -> bool:
    """None、NaN、无穷都当成没有这个数。"""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _pct(value) -> str:
    if not _finite(value):
        return "无效"
    return f"{float(value):.2%}"


def _num(value, digits: int = 2) -> str:
    if not _finite(value):
        return "无效"
    return f"{float(value):.{digits}f}"


def _day_text(value) -> str:
    if value is None:
        return "未知日期"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)


def _phase_name(phase: int) -> str:
    names = {0: "S0其他", 1: "S1试探", 2: "S2三线", 3: "S3四线", 4: "S4五线"}
    return names.get(int(phase), f"S{phase}")


def _bar_day(timestamp) -> str:
    if timestamp is None:
        return "未知日期"
    try:
        stamp = pd.Timestamp(timestamp)
    except (TypeError, ValueError):
        return "未知日期"
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp.strftime("%Y-%m-%d")


def _ema(close: pd.Series, span: int) -> pd.Series:
    """通达信 EMA：第一根收盘做种子，adjust=False。只依赖当日及更早的收盘。"""
    if close is None or span is None or int(span) <= 0:
        raise ValueError("EMA 参数无效")
    return close.ewm(span=int(span), adjust=False).mean()


def _flag(mask: pd.Series) -> np.ndarray:
    """比较结果里的缺失一律当成不成立，避免 pd.NA 被当成真。"""
    if mask is None:
        raise ValueError("条件结果为空")
    return mask.fillna(False).astype(bool).to_numpy()


def is_st_name(name: str | None) -> bool:
    """通达信 NAMELIKE('S') 或 NAMELIKE('*')。没有名称时不当成 ST，调用方另写日志。"""
    if name is None:
        return False
    text = str(name).strip()
    if not text:
        return False
    upper = text.upper()
    return "S" in upper or "*" in text


def current_params() -> dict:
    module = globals()
    return {key.lower(): module[key] for key in TUNABLE}


def set_choices(exit_mode: str | None = None, high_field: str | None = None) -> None:
    """止盈方案和高位口径是字符串，不进数字参数表。"""
    global EXIT_MODE, HIGH_FIELD
    if exit_mode is not None and str(exit_mode).strip():
        mode = str(exit_mode).strip().lower()
        if mode not in ("macd", "rr"):
            raise ValueError("止盈方案只能是 macd 或 rr")
        EXIT_MODE = mode
    if high_field is not None and str(high_field).strip():
        field = str(high_field).strip().lower()
        if field not in ("close", "high"):
            raise ValueError("高位口径只能是 close 或 high")
        HIGH_FIELD = field


def apply_params(params: dict | None) -> None:
    """覆盖数字参数。未知键、非数字、以及会让公式失去意义的范围直接报错。"""
    if not params:
        return
    if not isinstance(params, dict):
        raise ValueError("参数必须是字典")
    module = globals()
    for raw_key, raw_value in params.items():
        key = str(raw_key or "").strip().upper()
        if key in ("EXIT_MODE", "HIGH_FIELD"):
            raise ValueError(f"{key} 要用 --exit / --high，不能放进数字参数")
        if key not in TUNABLE:
            raise ValueError(f"未知参数 {raw_key}")
        if isinstance(raw_value, bool) or not _finite(raw_value) or float(raw_value) < 0:
            raise ValueError(f"参数 {key} 必须是非负数字，实际 {raw_value!r}")
        if isinstance(module[key], int):
            if float(raw_value) != int(float(raw_value)) or int(float(raw_value)) < 1:
                raise ValueError(f"参数 {key} 必须是正整数，实际 {raw_value!r}")
            module[key] = int(float(raw_value))
        else:
            module[key] = float(raw_value)
    _validate_params()


def _increasing(names: tuple[str, ...], title: str) -> None:
    module = globals()
    spans = [int(module[name]) for name in names]
    if any(spans[i] >= spans[i + 1] for i in range(len(spans) - 1)):
        raise ValueError(f"{title}必须从短到长且互不相同，实际 {spans}")


def _validate_params() -> None:
    _increasing(("EMA_FAST", "EMA_MID", "EMA_TREND", "EMA_S3", "EMA_LONG"), "五线周期")
    _increasing(("MA_FAST", "MA_10", "MA_20", "MA_30"), "粘合均线")
    _increasing(("MA_60", "MA_120", "MA_250"), "趋势均线")
    if MACD_FAST >= MACD_SLOW:
        raise ValueError(f"MACD 快线 {MACD_FAST} 必须短于慢线 {MACD_SLOW}")
    if RISE_MIN < 1:
        raise ValueError(f"阳线涨幅 RISE_MIN 必须 ≥ 1，实际 {RISE_MIN}")
    if MIN_PHASE not in (2, 3, 4):
        raise ValueError(f"最低阶段 MIN_PHASE 只能是 2、3、4，实际 {MIN_PHASE}")
    if BOND_THRESH <= 0 or VOL_MULTIPLE <= 0 or HIGH_POS_CAP <= 0 or YEAR_FLOOR <= 0:
        raise ValueError("粘合阈值、放量倍数、高位上限、年线地板都必须大于 0")
    if BREAKOUT_BUFFER >= 1:
        raise ValueError(f"突破缓冲必须小于 1，实际 {BREAKOUT_BUFFER}")
    for name in OPTIONAL_ENTRY_FLAGS.values():
        if globals()[name] not in (0, 1):
            raise ValueError(f"{name} 只能是 0（关闭）或 1（启用）")
    if not 0 < SINGLE_CAP <= 1:
        raise ValueError(f"单票仓位必须在 (0, 1] 内，实际 {SINGLE_CAP}")
    if not 0 < STOP_PCT < 1:
        raise ValueError(f"固定止损必须在 (0, 1) 内，实际 {STOP_PCT}")
    if TARGET_PCT <= 0 or INITIAL_CASH <= 0:
        raise ValueError("止盈比例和初始资金必须大于 0")


def compute_phase(
    close: pd.Series,
    e5: pd.Series,
    e13: pd.Series,
    e55: pd.Series,
    e125: pd.Series,
    e250: pd.Series,
) -> np.ndarray:
    """通达信 PHASE。高阶段后写入，所以 S2 不会被「离 E55 不到 1%」的 S1 条件盖掉。

    S4 的两日确认只要求昨天和今天都满足 S4_BASE；55/125/250 的上行只看今天。
    """
    for series in (close, e5, e13, e55, e125, e250):
        if series is None:
            raise ValueError("阶段计算缺少收盘或均线")
    n = len(close)
    if not (n == len(e5) == len(e13) == len(e55) == len(e125) == len(e250)):
        raise ValueError("阶段计算的长度不一致")

    def above(price: pd.Series, line: pd.Series) -> pd.Series:
        # 均线缺失或 ≤0 不算站上，避免除零和前复权残值。
        return price.notna() & line.notna() & (line > 0) & (price > line * (1.0 + TOL))

    s2_raw = (e5 > e13) & (e13 > e55) & above(close, e55) & above(close.shift(1), e55.shift(1))
    s3_raw = (
        (e5 > e13) & (e13 > e55) & (e55 > e125)
        & above(close, e125) & above(close.shift(1), e125.shift(1))
    )
    s4_base = above(close, e5) & (e5 > e13) & (e13 > e55) & (e55 > e125) & (e125 > e250)
    lag = int(SLOPE_LAG)
    slopes = (e55 > e55.shift(lag)) & (e125 > e125.shift(lag)) & (e250 > e250.shift(lag))
    # 昨天只要排列在，斜率不必昨天也成立。
    s4_raw = s4_base.fillna(False) & slopes.fillna(False) & s4_base.shift(1).fillna(False)

    distance = (close - e55).abs() / e55.where(e55 > 0)
    near = close.notna() & e55.notna() & (e55 > 0) & ((close < e55) | (distance <= 0.01))
    s1 = (e5 > e13) & near

    phase = np.zeros(n, dtype=int)
    phase[_flag(s1)] = 1
    phase[_flag(s2_raw)] = 2
    phase[_flag(s3_raw)] = 3
    phase[_flag(s4_raw)] = 4
    return phase


def bond_parts(ma5: pd.Series, ma10: pd.Series, ma20: pd.Series, ma30: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """四条 EMA 的最高、最低，以及 (高-低)/低 < 阈值。分母 ≤0 不算粘合。"""
    for series in (ma5, ma10, ma20, ma30):
        if series is None:
            raise ValueError("粘合计算缺少均线")
    high = pd.concat([ma5, ma10, ma20, ma30], axis=1).max(axis=1)
    low = pd.concat([ma5, ma10, ma20, ma30], axis=1).min(axis=1)
    width = (high - low) / low.where(low > 0)
    bonded = high.notna() & low.notna() & (low > 0) & width.notna() & (width < BOND_THRESH)
    return high, low, bonded


def macd_kin(dif: pd.Series, dea: pd.Series, bar: pd.Series) -> pd.Series:
    """DIF>DEA 且柱不缩短，或连续两根红柱。柱用 2*(DIF-DEA)，乘 2 不改变这两个判断。"""
    if dif is None or dea is None or bar is None:
        raise ValueError("MACD 动能缺少 DIF、DEA 或柱")
    prev = bar.shift(1)
    kin1 = dif.notna() & dea.notna() & bar.notna() & prev.notna() & (dif > dea) & (bar >= prev)
    kin2 = bar.notna() & prev.notna() & (bar > 0) & (prev > 0)
    return (kin1 | kin2).fillna(False)


def macd_cross_down(dif: pd.Series, dea: pd.Series) -> pd.Series:
    """通达信 CROSS(DEA, DIF)：昨天 DEA 还在 DIF 下（或相等），今天 DEA 上穿 DIF。"""
    if dif is None or dea is None:
        raise ValueError("死叉判断缺少 DIF 或 DEA")
    prev_dif = dif.shift(1)
    prev_dea = dea.shift(1)
    crossed = (
        dif.notna() & dea.notna() & prev_dif.notna() & prev_dea.notna()
        & (prev_dea <= prev_dif) & (dea > dif)
    )
    return crossed.fillna(False)


def below_two(close: pd.Series, e55: pd.Series) -> pd.Series:
    """今日收盘和昨日收盘都低于「当天」E55×(1-容差)。昨日不跟昨天那根 E55 比。"""
    if close is None or e55 is None:
        raise ValueError("止损判断缺少收盘或 E55")
    level = e55 * (1.0 - TOL)
    prev = close.shift(1)
    hit = close.notna() & prev.notna() & e55.notna() & (e55 > 0) & (close < level) & (prev < level)
    return hit.fillna(False)


def bull_flag(open_: pd.Series, close: pd.Series) -> pd.Series:
    """阳线，且收盘 / 前收 ≥ RISE_MIN。前收 ≤0 不算。"""
    if open_ is None or close is None:
        raise ValueError("阳线判断缺少开盘或收盘")
    prev = close.shift(1)
    return (
        open_.notna() & close.notna() & prev.notna() & (prev > 0)
        & (close > open_) & (close / prev >= RISE_MIN)
    ).fillna(False)


def volume_flag(volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    """通达信 MA(VOL, N) 含当天。这是收盘才知道的量，不是未来数据。"""
    if volume is None:
        raise ValueError("放量判断缺成交量")
    ma = volume.rolling(int(VOL_MA), min_periods=int(VOL_MA)).mean()
    ok = volume.notna() & ma.notna() & (ma > 0) & (volume > ma * VOL_MULTIPLE)
    return ma, ok.fillna(False)


def combine_entry(frame: pd.DataFrame, is_st: bool) -> pd.Series:
    """核心条件和已启用的过滤条件同时成立。ST 一票否决。"""
    if frame is None or frame.empty:
        raise ValueError("没有日线，无法合成买点")
    if is_st:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for key in ENTRY_FLAGS:
        switch = OPTIONAL_ENTRY_FLAGS.get(key)
        if switch is not None and globals()[switch] == 0:
            continue
        if key not in frame.columns:
            raise ValueError(f"买点缺少 {key}")
        column = frame[key]
        if column is None:
            raise ValueError(f"买点列 {key} 为空")
        mask &= column.fillna(False).astype(bool)
    return mask


def prepare(raw: pd.DataFrame, is_st: bool = False) -> pd.DataFrame:
    """在全部历史上算指标和买点。rolling / shift / EMA 都只看当日及以前。"""
    if raw is None or raw.empty:
        raise ValueError("日线为空")
    frame = raw.sort_values("date").reset_index(drop=True).copy()
    for column in ("date", "open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            raise ValueError(f"日线缺少 {column}")
    frame["volume"] = frame["volume"].fillna(0.0)
    close = frame["close"]

    frame["e5"] = _ema(close, EMA_FAST)
    frame["e13"] = _ema(close, EMA_MID)
    frame["e55"] = _ema(close, EMA_TREND)
    frame["e125"] = _ema(close, EMA_S3)
    frame["e250"] = _ema(close, EMA_LONG)
    frame["phase"] = compute_phase(
        close, frame["e5"], frame["e13"], frame["e55"], frame["e125"], frame["e250"]
    )
    frame["phase_ok"] = frame["phase"] >= int(MIN_PHASE)

    frame["ma5"] = _ema(close, MA_FAST)
    frame["ma10"] = _ema(close, MA_10)
    frame["ma20"] = _ema(close, MA_20)
    frame["ma30"] = _ema(close, MA_30)
    frame["ma60"] = _ema(close, MA_60)
    frame["ma120"] = _ema(close, MA_120)
    frame["ma250"] = _ema(close, MA_250)
    high, low, bonded = bond_parts(frame["ma5"], frame["ma10"], frame["ma20"], frame["ma30"])
    frame["ma_high"] = high
    frame["ma_low"] = low
    frame["bond_width"] = (high - low) / low.where(low > 0)
    # 粘合后通常会先扩散再突破；允许最近 N 根内出现过粘合，避免只认“当天仍粘合”的过严信号。
    frame["bond"] = bonded.rolling(int(BOND_LOOKBACK), min_periods=1).max().fillna(False).astype(bool)
    last_bond = pd.Series(np.where(bonded, np.arange(len(frame)), np.nan), index=frame.index).ffill()
    frame["bond_age"] = pd.Series(np.arange(len(frame)), index=frame.index) - last_bond
    frame["bond_age"] = frame["bond_age"].where(frame["bond"])
    frame["broke"] = close.notna() & high.notna() & (close > high * (1.0 + BREAKOUT_BUFFER))

    # 中长期：站上半年线、不低于年线的 YEAR_FLOOR、60 线向上或走平、远离 120 日高点。
    frame["half_year"] = close.notna() & frame["ma120"].notna() & (frame["ma120"] > 0) & (close > frame["ma120"])
    frame["year_ok"] = (
        close.notna() & frame["ma250"].notna() & (frame["ma250"] > 0)
        & (close > frame["ma250"] * YEAR_FLOOR)
    )
    prev_ma60 = frame["ma60"].shift(1)
    frame["trend_ok"] = (
        close.notna() & frame["ma60"].notna() & prev_ma60.notna()
        & (close > frame["ma60"]) & (frame["ma60"] >= prev_ma60)
    )
    src = frame["high"] if HIGH_FIELD == "high" else close
    if src is None:
        raise ValueError("高位口径没有价格")
    hhv = src.rolling(int(HIGH_LOOKBACK), min_periods=int(HIGH_LOOKBACK)).max()
    frame["not_high"] = close.notna() & hhv.notna() & (hhv > 0) & (close < hhv * HIGH_POS_CAP)

    frame["vol_ma"], frame["vol_ok"] = volume_flag(frame["volume"])
    frame["bull"] = bull_flag(frame["open"], close)
    dif = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    dea = _ema(dif, MACD_SIGNAL)
    macd_bar = 2.0 * (dif - dea)
    frame["dif"] = dif
    frame["dea"] = dea
    frame["macd_bar"] = macd_bar
    frame["macd_ok"] = macd_kin(dif, dea, macd_bar)
    frame["cross_down"] = macd_cross_down(dif, dea)
    frame["below2"] = below_two(close, frame["e55"])
    frame["prev_close"] = close.shift(1)
    frame["vol"] = frame["volume"]
    # BARSCOUNT(C) > N：第 N+1 根起才允许买。用全历史序号，不用截取后的区间重新计数。
    frame["bars_ok"] = np.arange(1, len(frame) + 1) > int(MIN_BAR_COUNT)
    frame["tradable"] = frame["volume"].notna() & (frame["volume"] > 0)
    frame["entry"] = combine_entry(frame, is_st)
    return frame


def size_order(equity: float, cash: float, price: float) -> tuple[int, str]:
    """单票不超过权益的 SINGLE_CAP，再按现金和手数取整。留出佣金，避免差 5 元被拒。"""
    if not (_finite(equity) and _finite(cash) and _finite(price)):
        return 0, "权益、现金或价格无效，算不出股数"
    if equity <= 0 or price <= 0:
        return 0, "权益或价格不是正数，无法下单"
    budget = min(float(equity) * SINGLE_CAP, float(cash))
    raw_qty = int(budget // price // LOT * LOT) if price > 0 else 0
    text = (
        f"权益 {_num(equity, 0)}，单票上限 {SINGLE_CAP:.0%} 即 {_num(equity * SINGLE_CAP, 0)}，"
        f"现金 {_num(cash, 0)}，按收盘 {_num(price)} 可买 {raw_qty} 股。"
    )
    if raw_qty < LOT:
        return 0, text + f"不足 1 手（{LOT} 股），这次不下单。"
    buffer = max(5.0, raw_qty * price * 0.0003)
    if raw_qty * price + buffer > cash:
        raw_qty = int((cash - buffer) // price // LOT * LOT) if price > 0 else 0
        text += f"扣掉佣金预留 {_num(buffer)} 后剩 {raw_qty} 股。"
    if raw_qty < LOT:
        return 0, text + f"留出佣金后不足 1 手（{LOT} 股），这次不下单。"
    return raw_qty, text


class Journal:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str) -> None:
        if text is None:
            return
        self.lines.append(str(text))
        if STREAM:
            print(str(text), flush=True)

    def blank(self) -> None:
        self.lines.append("")
        if STREAM:
            print("", flush=True)

    def save(self, path: Path) -> None:
        if path is None:
            raise ValueError("日志路径不能为空")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _call_number(strategy, name: str) -> float | None:
    if strategy is None or not name:
        return None
    value = getattr(strategy, name, None)
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    if not _finite(value):
        return None
    return float(value)


def _position_qty(strategy, symbol: str) -> float:
    if strategy is None or symbol is None:
        return 0.0
    pos = strategy.get_position(symbol)
    if pos is None:
        return 0.0
    if isinstance(pos, (int, float)):
        if not _finite(pos):
            return 0.0
        return float(pos)
    for field in ("quantity", "size", "volume"):
        qty = getattr(pos, field, None)
        if _finite(qty):
            return float(qty)
    return 0.0


def _on(extra: dict, key: str) -> bool:
    if extra is None or key not in extra:
        return False
    value = extra.get(key)
    return _finite(value) and float(value) >= 1.0


def to_engine_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """收成 AKQuant 日线。信号列在收盘已经算完，引擎只负责下一根开盘撮合。"""
    if frame is None or frame.empty:
        raise ValueError("没有日线，无法交给回测引擎")
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    missing = [column for column in ("date", "open", "high", "low", "close", "volume", *ENGINE_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"回测数据缺少列：{missing}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize() + pd.Timedelta(hours=15)
    out["symbol"] = str(symbol).zfill(6)
    view = out[["date", "open", "high", "low", "close", "volume", "symbol", *ENGINE_COLUMNS]].copy()
    for column in ENGINE_COLUMNS:
        view[column] = pd.to_numeric(view[column], errors="coerce")
    view[ENGINE_COLUMNS] = view[ENGINE_COLUMNS].astype(float).fillna(0.0)
    return view


class BondBreakStrategy(Strategy):
    """收盘读预计算买点；止损止盈也在收盘确认，交给引擎下一根开盘成交。"""

    def __init__(self):
        super().__init__()
        self.journal = Journal()
        self.code = ""
        self.name = ""
        self.total = 0
        self.seen = 0
        self.armed = False
        self.prev_phase = 0
        self.entry_price = None
        self.exit_reason = None
        self.buys = 0
        self.sells = 0
        self.signals = 0
        self.too_small = 0
        self._bar = None
        self.blocked = {key: 0 for key in ENTRY_FLAGS}
        self.phase_days = {2: 0, 3: 0, 4: 0}

    def on_trade(self, trade):
        if trade is None:
            return
        side = str(getattr(trade, "side", "") or "").lower()
        price = getattr(trade, "price", None)
        qty = getattr(trade, "quantity", None)
        day = _bar_day(getattr(trade, "timestamp", None))
        if "buy" in side:
            self.buys += 1
            self.entry_price = float(price) if _finite(price) and float(price) > 0 else None
            stop = None
            target = None
            if self.entry_price is not None:
                stop = self.entry_price * (1.0 - STOP_PCT)
                target = self.entry_price * (1.0 + TARGET_PCT)
            self.journal.add(
                f"{day} 成交买入 {_num(qty, 0)} 股，开盘成交价 {_num(price)}。"
                f"固定止损 {_num(stop)}，固定止盈 {_num(target)}（仅方案 rr 使用）。"
            )
            return
        if "sell" in side:
            self.sells += 1
            self.journal.add(
                f"{day} 成交卖出 {_num(qty, 0)} 股，开盘成交价 {_num(price)}。{self.exit_reason or ''}"
            )
            self.entry_price = None
            self.exit_reason = None

    def on_reject(self, order):
        if order is None:
            self.journal.add("有一张单被引擎拒绝，但没有拒单详情。")
            return
        reason = getattr(order, "reason", None) or getattr(order, "message", None) or order
        self.journal.add(f"引擎拒单：{reason}")

    def _log(self, text: str) -> None:
        if text is None:
            return
        bar = self._bar
        day = _bar_day(getattr(bar, "timestamp", None)) if bar is not None else ""
        if bar is not None and day and text.startswith(day + " "):
            text = f"{day} 开 {_num(getattr(bar, 'open', None))} 收 {_num(bar.close)}{text[len(day):]}"
        self.journal.add(text)

    def _plot(self, extra: dict) -> None:
        if extra is None:
            return
        roles = (
            ("e5", EMA_FAST, "e5"),
            ("e13", EMA_MID, "e13"),
            ("e55", EMA_TREND, "e55"),
            ("e125", EMA_S3, "e125"),
            ("e250", EMA_LONG, "e250"),
            ("ma_high", None, "ma_high"),
            ("ma_low", None, "ma_low"),
        )
        for column, span, color_key in roles:
            value = extra.get(column)
            if not _finite(value):
                continue
            label = f"EMA{int(span)}" if span is not None else ("簇高" if column == "ma_high" else "簇低")
            self.record_indicator(
                name=column, value=float(value), display_name=label,
                pane=0, render_type="line", color=EMA_COLOR[color_key], precision=2,
            )
        # 复盘图的成交量占用第 1 幅副图。这里只补 MACD，避免再叠一根成交量。
        dif = extra.get("dif")
        if _finite(dif):
            self.record_indicator(
                name="dif", value=float(dif), display_name="DIF",
                pane=1, render_type="line", color="#3d8bfd", precision=3,
            )
        dea = extra.get("dea")
        if _finite(dea):
            self.record_indicator(
                name="dea", value=float(dea), display_name="DEA",
                pane=1, render_type="line", color="#f5c542", precision=3,
            )
        macd_bar = extra.get("macd_bar")
        if _finite(macd_bar):
            value = float(macd_bar)
            rising = value >= 0
            self.record_indicator(
                name="macd_up" if rising else "macd_down",
                value=value,
                display_name="MACD红柱" if rising else "MACD绿柱",
                pane=1, render_type="histogram",
                color="#ff5d6c" if rising else "#22c58b",
                precision=3,
            )

    def _entry_price(self, symbol: str) -> float | None:
        if _finite(self.entry_price) and float(self.entry_price) > 0:
            return float(self.entry_price)
        if self.ctx is None or symbol is None:
            return None
        entry = self.ctx.get_position_entry_price(symbol)
        if _finite(entry) and float(entry) > 0:
            return float(entry)
        return None

    def _try_close(self, symbol: str, day: str, reason: str) -> None:
        self.exit_reason = reason
        available = self.get_available_position(symbol)
        if available is None or available <= 0:
            self._log(f"{day} 还不能卖（T+1 或可卖数量为 0），清仓意图保留。{reason}")
            return
        self.close_position(symbol=symbol)
        self._log(f"{day} 下一根开盘清仓。{reason}")

    def _exit_text(self, extra: dict, close: float, symbol: str) -> str | None:
        """硬止损优先于止盈。返回 None 表示今天不平仓。"""
        if extra is None:
            return None
        e55 = extra.get("e55")
        if _on(extra, "below2"):
            level = float(e55) * (1.0 - TOL) if _finite(e55) else None
            return (
                f"硬止损：收盘 {_num(close)}、昨日收盘 {_num(extra.get('prev_close'))}"
                f" 都低于当天 EMA{EMA_TREND}×(1-{TOL:.3f}) = {_num(level)}"
            )
        if EXIT_MODE == "macd":
            if _on(extra, "cross_down"):
                return (
                    f"MACD 离场：DEA {_num(extra.get('dea'), 3)} 上穿 DIF {_num(extra.get('dif'), 3)}"
                )
            return None
        entry = self._entry_price(symbol)
        if entry is None:
            return None
        stop = entry * (1.0 - STOP_PCT)
        target = entry * (1.0 + TARGET_PCT)
        if close <= stop:
            return f"固定止损：收盘 {_num(close)} ≤ 成交价 {_num(entry)}×(1-{STOP_PCT:.0%}) = {_num(stop)}"
        if close >= target:
            return f"固定止盈：收盘 {_num(close)} ≥ 成交价 {_num(entry)}×(1+{TARGET_PCT:.0%}) = {_num(target)}"
        return None

    def _try_buy(self, symbol: str, day: str, close: float, extra: dict) -> None:
        if extra is None:
            extra = {}
        self.signals += 1
        width = extra.get("bond_width") if extra else None
        bond_age = extra.get("bond_age") if extra else None
        prev = extra.get("prev_close") if extra else None
        rise = (close / float(prev) - 1.0) if _finite(prev) and float(prev) > 0 else None
        bond_text = "当天粘合" if _finite(bond_age) and int(float(bond_age)) == 0 else f"{_num(bond_age, 0)} 根前粘合"
        volume_text = (
            f"量 {_num(extra.get('vol'), 0)} > 均量 {_num(extra.get('vol_ma'), 0)}×{VOL_MULTIPLE:g}"
            if USE_VOLUME else "量能过滤已关闭"
        )
        momentum_text = f"涨幅 {_pct(rise)}，MACD 通过" if USE_MACD else f"涨幅 {_pct(rise)}，MACD 过滤已关闭"
        self._log(
            f"{day} 量化买点。阶段 {_phase_name(int(extra.get('phase') or 0))}，"
            f"{bond_text}，当前簇宽 {_pct(width)}（阈值 {_pct(BOND_THRESH)}），"
            f"收盘 {_num(close)} 站上簇高 {_num(extra.get('ma_high'))}，"
            f"{volume_text}，{momentum_text}。"
        )
        equity = _call_number(self, "equity") or INITIAL_CASH
        cash = _call_number(self, "cash")
        if cash is None:
            cash = equity
        qty, text = size_order(equity, cash, close)
        self.journal.add(f"    {text}")
        if qty < LOT:
            self.too_small += 1
            self.journal.add("    结论：今天不报单。")
            return
        receipt = self.buy(symbol=symbol, quantity=qty)
        if receipt is None:
            self.journal.add("    结论：引擎没有收下这张买单。")
            return
        self.journal.add(f"    结论：已提交买单 {qty} 股，下一根开盘撮合。")

    def on_bar(self, bar):
        if bar is None:
            return
        extra = bar.extra or {}
        symbol = bar.symbol
        if symbol is None:
            return
        self._bar = bar
        self.seen += 1
        close = bar.close
        day = _bar_day(bar.timestamp)
        phase = int(extra.get("phase") or 0)
        bars_ok = _on(extra, "bars_ok")
        self._plot(extra)
        if phase in self.phase_days and bars_ok:
            self.phase_days[phase] += 1
        if bars_ok and not self.armed:
            self.armed = True
            self._log(f"{day} 已满 {MIN_BAR_COUNT} 根 K 线，当前阶段 {_phase_name(phase)}。之后才允许新开仓。")
        elif bars_ok and phase != self.prev_phase and (phase >= 2 or self.prev_phase >= 2):
            self._log(f"{day} 阶段 {_phase_name(self.prev_phase)} → {_phase_name(phase)}。")
        self.prev_phase = phase
        if self.total and self.seen % 500 == 0:
            self.journal.add(f"{day} 回测进行中 {self.seen}/{self.total}")

        # 上一根收盘已经决定离场时，不依赖今天的收盘是否有效，继续把卖单交出去。
        qty = _position_qty(self, symbol)
        open_orders = self.get_open_orders(symbol=symbol) or []
        if not open_orders and self.exit_reason and qty > 0:
            self._try_close(symbol, day, self.exit_reason)
            return
        if close is None or not _finite(close) or float(close) <= 0:
            self.journal.add(f"{day} 收盘价无效，当天不产生新信号。")
            return
        close = float(close)

        if bars_ok and phase >= MIN_PHASE and not _on(extra, "entry"):
            for key in ENTRY_FLAGS:
                switch = OPTIONAL_ENTRY_FLAGS.get(key)
                if switch is not None and globals()[switch] == 0:
                    continue
                if not _on(extra, key):
                    self.blocked[key] += 1

        if qty > 0 and not open_orders and self.exit_reason is None:
            reason = self._exit_text(extra, close, symbol)
            if reason:
                self._try_close(symbol, day, reason)
                return
        if _on(extra, "entry") and qty <= 0 and not open_orders and self.exit_reason is None:
            self._try_buy(symbol, day, close, extra)
            return
        if _on(extra, "entry") and qty > 0:
            self._log(f"{day} 量化买点再次出现，已有持仓，不加仓。")

    def on_stop(self):
        equity = _call_number(self, "equity")
        self.journal.blank()
        self.journal.add("========== 回测结束 ==========")
        self.journal.add(
            f"初始资金 {_num(INITIAL_CASH, 0)}，期末权益 {_num(equity, 0)}。"
            "权益、回撤和买卖点以报告为准。"
        )
        self.journal.add(
            f"买点 {self.signals} 次，买入成交 {self.buys} 次，卖出成交 {self.sells} 次。"
        )
        self.journal.add(
            f"阶段天数：S2 {self.phase_days[2]}，S3 {self.phase_days[3]}，S4 {self.phase_days[4]}。"
        )
        blocked = {key: count for key, count in self.blocked.items() if count}
        if blocked:
            self.journal.add(f"S{MIN_PHASE}+ 但没买成的天数（同一天可计多项）：")
            labels = {
                "phase_ok": "阶段不够", "bond": "没有粘合", "broke": "没站上均线簇",
                "half_year": "低于 MA120", "year_ok": "弱于年线地板", "trend_ok": "MA60 不配合",
                "not_high": "靠近 120 日高点", "vol_ok": "量能不够", "bull": "阳线或涨幅不够",
                "macd_ok": "MACD 动能不够", "bars_ok": "K 线不足", "tradable": "停牌或无成交",
            }
            for key, count in blocked.items():
                self.journal.add(f"    {labels.get(key, key)}：{count}")
        if self.too_small:
            self.journal.add(f"买点出现但不足 1 手：{self.too_small} 次。")


def load_bars(conn, symbol: str) -> pd.DataFrame:
    if conn is None:
        raise ValueError("数据库连接不能为空")
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    code = str(symbol).strip().zfill(6)
    frame = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM daily_bars
        WHERE symbol = ?
        ORDER BY date
        """,
        conn,
        params=(code,),
    )
    if frame is None or frame.empty:
        raise LookupError(f"{code} 在本地库没有日线")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    if frame.empty:
        raise LookupError(f"{code} 的本地日线没有有效 OHLC")
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame


def stock_name(conn, symbol: str) -> str:
    if conn is None or symbol is None:
        return ""
    row = conn.execute("SELECT name FROM stocks WHERE symbol = ?", (str(symbol).zfill(6),)).fetchone()
    if row is None or row[0] is None:
        return ""
    return str(row[0]).strip()


def clip_range(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """指标已用全历史算完。这里只把回测区间切出来，开始日之前的 K 线只起预热作用。"""
    if frame is None or frame.empty:
        raise ValueError("没有日线，无法截取区间")
    mask = pd.Series(True, index=frame.index)
    if start and str(start).strip():
        mask &= frame["date"] >= pd.Timestamp(str(start).strip())
    if end and str(end).strip():
        mask &= frame["date"] <= pd.Timestamp(str(end).strip())
    out = frame[mask].reset_index(drop=True)
    if out.empty:
        raise LookupError(f"区间 {start or '最早'} ~ {end or '最新'} 内没有日线")
    return out


def output_dir(symbol: str) -> Path:
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    return OUTPUT_ROOT / str(symbol).strip().zfill(6)


def _stack_report_kline(fig, data):
    """报告 K 线原图只有价格和成交量。底部拆成成交量、MACD 两幅副图，刻度分开。"""
    if fig is None or data is None or not hasattr(data, "columns"):
        return fig
    if any(name not in data.columns for name in ("dif", "dea", "macd_bar")):
        return fig
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rebuilt = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.20, 0.22],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )
    for trace in fig.data or []:
        if trace is None:
            continue
        name = str(getattr(trace, "name", "") or "")
        if name == "Volume":
            trace.name = "成交量"
            rebuilt.add_trace(trace, row=2, col=1)
        else:
            rebuilt.add_trace(trace, row=1, col=1)
    bars = pd.to_numeric(data["macd_bar"], errors="coerce")
    colors = []
    for value in bars:
        if not _finite(value):
            colors.append("#8b97b3")
        elif float(value) >= 0:
            colors.append("#ff5d6c")
        else:
            colors.append("#22c58b")
    index = data.index
    rebuilt.add_trace(
        go.Bar(
            x=index, y=bars, name="MACD",
            marker=dict(color=colors, line=dict(width=0)),
        ),
        row=3, col=1,
    )
    rebuilt.add_trace(
        go.Scatter(
            x=index, y=pd.to_numeric(data["dif"], errors="coerce"),
            name="DIF", mode="lines", line=dict(color="#3d8bfd", width=1.5),
        ),
        row=3, col=1,
    )
    rebuilt.add_trace(
        go.Scatter(
            x=index, y=pd.to_numeric(data["dea"], errors="coerce"),
            name="DEA", mode="lines", line=dict(color="#f5c542", width=1.5),
        ),
        row=3, col=1,
    )
    title = fig.layout.title.text if fig.layout is not None and fig.layout.title is not None else None
    rebuilt.update_layout(
        title=title,
        height=800,
        template="plotly_white",
        dragmode="zoom",
    )
    rebuilt.update_xaxes(rangeslider_visible=False)
    rebuilt.update_yaxes(title_text="成交量", row=2, col=1)
    rebuilt.update_yaxes(title_text="MACD", row=3, col=1)
    return rebuilt


def run_backtest(symbol: str, start: str | None = None, end: str | None = None, show: bool = True) -> Path:
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"本地数据库不存在：{DB_PATH}")
    code = str(symbol).strip().zfill(6)
    conn = connect(DB_PATH)
    try:
        name = stock_name(conn, code)
        raw = load_bars(conn, code)
    finally:
        conn.close()
    warnings.filterwarnings("ignore", message="invalid value encountered in", category=RuntimeWarning)
    st = is_st_name(name)
    full = prepare(raw, is_st=st)
    frame = clip_range(full, start, end)
    journal = Journal()
    journal.add(f"五线粘合突破 · {code} {name}".rstrip())
    if not name:
        journal.add("本地库没有这只股票的名称，ST 过滤无法判断，本次不按 ST 剔除。")
    elif st:
        journal.add(f"名称「{name}」含 S 或 *，按 ST/*ST 剔除，本次不产生买点。")
    journal.add(
        f"区间 {_day_text(frame['date'].iloc[0])} ~ {_day_text(frame['date'].iloc[-1])}，共 {len(frame)} 根日线。"
        f"全历史 {len(full)} 根，指标在截取前就算完。"
    )
    journal.add(
        "执行：收盘确认买点和卖点，AKQuant 在下一根开盘撮合。"
        f"佣金万三、最低 5 元、卖出印花税万五、过户费万一、滑点 {SLIPPAGE:.2%}。单票仓位 {SINGLE_CAP:.0%}。"
    )
    high_text = f"{HIGH_LOOKBACK} 日最高收盘" if HIGH_FIELD == "close" else f"{HIGH_LOOKBACK} 日最高价"
    exit_text = "止盈看 DEA 上穿 DIF" if EXIT_MODE == "macd" else f"止盈止损按成交价 -{STOP_PCT:.0%}/+{TARGET_PCT:.0%}"
    enabled = [
        label for flag, label in (
            (USE_HALF_YEAR, "半年线"), (USE_YEAR_FLOOR, "年线地板"), (USE_TREND, "MA60趋势"),
            (USE_HIGH_FILTER, "高位"), (USE_VOLUME, "量能"), (USE_BULL, "阳线"), (USE_MACD, "MACD"),
        ) if flag
    ]
    journal.add(
        f"最低阶段 S{MIN_PHASE}，粘合后 {BOND_LOOKBACK} 根内突破有效，突破缓冲 {BREAKOUT_BUFFER:.2%}。"
        f"启用过滤：{'、'.join(enabled) if enabled else '无'}。"
    )
    journal.add(f"高位口径为{high_text}的 {HIGH_POS_CAP:.0%}。硬止损两根收盘低于当天 E55。{exit_text}。")
    journal.add("参数：" + json.dumps(
        {**current_params(), "exit_mode": EXIT_MODE, "high_field": HIGH_FIELD},
        ensure_ascii=False,
    ))
    journal.blank()
    journal.add("========== 信号与成交 ==========")
    data = to_engine_frame(code, frame)
    strategy = BondBreakStrategy()
    strategy.journal = journal
    strategy.code = code
    strategy.name = name
    strategy.total = len(data)
    result = aq.run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=INITIAL_CASH,
        symbols=code,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        min_commission=5.0,
        slippage={"type": "percent", "value": SLIPPAGE},
        lot_size=LOT,
        t_plus_one=True,
        timezone="Asia/Shanghai",
    )
    if result is None or getattr(result, "viz", None) is None:
        raise RuntimeError("回测引擎没有返回可视化结果")
    folder = output_dir(code)
    path = folder / LOG_NAME
    journal.save(path)
    benchmark = data.set_index("date")["close"].pct_change().fillna(0.0).rename("BUY_AND_HOLD")
    report_path = folder / REPORT_NAME
    import akquant.plot.report as report_mod
    previous_plot = report_mod.plot_strategy

    def _plot_strategy_macd(result, symbol, data, **kwargs):
        # 引擎先画价格+成交量，这里把同一张图的底部换成成交量和 MACD。
        fig = previous_plot(result, symbol, data, **kwargs)
        return _stack_report_kline(fig, data)

    report_mod.plot_strategy = _plot_strategy_macd
    try:
        result.viz.report(
            title=f"{code} {name} · 五线粘合突破".rstrip(),
            filename=str(report_path),
            show=show,
            benchmark=benchmark,
            market_data=data,
            plot_symbol=code,
            include_trade_kline=True,
        )
    finally:
        report_mod.plot_strategy = previous_plot
    review_path = folder / REVIEW_NAME
    result.viz.review(
        market_data=data,
        title=f"{code} · 粘合突破复盘",
        filename=str(review_path),
        theme="light",
        initial_symbol=code,
        show=False,
    )
    print(f"{code} {name} 回测结束：买点 {strategy.signals}，买入 {strategy.buys}，卖出 {strategy.sells}。", flush=True)
    print(f"日志：{path}", flush=True)
    print(f"报告：{report_path}", flush=True)
    print(f"复盘：{review_path}", flush=True)
    return report_path


def _snapshot() -> dict:
    return {**current_params(), "exit_mode": EXIT_MODE, "high_field": HIGH_FIELD}


def _restore(saved: dict) -> None:
    if saved is None:
        raise ValueError("无法恢复空参数")
    set_choices(saved.get("exit_mode"), saved.get("high_field"))
    apply_params({key: value for key, value in saved.items() if str(key).upper() in TUNABLE})


def self_check() -> None:
    """不连数据库。阶段优先级、粘合、含当天的均量、止损口径或无未来函数被改坏时会失败。"""
    saved = _snapshot()
    try:
        _check_rules(saved)
    finally:
        _restore(saved)
    print("selfcheck 通过")


def _check_rules(saved: dict) -> None:
    if saved is None:
        raise AssertionError("自检缺少参数快照")
    # 收盘只比 E55 高 0.5%：站上了 0.2% 容差，也落在 S1 的 1% 附近。阶段必须是 S2，不能被 S1 盖掉。
    close = pd.Series([100.5, 100.5])
    e5 = pd.Series([120.0, 120.0])
    e13 = pd.Series([110.0, 110.0])
    e55 = pd.Series([100.0, 100.0])
    e125 = pd.Series([100.0, 100.0])
    e250 = pd.Series([90.0, 90.0])
    phase = compute_phase(close, e5, e13, e55, e125, e250)
    if int(phase[-1]) != 2:
        raise AssertionError(f"0.5% 上方应是 S2，实际 {phase[-1]}")

    # S4：昨天和今天排列都在；斜率只看今天相对 5 根前。昨天斜率不成立也要给 S4。
    base_close = pd.Series([10.0] * 7, dtype=float)
    e5 = pd.Series([9.0] * 7, dtype=float)
    e13 = pd.Series([8.0] * 7, dtype=float)
    e55 = pd.Series([7.0] * 7, dtype=float)
    e125 = pd.Series([6.0] * 7, dtype=float)
    e250 = pd.Series([5.0] * 7, dtype=float)
    flat = compute_phase(base_close, e5, e13, e55, e125, e250)
    if int(flat[-1]) == 4:
        raise AssertionError("均线没有上行，不应是 S4")
    e55 = e55.copy()
    e125 = e125.copy()
    e250 = e250.copy()
    e55.iloc[-1] = 7.2
    e125.iloc[-1] = 6.2
    e250.iloc[-1] = 5.2
    lifted = compute_phase(base_close, e5, e13, e55, e125, e250)
    if int(lifted[-1]) != 4:
        raise AssertionError(f"当天斜率成立应是 S4，实际 {lifted[-1]}")

    # 含当天的 5 日均量：16 不超过 11.2×1.5。若误用「不含当天」会放行。
    _ma, vol_ok = volume_flag(pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 16.0]))
    if bool(vol_ok.iloc[-1]):
        raise AssertionError("含当天均量时 16 不应算放量")

    kin = macd_kin(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]), pd.Series([0.2, -0.1]))
    if bool(kin.iloc[-1]):
        raise AssertionError("绿柱且 DIF 在 DEA 下，动能不应通过")
    red = macd_kin(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]), pd.Series([0.2, 0.1]))
    if not bool(red.iloc[-1]):
        raise AssertionError("连续两根红柱应通过")

    crossed = macd_cross_down(pd.Series([1.0, 0.0]), pd.Series([0.5, 0.2]))
    if not bool(crossed.iloc[-1]):
        raise AssertionError("DEA 上穿 DIF 没有识别出来")

    stopped = below_two(pd.Series([9.0, 9.0]), pd.Series([10.0, 10.0]))
    if not bool(stopped.iloc[-1]):
        raise AssertionError("两根收盘都低于当天 E55 应触发硬止损")
    # 两根各自都在自己的 E55 下方，但昨日收盘并没有低于「当天」抬不上去的 E55。公式要求跟当天 E55 比。
    not_yet = below_two(pd.Series([9.5, 8.5]), pd.Series([10.0, 9.0]))
    if bool(not_yet.iloc[-1]):
        raise AssertionError("昨日收盘没有低于当天 E55，不应止损")

    flags = {key: pd.Series([True, True]) for key in ENTRY_FLAGS}
    flags["bond"] = pd.Series([True, False])
    sample = pd.DataFrame(flags)
    blocked = combine_entry(sample, is_st=False)
    if bool(blocked.iloc[0]) is not True or bool(blocked.iloc[1]) is not False:
        raise AssertionError(f"缺一项就不应买：{blocked.tolist()}")
    if bool(combine_entry(sample, is_st=True).any()):
        raise AssertionError("ST 必须整段没有买点")
    # 可选过滤关闭后不再拦截；阶段、粘合、突破、历史长度和可交易性仍是核心条件。
    old_macd = USE_MACD
    globals()["USE_MACD"] = 0
    sample["macd_ok"] = False
    if not bool(combine_entry(sample, is_st=False).iloc[0]):
        raise AssertionError("关闭 MACD 过滤后不应被 macd_ok 拦截")
    globals()["USE_MACD"] = old_macd
    if not is_st_name("*ST测试") or not is_st_name("ST测试") or is_st_name("平安银行") or is_st_name(None):
        raise AssertionError("ST 名称判断错了")

    qty, _text = size_order(1_000_000.0, 1_000_000.0, 10.0)
    expected_qty = int(1_000_000.0 * SINGLE_CAP // 10.0 // LOT * LOT)
    if qty != expected_qty:
        raise AssertionError(f"{SINGLE_CAP:.0%} 仓位应为 {expected_qty} 股，实际 {qty}")
    for bad in (
        {"nope": 1}, {"rise_min": 0.5}, {"ema_fast": 20, "ema_mid": 10},
        {"bond_thresh": 0}, {"min_phase": 1}, {"use_macd": 2},
    ):
        try:
            apply_params(bad)
        except ValueError:
            _restore(saved)
            continue
        _restore(saved)
        raise AssertionError(f"非法参数没有被拦住：{bad}")
    _restore(saved)
    try:
        set_choices("nope", None)
    except ValueError:
        pass
    else:
        raise AssertionError("非法止盈方案没有被拦住")
    _restore(saved)

    # 删掉末尾 40 根再算，前面的阶段和买点必须一模一样。
    rng = np.random.default_rng(7)
    path = 20.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 380)))
    synth = pd.DataFrame({
        "date": pd.bdate_range("2018-01-01", periods=len(path)),
        "open": path * 0.99,
        "high": path * 1.02,
        "low": path * 0.98,
        "close": path,
        "volume": rng.integers(1000, 8000, len(path)).astype(float),
    })
    full = prepare(synth, is_st=False)
    short = prepare(synth.iloc[:-40].reset_index(drop=True), is_st=False)
    for column in ("phase", "entry"):
        left = full.iloc[:-40][column].reset_index(drop=True)
        right = short[column].reset_index(drop=True)
        if not left.equals(right):
            raise AssertionError(f"{column} 用到了未来数据")


def configure(cfg: dict | None) -> None:
    """灌入数据库路径和 bond_break 参数。命令行 --params / --exit / --high 在这之后再生效。"""
    global DB_PATH, OUTPUT_ROOT, DEFAULT_SYMBOL
    data = section(cfg, "data")
    output = section(cfg, "output")
    backtest = section(cfg, "backtest")
    if data.get("db_path"):
        DB_PATH = resolve_path(data["db_path"])
    if output.get("dir"):
        OUTPUT_ROOT = resolve_path(output["dir"])
    if backtest.get("default_symbol"):
        DEFAULT_SYMBOL = str(backtest["default_symbol"]).strip()
    raw = dict(section(cfg, "bond_break"))
    set_choices(raw.pop("exit_mode", None), raw.pop("high_field", None))
    apply_params(raw)


def main() -> None:
    global STREAM
    cfg = load_config()
    configure(cfg)
    backtest_cfg = section(cfg, "backtest")
    parser = argparse.ArgumentParser(description="五线粘合中继突破：本地数据单股回测")
    parser.add_argument("command", nargs="?", default="backtest", choices=("backtest", "selfcheck"))
    parser.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL)
    parser.add_argument("--params", help="JSON 数字参数，覆盖 config.yaml 的 bond_break 段")
    parser.add_argument("--exit", default="", choices=("", "macd", "rr"), help="止盈方案，默认 macd")
    parser.add_argument("--high", default="", choices=("", "close", "high"), help="高位口径，默认 close")
    parser.add_argument("--start", default=str(backtest_cfg.get("start_date") or ""), help="回测起始 YYYY-MM-DD")
    parser.add_argument("--end", default=str(backtest_cfg.get("end_date") or ""), help="回测结束 YYYY-MM-DD")
    parser.add_argument("--stream", action="store_true", help="日志逐行打印且不自动打开报告（网页端用）")
    args = parser.parse_args()
    if args.params:
        try:
            overrides = json.loads(args.params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--params 不是合法 JSON：{exc}") from exc
        apply_params(overrides)
    if args.exit or args.high:
        set_choices(args.exit or None, args.high or None)
    STREAM = bool(args.stream)
    if args.command == "selfcheck":
        self_check()
        return
    if args.symbol is None or not str(args.symbol).strip():
        raise ValueError("回测股票代码不能为空")
    try:
        run_backtest(args.symbol, start=args.start or None, end=args.end or None, show=not args.stream)
    except (ValueError, LookupError, FileNotFoundError, RuntimeError) as exc:
        print(f"错误：{exc}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
