"""五线探戈：单股回测 + 大盘选股。只读本地 data/a_share.db。

规则来源是 docs/SOURCE_SPEC.md 的首轮可执行摘要，口径按 docs/STRATEGY.md：

- 选股：日线阶段 S3/S4，且只用已经走完的周线（收盘 > 周趋势均线，并且周 DIF≥DEA 或柱连续两周变长）。
- 买入：进入 S3+ 或升到 S4 之后的 3–15 根里先缩量回撤，再放量突破前 10 根最高收盘。信号在收盘产生，下一交易日开盘成交。
- 直接放量突破只记在日志里，首轮不下单。做 T 不开。
- 卖出：收盘跌破「成交价 − 2×信号日 ATR」则下一开盘清仓；或连续两根收盘低于趋势均线至少 0.5% 则清仓。

用法（在项目根目录）。默认参数取自 config/config.yaml 的 five_line 段：

    python scripts/five_line.py                  # 回测 002432，报告在 output/002432/report.html
    python scripts/five_line.py backtest 600519 --start 2020-01-01 --params '{"rvol_high": 1.5}'
    python scripts/five_line.py pick             # 扫本地全市场最后一个交易日，汇总在 output/market/pick.txt
    python scripts/five_line.py selfcheck
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
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

# 首轮初始参数。文档写明这些数还没被验证过，只是回测起点。
TOLERANCE = 0.002          # 站上 / 跌破要离开均线至少 0.2%，避免贴线来回打脸
RVOL_HIGH = 1.30
RVOL_LOW = 0.80
PULLBACK_MIN = 0.03
PULLBACK_HARD_CAP = 0.12
WINDOW_MIN = 3
WINDOW_MAX = 15
BREAKOUT_LOOKBACK = 10
CHASE_MAX = 2.0            # (收盘-中线)/ATR 超过 2 视为追高
EMA_EXIT_GAP = 0.005       # 收盘低于趋势均线至少 0.5% 才算有效跌破
# 日线五条均线，从短到长。默认就是文档里的 5/13/55/125/250，网页和 config 可以改。
EMA_FAST = 5
EMA_MID = 13
EMA_TREND = 55
EMA_S3 = 125
EMA_LONG = 250
# 复盘图主图上的颜色，和周期无关，避免改了数字以后线对不上。
EMA_LINE_COLOR = {
    "ema_fast": "#f5c542",
    "ema_mid": "#e07a3d",
    "ema_trend": "#3d8bfd",
    "ema_s3": "#9b6bff",
    "ema_long": "#d6456a",
}
DAILY_WARMUP = 250         # 至少要盖住最长那条 EMA；文档建议 500，这里用最低可算长度
WEEKLY_WARMUP = 55
SINGLE_CAP = 0.15          # 单票计划仓位上限（占总权益）
BASE_WEIGHT = 0.50         # 计划仓位内部的基础仓
ADD_WEIGHT = 0.30          # 计划仓位内部的趋势加仓；剩下 20% 战术仓本轮不用
PORTFOLIO_CAP = 0.70
RISK_FRACTION = 0.008      # 单笔从买入价到初始止损的亏损不超过权益的 0.8%
MAX_NAMES = 5
SLIPPAGE = 0.001
INITIAL_CASH = 1_000_000.0
LOT = 100
# ponytail: 科创板最小 200 股且可按 1 股递增，这里全按 100 股整手。真做科创板再改。
OUTPUT_ROOT = PROJECT_ROOT / "output"
STREAM = False             # True 时日志每加一行就打印，给网页端逐行展示

# 允许被 config.yaml / 网页覆盖的参数。类型跟着默认值走：int 常量只收整数。
TUNABLE = (
    "TOLERANCE", "RVOL_HIGH", "RVOL_LOW", "PULLBACK_MIN", "PULLBACK_HARD_CAP",
    "WINDOW_MIN", "WINDOW_MAX", "BREAKOUT_LOOKBACK", "CHASE_MAX", "EMA_EXIT_GAP",
    "EMA_FAST", "EMA_MID", "EMA_TREND", "EMA_S3", "EMA_LONG",
    "DAILY_WARMUP", "WEEKLY_WARMUP", "SINGLE_CAP", "BASE_WEIGHT", "ADD_WEIGHT",
    "PORTFOLIO_CAP", "RISK_FRACTION", "SLIPPAGE", "INITIAL_CASH", "LOT",
)


def apply_params(params: dict | None) -> None:
    """用外部参数覆盖模块级常量。键不分大小写；未知键、非数字、负数直接报错，不默默吞掉。"""
    if not params:
        return
    if not isinstance(params, dict):
        raise ValueError("参数必须是字典")
    module = globals()
    for raw_key, raw_value in params.items():
        key = str(raw_key or "").strip().upper()
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
    # 窗口上下限反了会让所有回撤都判不合格，提前拦住。
    if module["WINDOW_MIN"] > module["WINDOW_MAX"]:
        raise ValueError(f"WINDOW_MIN {module['WINDOW_MIN']} 不能大于 WINDOW_MAX {module['WINDOW_MAX']}")
    spans = [module[key] for key in ("EMA_FAST", "EMA_MID", "EMA_TREND", "EMA_S3", "EMA_LONG")]
    if any(spans[i] >= spans[i + 1] for i in range(len(spans) - 1)):
        raise ValueError(f"日线 EMA 必须从短到长且互不相同，实际 {spans}")
    if module["INITIAL_CASH"] <= 0:
        raise ValueError("INITIAL_CASH 必须大于 0")


def ema_roles() -> tuple[tuple[str, int, str], ...]:
    """日线五条均线：(列名, 周期, 颜色)。顺序固定为快、中、趋势、四线、长线。"""
    return (
        ("ema_fast", int(EMA_FAST), EMA_LINE_COLOR["ema_fast"]),
        ("ema_mid", int(EMA_MID), EMA_LINE_COLOR["ema_mid"]),
        ("ema_trend", int(EMA_TREND), EMA_LINE_COLOR["ema_trend"]),
        ("ema_s3", int(EMA_S3), EMA_LINE_COLOR["ema_s3"]),
        ("ema_long", int(EMA_LONG), EMA_LINE_COLOR["ema_long"]),
    )


def current_params() -> dict:
    """当前生效的可调参数（小写键），网页用它做默认值。"""
    module = globals()
    return {key.lower(): module[key] for key in TUNABLE}


class Journal:
    """把回测故事按发生顺序记下来，最后整份写入 txt。"""

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


def bad_close_span(start: str, end: str, count: int) -> str:
    """连续多天收盘无效时合成一行，避免前复权打成负数的日子把日志刷满。"""
    if not start or count is None or count < 1:
        return ""
    if count == 1 or not end or end == start:
        return f"{start} 收盘价无效（前复权后 ≤0），当天不产生信号。"
    return f"{start} ~ {end} 共 {count} 天收盘价无效（前复权后 ≤0），这些天不产生信号。"


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


def trade_cost(side: str, notional: float, qty: int, when: date, symbol: str) -> float:
    """佣金两边都收；印花税只收卖出；过户费按成交日分段。滑点已经含在价格里，这里不再加。"""
    if side not in ("buy", "sell"):
        raise ValueError(f"未知方向 {side}")
    if not _finite(notional) or notional < 0 or qty is None or qty < 0 or when is None:
        raise ValueError("费用参数无效")
    if qty == 0 or notional == 0:
        return 0.0

    commission = max(5.0, notional * 0.0003)
    # 2023-08-28 起卖出印花税 0.05%，之前 0.1%。
    stamp_rate = 0.0005 if when >= date(2023, 8, 28) else 0.001
    stamp = notional * stamp_rate if side == "sell" else 0.0

    code = str(symbol or "")
    beijing = code.startswith(("8", "4")) or code.startswith("92")
    if when >= date(2022, 4, 29):
        transfer_rate = 0.00001
    elif when >= date(2015, 8, 1):
        transfer_rate = 0.000025 if beijing else 0.00002
    elif code.startswith("6"):
        # 2015-08-01 前沪市：股数 × 面值 1 元 × 0.03%。非 1 元面值这里覆盖不到。
        return commission + stamp + qty * 0.0003
    else:
        transfer_rate = 0.0000255
    return commission + stamp + notional * transfer_rate


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """ATR14：第一段用前 14 根 TR 的均值，之后每根按 (前值×13+当日TR)/14。"""
    if high is None or low is None or close is None:
        raise ValueError("ATR 输入不能为空")
    n = len(close)
    atr = np.full(n, np.nan)
    if n == 0:
        return atr
    tr = np.empty(n)
    tr[0] = float(high[0] - low[0]) if _finite(high[0]) and _finite(low[0]) else np.nan
    for i in range(1, n):
        if not (_finite(high[i]) and _finite(low[i]) and _finite(close[i - 1])):
            tr[i] = np.nan
            continue
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    if n < 14 or not np.isfinite(tr[:14]).all():
        return atr
    atr[13] = float(tr[:14].mean())
    for i in range(14, n):
        if not _finite(tr[i]) or not _finite(atr[i - 1]):
            atr[i] = np.nan
            continue
        atr[i] = (atr[i - 1] * 13.0 + tr[i]) / 14.0
    return atr


def _ema(close: pd.Series, span: int) -> pd.Series:
    """EMA 用第一根收盘做起点，adjust=False，和文档的递推一致。"""
    if close is None or span is None or span <= 0:
        raise ValueError("EMA 参数无效")
    return close.ewm(span=span, adjust=False).mean()


def _above(close: pd.Series, ema: pd.Series) -> pd.Series:
    """收盘至少高出均线 0.2% 才算站上。均线缺失或为 0 则不算。"""
    if close is None or ema is None:
        raise ValueError("站上判断缺少收盘或均线")
    return close.notna() & ema.notna() & (ema != 0) & (close > ema * (1.0 + TOLERANCE))


def _up_over_5(ema: pd.Series) -> pd.Series:
    """比 5 根之前高，才算这根均线在上行。"""
    if ema is None:
        raise ValueError("斜率判断缺少均线")
    previous = ema.shift(5)
    return ema.notna() & previous.notna() & (ema > previous)


def classify_phase(frame: pd.DataFrame) -> np.ndarray:
    """当天能满足的最高阶段。连续两根的确认只用当天和前一天，不把信号回填到第一根。

    S1 双线先行，价格还在趋势均线附近或下方，只观察。
    S2 三线多头，收盘站上趋势均线连续两根，进候选池，不买。
    S3 四线多头，收盘站上四线均线连续两根，允许第一笔基础仓。
    S4 五线多头且趋势/四线/长线都比 5 根前高，整组条件连续两根，允许加仓。
    """
    if frame is None or frame.empty:
        raise ValueError("没有日线，无法划分阶段")
    for column in ("close", "ema_fast", "ema_mid", "ema_trend", "ema_s3", "ema_long"):
        if column not in frame.columns:
            raise ValueError(f"划分阶段缺少 {column}")

    close = frame["close"]
    ema_fast = frame["ema_fast"]
    ema_mid = frame["ema_mid"]
    ema_trend = frame["ema_trend"]
    ema_s3 = frame["ema_s3"]
    ema_long = frame["ema_long"]

    three = (ema_fast > ema_mid) & (ema_mid > ema_trend)
    four = three & (ema_trend > ema_s3)
    five = four & (ema_s3 > ema_long) & _above(close, ema_fast)
    raw_s4 = five & _up_over_5(ema_trend) & _up_over_5(ema_s3) & _up_over_5(ema_long)
    raw_s4 = raw_s4.fillna(False).astype(bool)
    s4 = raw_s4 & raw_s4.shift(1, fill_value=False)

    above_s3 = _above(close, ema_s3).fillna(False).astype(bool)
    s3 = four.fillna(False) & above_s3 & above_s3.shift(1, fill_value=False) & ~s4

    above_trend = _above(close, ema_trend).fillna(False).astype(bool)
    s2 = three.fillna(False) & above_trend & above_trend.shift(1, fill_value=False) & ~s3 & ~s4

    # S1：快线在中线之上，收盘仍低于趋势线，或离趋势线不超过 1%。更高阶段优先。
    distance = (close - ema_trend).abs() / ema_trend.where(ema_trend != 0)
    near_trend = close.notna() & ema_trend.notna() & ((close < ema_trend) | (distance <= 0.01))
    s1 = (ema_fast > ema_mid).fillna(False) & near_trend.fillna(False) & ~s2 & ~s3 & ~s4

    phase = np.zeros(len(frame), dtype=int)
    phase[s1.to_numpy()] = 1
    phase[s2.to_numpy()] = 2
    phase[s3.to_numpy()] = 3
    phase[s4.to_numpy()] = 4
    # 最长均线没走完预热就标阶段，早期残值会被当成趋势。
    warmup = min(max(int(DAILY_WARMUP), int(EMA_LONG)), len(phase))
    phase[:warmup] = 0
    return phase


def attach_weekly(frame: pd.DataFrame) -> np.ndarray:
    """周线只在该自然周结束后的下一日才允许被日线看到。当周未完成的 K 不参与。

    许可条件：上一根完整周线收盘 > 周线趋势均线（周期与日线 EMA_TREND 相同），
    并且（DIF≥DEA 或柱连续两周变长）。周线没攒够预热时一律不允许开新仓。
    """
    if frame is None or frame.empty or "date" not in frame.columns:
        raise ValueError("周线对齐缺少日线日期")
    denied = np.zeros(len(frame), dtype=bool)
    indexed = frame.set_index("date").sort_index()
    weekly = indexed.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    weekly = weekly.dropna(subset=["close"])
    trend_span = int(EMA_TREND)
    weekly_ready = max(int(WEEKLY_WARMUP), trend_span)
    if weekly.empty or len(weekly) < weekly_ready:
        return denied

    wclose = weekly["close"]
    ema_trend = _ema(wclose, trend_span)
    dif = _ema(wclose, 12) - _ema(wclose, 26)
    dea = _ema(dif, 9)
    hist = dif - dea
    # ewm 从第一根就会出数。周线没攒够之前这些数不能当过滤器。
    ema_trend = ema_trend.copy()
    dif = dif.copy()
    dea = dea.copy()
    hist = hist.copy()
    ema_trend.iloc[: weekly_ready - 1] = np.nan
    dif.iloc[:34] = np.nan
    dea.iloc[:34] = np.nan
    hist.iloc[:34] = np.nan

    above = wclose.notna() & ema_trend.notna() & (wclose > ema_trend)
    macd_ok = dif.notna() & dea.notna() & (dif >= dea)
    improving = (
        hist.notna()
        & hist.shift(1).notna()
        & hist.shift(2).notna()
        & (hist > hist.shift(1))
        & (hist.shift(1) > hist.shift(2))
    )
    weekly = weekly.copy()
    weekly["weekly_ok"] = (above & (macd_ok | improving)).fillna(False)
    # 周五这根周 K 要到周六才算完成，所以周五当天的日线仍只能看到上一周。
    weekly["available_on"] = weekly.index + pd.Timedelta(days=1)
    right = weekly[["available_on", "weekly_ok"]].reset_index(drop=True)
    left = frame[["date"]].copy()
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("available_on"),
        left_on="date",
        right_on="available_on",
        direction="backward",
    )
    if merged is None or "weekly_ok" not in merged.columns:
        return denied
    # merge_asof 按日期排过序。调用方传入的 frame 已经按日期升序，可以对齐回去。
    return merged["weekly_ok"].fillna(False).to_numpy(dtype=bool)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """补上均线、MACD、ATR、20 日量比、阶段和周线许可。量比分母不含当天。"""
    if raw is None or raw.empty:
        raise ValueError("日线为空")
    frame = raw.sort_values("date").reset_index(drop=True).copy()
    for column in ("date", "open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            raise ValueError(f"日线缺少 {column}")
    frame["volume"] = frame["volume"].fillna(0.0)
    close = frame["close"]
    for column, span, _color in ema_roles():
        frame[column] = _ema(close, span)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    frame["dif"] = ema12 - ema26
    frame["dea"] = _ema(frame["dif"], 9)
    frame["hist"] = frame["dif"] - frame["dea"]
    frame["atr"] = wilder_atr(
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        close.to_numpy(dtype=float),
    )
    prior_mean = frame["volume"].shift(1).rolling(20, min_periods=20).mean()
    frame["rvol"] = frame["volume"] / prior_mean.where(prior_mean > 0)
    frame["phase"] = classify_phase(frame)
    frame["weekly_ok"] = attach_weekly(frame)
    return frame


def window_verdict(
    closes: np.ndarray,
    atrs: np.ndarray,
    ema55s: np.ndarray,
    rvols: np.ndarray,
) -> tuple[bool, str]:
    """回撤区间是否合格。最高价用「走到这里为止」的运行高点，不能拿后面的新高倒算前面的低点。"""
    if closes is None or atrs is None or ema55s is None or rvols is None:
        return False, "回撤数据缺失"
    length = len(closes)
    if length < WINDOW_MIN:
        return False, f"回撤才 {length} 根，不到 {WINDOW_MIN} 根"
    if length > WINDOW_MAX:
        return False, f"回撤已有 {length} 根，超过 {WINDOW_MAX} 根"
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        return False, "回撤区间收盘无效"

    running_high = np.maximum.accumulate(closes)
    drawdown = (running_high - closes) / running_high
    deepest = int(np.nanargmax(drawdown))
    max_dd = float(drawdown[deepest])
    peak = float(running_high[deepest])
    atr = float(atrs[deepest]) if deepest < len(atrs) else np.nan
    if not _finite(atr) or atr < 0 or peak <= 0:
        return False, "回撤最深处的 ATR 无效，无法判断回撤是不是过深"
    cap = max(2.0 * atr / peak, PULLBACK_HARD_CAP)
    shrink = int(np.sum(np.isfinite(rvols) & (rvols <= RVOL_LOW)))
    last_close = float(closes[-1])
    last_ema = float(ema55s[-1]) if len(ema55s) else np.nan

    reasons = []
    if shrink < 2:
        reasons.append(f"缩量只有 {shrink} 根（RVOL≤{RVOL_LOW:.2f}），需要至少 2 根")
    if max_dd < PULLBACK_MIN:
        reasons.append(f"最大回撤 {_pct(max_dd)}，不到 {_pct(PULLBACK_MIN)}")
    if max_dd > cap:
        reasons.append(
            f"最大回撤 {_pct(max_dd)}，超过上限 {_pct(cap)}（2×ATR/峰值 与 {_pct(PULLBACK_HARD_CAP)} 取较大）"
        )
    if not _finite(last_ema) or last_close < last_ema:
        reasons.append(
            f"突破前一日收盘 {_num(last_close)} 低于 EMA{EMA_TREND} {_num(last_ema)}，回撤把趋势跌坏了"
        )
    if reasons:
        return False, "；".join(reasons)
    return True, (
        f"回撤合格：幅度 {_pct(max_dd)}，上限 {_pct(cap)}，缩量 {shrink} 根，"
        f"突破前一日收盘 {_num(last_close)} 仍在 EMA{EMA_TREND} {_num(last_ema)} 之上"
    )


def _breakout(closes: np.ndarray, rvols: np.ndarray, index: int) -> tuple[bool, str]:
    """当天收盘高于前 10 根最高收盘（不含今天），并且 20 日量比 ≥ 1.30。"""
    if closes is None or rvols is None or index is None:
        return False, "突破数据缺失"
    if index < BREAKOUT_LOOKBACK:
        return False, f"前面不足 {BREAKOUT_LOOKBACK} 根，不能算突破"
    prior = closes[index - BREAKOUT_LOOKBACK : index]
    if not np.isfinite(prior).all():
        return False, "前 10 根收盘有缺失"
    prior_high = float(np.max(prior))
    today = float(closes[index]) if _finite(closes[index]) else np.nan
    rvol = float(rvols[index]) if index < len(rvols) and _finite(rvols[index]) else np.nan
    price_ok = _finite(today) and today > prior_high
    volume_ok = _finite(rvol) and rvol >= RVOL_HIGH
    if price_ok and volume_ok:
        return True, f"收盘 {_num(today)} 突破前10日高 {_num(prior_high)}，RVOL {_num(rvol)}"
    bits = []
    if not price_ok:
        bits.append(f"收盘 {_num(today)} 没有高于前10日高 {_num(prior_high)}")
    if not volume_ok:
        bits.append(f"RVOL {_num(rvol)} 没有达到 {RVOL_HIGH:.2f}")
    return False, "；".join(bits)


def replay(frame: pd.DataFrame) -> dict:
    """沿日线往前走一遍回撤窗口。锚点只在「第一次进入 S3+」或「从更低阶段升到 S4」时重置。

    跌回 S2 及以下，窗口作废。超过 15 根还没走出合格突破，窗口过期，要等下一次阶段事件。
    """
    if frame is None or frame.empty:
        raise ValueError("无法回放空行情")
    n = len(frame)
    phase = frame["phase"].to_numpy(dtype=int)
    closes = frame["close"].to_numpy(dtype=float)
    atrs = frame["atr"].to_numpy(dtype=float)
    ema_trend = frame["ema_trend"].to_numpy(dtype=float)
    rvols = frame["rvol"].to_numpy(dtype=float)
    dates = frame["date"]

    pullback = np.zeros(n, dtype=bool)
    direct = np.zeros(n, dtype=bool)
    events: list[tuple[int, str]] = []
    anchor = None
    previous = 0

    for i in range(n):
        today = int(phase[i])
        day = _day_text(dates.iloc[i])
        invalidated = False
        if today <= 2:
            if anchor is not None:
                invalidated = True
                events.append((
                    i,
                    f"{day} 阶段 {_phase_name(previous)} → {_phase_name(today)}，回撤窗口作废。",
                ))
            anchor = None
        elif previous <= 2 or (today == 4 and previous < 4):
            anchor = i
            why = "第一次进入 S3 及以上" if previous <= 2 else "从更低阶段升到 S4"
            events.append((i, f"{day} {_phase_name(today)}，{why}，回撤锚点改到今天。之后 3–15 根里等缩量回撤再突破。"))
        elif anchor is not None and (i - anchor) > WINDOW_MAX:
            events.append((i, f"{day} 锚点后已经超过 {WINDOW_MAX} 根仍没有合格的回踩突破，这个窗口过期。"))
            anchor = None

        # 直接突破：昨天已经是 S3/S4，今天放量突破前 10 日高。首轮只记录，不作为开仓理由。
        broke, broke_text = _breakout(closes, rvols, i)
        if previous >= 3 and broke and today >= 3:
            direct[i] = True

        logged_breakout = False
        if anchor is not None and anchor < i and broke:
            ok, detail = window_verdict(closes[anchor:i], atrs[anchor:i], ema_trend[anchor:i], rvols[anchor:i])
            logged_breakout = True
            if ok:
                pullback[i] = True
                events.append((i, f"{day} 回踩后再突破成立。{detail}。{broke_text}。"))
            else:
                events.append((i, f"{day} 价格放量突破了，但这轮回撤不合格，不下单。{detail}。{broke_text}。"))
        elif previous >= 3 and broke and today >= 3 and not logged_breakout:
            if anchor == i:
                events.append((i, f"{day} 锚点当天就放量突破（{broke_text}）。回撤还没满 3 根，这次不买。"))
            else:
                events.append((i, f"{day} 直接放量突破（{broke_text}）。没有落在合格回撤窗口里，首轮不新开仓。"))

        entered_here = anchor == i and today >= 3 and (previous <= 2 or (today == 4 and previous < 4))
        # S0/S1 来回切太碎。锚点重置、窗口作废这两句里已经写了阶段，这里不重复。
        if today != previous and not entered_here and not invalidated and (today >= 2 or previous >= 2):
            events.append((i, f"{day} 阶段 {_phase_name(previous)} → {_phase_name(today)}。"))
        previous = today

    return {"pullback": pullback, "direct": direct, "events": events}


def _macd_supports(dif, dea, hist, prev_hist) -> tuple[bool, str]:
    """DIF>DEA 且柱不缩短，或者柱连续两根为正。两个条件满足一个即可。"""
    if not (_finite(dif) and _finite(dea) and _finite(hist)):
        return False, "MACD 还没有有效数值"
    rising = _finite(prev_hist) and hist >= prev_hist and dif > dea
    two_positive = _finite(prev_hist) and hist > 0 and prev_hist > 0
    if rising:
        return True, f"DIF {_num(dif, 3)} > DEA {_num(dea, 3)}，柱 {_num(hist, 3)} 没有缩短"
    if two_positive:
        return True, f"柱连续两根为正（今日 {_num(hist, 3)}，昨日 {_num(prev_hist, 3)}）"
    return False, (
        f"DIF {_num(dif, 3)} / DEA {_num(dea, 3)} / 柱 {_num(hist, 3)}，"
        "既没有「DIF 在 DEA 上且柱不缩短」，也没有连续两根正柱"
    )


def _chop_count(closes: np.ndarray, ema55: np.ndarray, index: int) -> int:
    """最近 20 根里，收盘相对 EMA55 的上下方向变了几次。"""
    if closes is None or ema55 is None or index < 20:
        return 0
    start = index - 19
    signs = []
    for j in range(start, index + 1):
        if not (_finite(closes[j]) and _finite(ema55[j]) and ema55[j] != 0):
            return 0
        diff = closes[j] - ema55[j]
        if diff == 0:
            signs.append(0)
        else:
            signs.append(1 if diff > 0 else -1)
    changes = 0
    for left, right in zip(signs, signs[1:]):
        if left != 0 and right != 0 and left != right:
            changes += 1
    return changes


def _divergence(closes: np.ndarray, dif: np.ndarray, index: int) -> bool:
    """收盘创前 20 根新高，但 DIF 没有创前 20 根新高。只拦住加仓，不拦第一笔。"""
    if closes is None or dif is None or index < 21:
        return False
    prior_close = closes[index - 20 : index]
    prior_dif = dif[index - 20 : index]
    if not np.isfinite(prior_close).all() or not np.isfinite(prior_dif).all():
        return False
    if not (_finite(closes[index]) and _finite(dif[index])):
        return False
    return bool(closes[index] > np.max(prior_close) and dif[index] <= np.max(prior_dif))


def _chase(close, ema13, atr) -> tuple[bool, str]:
    """返回 (是否追高, 说明)。ATR 无效时宁可放弃，也不用一个假距离去买。"""
    if not (_finite(close) and _finite(ema13) and _finite(atr) and atr > 0):
        return True, f"ATR 或 EMA{EMA_MID} 无效，追高过滤器无法计算，放弃"
    stretch = (close - ema13) / atr
    if stretch > CHASE_MAX:
        return True, f"(收盘-EMA{EMA_MID})/ATR = {_num(stretch)}，超过 {CHASE_MAX:.1f}，不追高"
    return False, f"(收盘-EMA{EMA_MID})/ATR = {_num(stretch)}，没有超过 {CHASE_MAX:.1f}"


def _size_order(
    equity: float,
    cash: float,
    price: float,
    atr: float,
    held_value: float,
    weight: float,
    half: bool,
) -> tuple[int, str]:
    """股数 = min(这一档金额, 0.8% 权益 / 止损距离, 单票 15% 剩余, 组合 70% 剩余, 现金)，再向下取整到 100 股。"""
    if not (_finite(equity) and _finite(cash) and _finite(price) and _finite(atr) and _finite(held_value)):
        return 0, "权益、现金、价格或 ATR 无效，算不出股数"
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0, "权益、价格或 ATR 不是正数，无法下单"
    slot = equity * SINGLE_CAP * weight
    if half:
        slot *= 0.5
    room_single = equity * SINGLE_CAP - held_value
    room_book = equity * PORTFOLIO_CAP - held_value
    room_cash = cash
    room = min(slot, room_single, room_book, room_cash)
    stop_distance = 2.0 * atr
    risk_shares = (equity * RISK_FRACTION) / stop_distance
    slot_shares = room / price if price > 0 else 0.0
    raw = min(risk_shares, slot_shares)
    qty = int(raw // LOT * LOT)
    text = (
        f"权益 {_num(equity, 0)}，这一档预算 {_num(slot, 0)}（计划仓位 {SINGLE_CAP:.0%} × {weight:.0%}"
        f"{'，震荡减半' if half else ''}），"
        f"单票剩余额度 {_num(room_single, 0)}，组合剩余额度 {_num(room_book, 0)}，现金 {_num(room_cash, 0)}。"
        f"风险预算可买 {_num(risk_shares, 1)} 股（{RISK_FRACTION:.1%} 权益 / 两倍 ATR {_num(stop_distance)}），"
        f"金额可买 {_num(slot_shares, 1)} 股，向下取整 {qty} 股。"
    )
    if qty < LOT:
        return 0, text + f"不足 1 手（{LOT} 股），这次不下单。"
    return qty, text


ENGINE_COLUMNS = [
    "phase",
    "weekly_ok",
    "pullback",
    "direct",
    "atr",
    "ema_fast",
    "ema_mid",
    "ema_trend",
    "ema_s3",
    "ema_long",
    "dif",
    "dea",
    "hist",
    "chop",
    "divergence",
]


def _bar_day(timestamp) -> str:
    """把引擎时间戳收成上海时区的交易日，用来和预计算事件对上。"""
    if timestamp is None:
        return "未知日期"
    try:
        stamp = pd.Timestamp(timestamp)
    except (TypeError, ValueError):
        return "未知日期"
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp.strftime("%Y-%m-%d")


def _call_number(strategy, name: str) -> float | None:
    """equity / cash 在不同版本里可能是方法也可能是属性，取不到就当没有。"""
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
    """持仓股数。没有仓位、字段缺失都当成 0，避免拿空对象去下单。"""
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


def to_engine_frame(symbol: str, frame: pd.DataFrame, signal: dict) -> pd.DataFrame:
    """收成 AKQuant 要的表：OHLCV 加 float 因子。信号仍是收盘算好的，撮合由引擎在下一根开盘做。"""
    if frame is None or frame.empty:
        raise ValueError("没有日线，无法交给回测引擎")
    if signal is None or "pullback" not in signal or "direct" not in signal:
        raise ValueError("缺少回撤/突破信号")
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    out = frame.copy()
    closes = out["close"].to_numpy(dtype=float)
    ema_trend = out["ema_trend"].to_numpy(dtype=float)
    dif = out["dif"].to_numpy(dtype=float)
    chop = np.zeros(len(out))
    divergence = np.zeros(len(out))
    for index in range(len(out)):
        chop[index] = _chop_count(closes, ema_trend, index)
        divergence[index] = 1.0 if _divergence(closes, dif, index) else 0.0
    out["chop"] = chop
    out["divergence"] = divergence
    out["pullback"] = np.asarray(signal["pullback"], dtype=float)
    out["direct"] = np.asarray(signal["direct"], dtype=float)
    out["weekly_ok"] = out["weekly_ok"].astype(float)
    out["phase"] = out["phase"].astype(float)
    # 日线落在收盘 15:00，和 stock_picking 一样，避免按午夜对齐。
    out["date"] = pd.to_datetime(out["date"]).dt.normalize() + pd.Timedelta(hours=15)
    out["symbol"] = str(symbol).zfill(6)
    missing = [column for column in ["date", "open", "high", "low", "close", "volume", *ENGINE_COLUMNS] if column not in out.columns]
    if missing:
        raise ValueError(f"回测数据缺少列：{missing}")
    view = out[["date", "open", "high", "low", "close", "volume", "symbol", *ENGINE_COLUMNS]].copy()
    view[ENGINE_COLUMNS] = view[ENGINE_COLUMNS].astype(float).fillna(0.0)
    return view


class FiveLineStrategy(Strategy):
    """收盘读预计算信号，用 buy / close_position 把单交给 AKQuant。成交价、费用、报告都由引擎出。"""

    def __init__(self):
        # 不在构造函数里收日志和代码。AKQuant 0.3 不把 __init__ 参数当成策略参数，
        # 这些字段在 run_backtest 之前直接赋到实例上。
        super().__init__()
        self.journal = Journal()
        self.events: dict = {}
        self.code = ""
        self.name = ""
        self.added = False
        self.add_blocked = False
        self.fresh_phase = True
        self.cooldown_left = 0
        self.batch_stops: list[float] = []
        self.prev_phase = 0
        self.prev_hist = None
        self.prev_below55 = False
        self.below13_streak = 0
        self.tactical_noted = False
        self.signal_atr = None
        self.pending_slot = ""
        self.exit_reason = None
        self.buys = 0
        self.sells = 0
        self._bar = None
        self.quotes: dict[str, tuple[float, float]] = {}
        self.bad_start = None
        self.bad_end = None
        self.bad_count = 0
        self.blocked = {
            "周线不许": 0,
            "追高或ATR无效": 0,
            "日线MACD不支持": 0,
            "退出后还没重新站上S3/S4": 0,
            "冷却期": 0,
            "一手都买不起": 0,
        }
        self.days = {"s3": 0, "s4": 0, "weekly": 0, "pullback": 0, "direct": 0}

    def on_trade(self, trade):
        """引擎真实成交。买入时用成交价锁定这批的初始止损，不随以后的 ATR 放宽。"""
        if trade is None:
            return
        side = str(getattr(trade, "side", "") or "").lower()
        price = getattr(trade, "price", None)
        qty = getattr(trade, "quantity", None)
        day = _bar_day(getattr(trade, "timestamp", None))
        if "buy" in side:
            self.buys += 1
            if self.pending_slot == "趋势加仓":
                self.added = True
            stop = None
            if _finite(price) and _finite(self.signal_atr) and self.signal_atr > 0:
                stop = float(price) - 2.0 * float(self.signal_atr)
                self.batch_stops.append(stop)
            self.journal.add(
                f"{day} {self._quote(day)}成交买入 {_num(qty, 0)} 股，开盘成交价 {_num(price)}。"
                f"这批止损 {_num(stop)} = 成交价 − 2×信号日 ATR {_num(self.signal_atr)}。"
                f"槽位：{self.pending_slot or '未知'}。"
            )
            return
        if "sell" in side:
            self.sells += 1
            self.journal.add(
                f"{day} {self._quote(day)}成交卖出 {_num(qty, 0)} 股，开盘成交价 {_num(price)}。{self.exit_reason or ''}"
            )
            self.batch_stops = []
            self.added = False
            self.add_blocked = False
            self.fresh_phase = False
            self.cooldown_left = 3
            self.prev_below55 = False
            self.below13_streak = 0
            self.tactical_noted = False
            self.exit_reason = None
            self.journal.add("    卖完后至少再等 3 根日线，并且要重新进入 S3/S4，才允许再买。")

    def on_reject(self, order):
        """引擎拒单时写进日志，避免报告里没成交、文本里却像已经买到。"""
        if order is None:
            self.journal.add("有一张单被引擎拒绝，但没有拒单详情。")
            return
        reason = getattr(order, "reason", None) or getattr(order, "message", None) or order
        self.journal.add(f"引擎拒单：{reason}")

    def _quote(self, day: str) -> str:
        """当天开盘、收盘。没有这天的行情时返回空串，调用方不用再判断。"""
        pair = self.quotes.get(day) if day and self.quotes else None
        if not pair or len(pair) < 2:
            return ""
        return f"开 {_num(pair[0])} 收 {_num(pair[1])} "

    def _log(self, text: str) -> None:
        """以日期开头的主行补上当天开盘、收盘；缩进的说明行不动。"""
        if text is None:
            return
        bar = self._bar
        day = _bar_day(getattr(bar, "timestamp", None)) if bar is not None else ""
        if bar is not None and day and text.startswith(day + " "):
            text = f"{day} 开 {_num(getattr(bar, 'open', None))} 收 {_num(bar.close)}{text[len(day):]}"
        self.journal.add(text)

    def _note_bad_close(self, day: str) -> None:
        """无效收盘先攒着，等下一段正常交易日再一次性写出来。"""
        if not day:
            return
        if self.bad_start is None:
            self.bad_start = day
        self.bad_end = day
        self.bad_count += 1

    def _flush_bad_close(self) -> None:
        if not self.bad_count:
            return
        text = bad_close_span(self.bad_start, self.bad_end, self.bad_count)
        self.bad_start = None
        self.bad_end = None
        self.bad_count = 0
        if text:
            self.journal.add(text)

    def _plot_emas(self, extra) -> None:
        """把这次用到的五条日线 EMA 报到主图。review 只画上报过的指标，行情列本身不会出现在图上。"""
        if extra is None:
            return
        for column, span, color in ema_roles():
            value = extra.get(column)
            if not _finite(value):
                continue
            self.record_indicator(
                name=column,
                value=float(value),
                display_name=f"EMA{int(span)}",
                pane=0,
                render_type="line",
                color=color,
                precision=2,
            )

    def on_bar(self, bar):
        extra = bar.extra or {}
        symbol = bar.symbol
        if symbol is None:
            return
        close = bar.close
        day = _bar_day(bar.timestamp)
        self._bar = bar
        # 这一天已经有阶段或突破要写，先把前面攒下的无效日结清，避免说明插在区间中间。
        if self.events.get(day):
            self._flush_bad_close()
            for text in self.events.get(day) or []:
                self._log(text)
        if close is None or close <= 0:
            # 这一天如果已经写了阶段/突破，开盘收盘就在那一行上，不再单独报「无效」。
            if not self.events.get(day):
                self._note_bad_close(day)
            return
        self._flush_bad_close()

        phase = int(extra.get("phase") or 0)
        weekly_ok = float(extra.get("weekly_ok") or 0.0) >= 1.0
        pullback = float(extra.get("pullback") or 0.0) >= 1.0
        direct = float(extra.get("direct") or 0.0) >= 1.0
        atr = extra.get("atr")
        ema_mid = extra.get("ema_mid")
        ema_trend = extra.get("ema_trend")
        self._plot_emas(extra)
        if phase == 3:
            self.days["s3"] += 1
        elif phase == 4:
            self.days["s4"] += 1
        if weekly_ok:
            self.days["weekly"] += 1
        if pullback:
            self.days["pullback"] += 1
        if direct:
            self.days["direct"] += 1

        qty = _position_qty(self, symbol)
        open_orders = self.get_open_orders(symbol=symbol) or []
        # 已经挂着单就不再报。卖单会在下一根开盘成交，当天不再找新买点。
        if not open_orders and self.exit_reason and qty > 0:
            available = self.get_available_position(symbol)
            if available is None or available <= 0:
                self._log(f"{day} 还不能卖（T+1 或可卖数量为 0），清仓意图保留。")
            else:
                self.close_position(symbol=symbol)
                self._log(f"{day} 继续提交清仓。{self.exit_reason}")

        if qty > 0 and not open_orders and self.exit_reason is None:
            stop = max(self.batch_stops) if self.batch_stops else None
            below_trend = _finite(ema_trend) and close < float(ema_trend) * (1.0 - EMA_EXIT_GAP)
            if stop is not None and close <= stop:
                self.exit_reason = f"收盘 {_num(close)} 跌到初始保护 {_num(stop)}"
                self.close_position(symbol=symbol)
                self._log(f"{day} 收盘触发紧急止损，下一根开盘清仓。{self.exit_reason}。")
            elif below_trend and self.prev_below55:
                self.exit_reason = (
                    f"连续两根收盘低于 EMA{EMA_TREND} 至少 {EMA_EXIT_GAP:.1%}："
                    f"今日收盘 {_num(close)}，EMA{EMA_TREND} {_num(ema_trend)}"
                )
                self.close_position(symbol=symbol)
                self._log(f"{day} 收盘确认 EMA{EMA_TREND} 趋势破坏，下一根开盘清仓。")
            elif below_trend and not self.prev_below55:
                self.add_blocked = True
                self._log(
                    f"{day} 收盘第一次有效跌破 EMA{EMA_TREND}"
                    f"（{_num(close)} < {_num(ema_trend)}×{1 - EMA_EXIT_GAP:.3f}）。"
                    "先停止加仓；还要再一根同样的跌破才清仓。"
                )
            self.prev_below55 = bool(below_trend)
            under_mid = _finite(ema_mid) and close < float(ema_mid) * (1.0 - TOLERANCE)
            self.below13_streak = self.below13_streak + 1 if under_mid else 0
            entry = None
            if self.ctx is not None:
                entry = self.ctx.get_position_entry_price(symbol)
            profitable = _finite(entry) and entry > 0 and close > float(entry)
            if self.below13_streak >= 2 and profitable and not self.tactical_noted:
                self.tactical_noted = True
                self._log(
                    f"{day} 持仓还在盈利，但收盘已连续两根跌破 EMA{EMA_MID}。"
                    f"按规则这里只平战术仓；本轮没开战术仓，核心仓继续拿到 EMA{EMA_TREND} 退出。"
                )
            if not under_mid:
                self.tactical_noted = False

        want_base = qty <= 0 and not open_orders and self.exit_reason is None and pullback and phase >= 3
        in_profit = False
        if qty > 0 and self.ctx is not None:
            entry = self.ctx.get_position_entry_price(symbol)
            in_profit = _finite(entry) and entry > 0 and close > float(entry)
        want_add = (
            qty > 0
            and not open_orders
            and self.exit_reason is None
            and not self.added
            and not self.add_blocked
            and phase == 4
            and _finite(ema_trend)
            and close >= float(ema_trend)
            and (pullback or (direct and in_profit))
        )
        if want_base or want_add:
            reasons = []
            if self.cooldown_left > 0:
                reasons.append(f"还在退出后的冷却里，还要再等 {self.cooldown_left} 根日线")
                self.blocked["冷却期"] += 1
            if want_base and not self.fresh_phase:
                reasons.append("卖出之后还没有重新进入 S3/S4，不能用旧趋势再买")
                self.blocked["退出后还没重新站上S3/S4"] += 1
            if not weekly_ok:
                reasons.append(f"最近一根已经完成的周线没有站上 EMA{EMA_TREND}，或周线 MACD 在变差")
                self.blocked["周线不许"] += 1
            chased, chase_text = _chase(close, ema_mid, atr)
            if chased:
                reasons.append(chase_text)
                self.blocked["追高或ATR无效"] += 1
            else:
                chase_text = "追高过滤通过：" + chase_text
            macd_ok, macd_text = _macd_supports(extra.get("dif"), extra.get("dea"), extra.get("hist"), self.prev_hist)
            if not macd_ok:
                reasons.append(macd_text)
                self.blocked["日线MACD不支持"] += 1
            chop = int(extra.get("chop") or 0)
            half = chop >= 4
            diverged = float(extra.get("divergence") or 0.0) >= 1.0
            if want_add and diverged:
                reasons.append("价格创前 20 根新高但 DIF 没有，背离期间停止加仓")
            slot = "基础仓" if want_base else "趋势加仓"
            weight = BASE_WEIGHT if want_base else ADD_WEIGHT
            self._log(f"{day} 收盘检查{slot}。阶段 {_phase_name(phase)}。")
            self.journal.add(f"    周线许可：{'是' if weekly_ok else '否'}。")
            self.journal.add(f"    {chase_text}。")
            self.journal.add(f"    日线 MACD：{macd_text}。")
            self.journal.add(
                f"    最近 20 根相对 EMA{EMA_TREND} 换边 {chop} 次"
                f"{'，新仓金额减半' if half else '，不减半'}。"
            )
            if want_add:
                self.journal.add(f"    背离：{'有，不加仓' if diverged else '没有'}。")
                how = "新的回踩突破" if pullback else "持仓盈利后的新放量突破"
                self.journal.add(f"    加仓形态：{how}。")
            if reasons or not _finite(atr) or float(atr) <= 0:
                if not _finite(atr) or float(atr) <= 0:
                    reasons.append("ATR 无效，无法设止损")
                    self.blocked["追高或ATR无效"] += 1
                self.journal.add("    结论：今天不报单。原因：" + "；".join(reasons) + "。")
            else:
                equity = _call_number(self, "equity") or INITIAL_CASH
                cash = _call_number(self, "cash")
                if cash is None:
                    cash = equity
                held = qty * float(close)
                order_qty, sizing = _size_order(equity, cash, float(close), float(atr), held, weight, half)
                self.journal.add(f"    {sizing}")
                if order_qty < LOT:
                    self.blocked["一手都买不起"] += 1
                else:
                    self.signal_atr = float(atr)
                    self.pending_slot = slot
                    receipt = self.buy(symbol=symbol, quantity=order_qty)
                    if receipt is None:
                        self.journal.add("    结论：引擎没有收下这张买单。")
                    else:
                        self.journal.add(f"    结论：已提交{slot}买单 {order_qty} 股，按引擎规则在下一根开盘撮合。")

        if qty <= 0 and phase >= 3 and phase > self.prev_phase:
            self.fresh_phase = True
        self.prev_phase = phase
        hist = extra.get("hist")
        self.prev_hist = float(hist) if _finite(hist) else None
        if qty <= 0 and self.cooldown_left > 0:
            self.cooldown_left -= 1

    def on_stop(self):
        """回测结束时把成交次数和没买到的原因补进同一份日志。"""
        self._flush_bad_close()
        equity = _call_number(self, "equity")
        self.journal.blank()
        self.journal.add("========== 这笔回测为什么这样结束 ==========")
        self.journal.add(
            f"初始资金 {_num(INITIAL_CASH, 0)}，期末权益 {_num(equity, 0)}。"
            "权益、回撤和买卖点以 AKQuant 报告为准。"
        )
        self.journal.add(f"买入成交 {self.buys} 次，卖出成交 {self.sells} 次。")
        if self.buys == 0:
            self.journal.add("一次都没买到。下面是买点被挡住的次数（同一天可以同时挡住好几项）：")
            for key, count in self.blocked.items():
                self.journal.add(f"    {key}：{count}")
            self.journal.add(
                f"阶段天数：S3 {self.days['s3']} 天，S4 {self.days['s4']} 天，"
                f"周线许可 {self.days['weekly']} 天，回踩突破信号 {self.days['pullback']} 天，"
                f"直接突破 {self.days['direct']} 天。"
            )
        else:
            for key, count in self.blocked.items():
                if count:
                    self.journal.add(f"中途放过或没做成：{key} {count} 次。")


def pick_symbol(frame: pd.DataFrame) -> dict | None:
    """站在最后一根已完成日线上，假设现在空仓，说明今天该不该进观察池、能不能买。"""
    if frame is None or frame.empty:
        return None
    last = len(frame) - 1
    need = max(int(DAILY_WARMUP), int(EMA_LONG))
    if last < need:
        return {"bucket": "历史不够", "text": f"只有 {len(frame)} 根日线，EMA{EMA_LONG} 需要 {need} 根"}
    phase = int(frame["phase"].iloc[last])
    if phase < 3:
        return {"bucket": "不是候选", "text": f"最后一天是 {_phase_name(phase)}，五线策略不开新仓"}

    signal = replay(frame)
    weekly_ok = bool(frame["weekly_ok"].iloc[last])
    close = float(frame["close"].iloc[last])
    notes = [text for index, text in signal["events"] if index == last]
    story = notes[-1] if notes else "今天没有新的突破事件。"
    if not weekly_ok:
        return {
            "bucket": "周线拦住",
            "phase": phase,
            "close": close,
            "text": f"{_phase_name(phase)}，但最后一根可用周线未许可。{story}",
        }
    if bool(signal["pullback"][last]):
        chased, chase_text = _chase(close, frame["ema_mid"].iloc[last], frame["atr"].iloc[last])
        prev_hist = float(frame["hist"].iloc[last - 1]) if last else None
        macd_ok, macd_text = _macd_supports(
            frame["dif"].iloc[last], frame["dea"].iloc[last], frame["hist"].iloc[last], prev_hist
        )
        if chased or not macd_ok:
            why = []
            if chased:
                why.append(chase_text)
            if not macd_ok:
                why.append(macd_text)
            return {
                "bucket": "观察池",
                "phase": phase,
                "close": close,
                "text": "回踩突破有了，但还不能买：" + "；".join(why),
            }
        half = _chop_count(
            frame["close"].to_numpy(dtype=float),
            frame["ema_trend"].to_numpy(dtype=float),
            last,
        ) >= 4
        qty, sizing = _size_order(
            INITIAL_CASH, INITIAL_CASH, close, float(frame["atr"].iloc[last]), 0.0, BASE_WEIGHT, half
        )
        return {
            "bucket": "可买",
            "phase": phase,
            "close": close,
            "text": f"空仓的话，下一交易日开盘可买基础仓。按 {INITIAL_CASH:.0f} 资金估算 {qty} 股。{sizing}",
        }
    return {"bucket": "观察池", "phase": phase, "close": close, "text": story}


def load_bars(conn, symbol: str) -> pd.DataFrame:
    """从已经打开的本地库读一只股票的全部前复权日线。"""
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


def _stock_name(conn, symbol: str) -> str:
    if conn is None or symbol is None:
        return ""
    row = conn.execute("SELECT name FROM stocks WHERE symbol = ?", (str(symbol).zfill(6),)).fetchone()
    if row is None or row[0] is None:
        return ""
    return str(row[0]).strip()


def output_dir(symbol: str) -> Path:
    """默认写到 output/ 下、以 6 位代码命名的目录。"""
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    return OUTPUT_ROOT / str(symbol).strip().zfill(6)


def clip_range(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """指标已在全部历史上算好（含预热），这里只截出回测区间。留空表示不限。"""
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


def run_backtest(symbol: str, start: str | None = None, end: str | None = None, show: bool = True) -> Path:
    if symbol is None or not str(symbol).strip():
        raise ValueError("股票代码不能为空")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"本地数据库不存在：{DB_PATH}")
    code = str(symbol).strip().zfill(6)
    conn = connect(DB_PATH)
    try:
        name = _stock_name(conn, code)
        raw = load_bars(conn, code)
    finally:
        conn.close()
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in",
        category=RuntimeWarning,
    )
    frame = clip_range(prepare(raw), start, end)
    signal = replay(frame)
    journal = Journal()
    journal.add(f"五线探戈 · 单股回测 · {code} {name}".rstrip())
    journal.add(f"区间 {_day_text(frame['date'].iloc[0])} ~ {_day_text(frame['date'].iloc[-1])}，共 {len(frame)} 根日线。")
    journal.add(
        "执行约定：收盘才知道阶段、回撤和买卖理由；买单和卖单交给 AKQuant，在下一根开盘撮合。"
        f"费用和滑点用引擎参数：佣金万三、卖出印花税万五、过户费万一、滑点 {SLIPPAGE:.2%}。"
    )
    journal.add(
        f"仓位：单票计划不超过权益的 {SINGLE_CAP:.0%}，其中基础仓 {BASE_WEIGHT:.0%}、"
        f"趋势加仓 {ADD_WEIGHT:.0%}。战术仓 20% 本轮不做。"
    )
    journal.blank()
    journal.add("========== 每天发生了什么 ==========")
    events: dict[str, list[str]] = {}
    for index, text in signal["events"]:
        if index is None or index < 0 or index >= len(frame):
            continue
        events.setdefault(_day_text(frame["date"].iloc[index]), []).append(text)
    data = to_engine_frame(code, frame, signal)
    strategy = FiveLineStrategy()
    if journal is None:
        raise ValueError("日志不能为空")
    strategy.journal = journal
    strategy.events = events
    strategy.code = code
    strategy.name = name
    strategy.quotes = {
        _day_text(row.date): (row.open, row.close)
        for row in frame.itertuples(index=False)
        if row is not None and getattr(row, "date", None) is not None
    }
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
    folder = output_dir(code)
    path = folder / "five_line_log.txt"
    journal.save(path)
    # 报告里的 K 线买卖点需要行情；基准用这只股票自身的买入持有。
    benchmark = (
        data.set_index("date")["close"].pct_change().fillna(0.0).rename("BUY_AND_HOLD")
    )
    report_path = folder / "report.html"
    result.viz.report(
        title=f"{code} {name} · 五线探戈".rstrip(),
        filename=str(report_path),
        show=show,
        benchmark=benchmark,
        market_data=data,
        plot_symbol=code,
        include_trade_kline=True,
    )
    review_path = folder / "review.html"
    result.viz.review(
        market_data=data,
        title=f"{code} · 交易复盘",
        filename=str(review_path),
        theme="light",
        initial_symbol=code,
        show=False,
    )
    print(
        f"{code} {name} 回测结束：买入 {strategy.buys} 次，卖出 {strategy.sells} 次。"
    )
    print(f"日志：{path}")
    print(f"报告：{report_path}")
    print(f"复盘：{review_path}")
    return report_path


def run_pick() -> Path:
    """扫本地库里每一只股票的最后一天。可买的票再往它自己的代码目录写一份说明。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"本地数据库不存在：{DB_PATH}")
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT d.symbol, COALESCE(s.name, ''), COUNT(*), MAX(d.date)
            FROM daily_bars d
            LEFT JOIN stocks s ON s.symbol = d.symbol
            GROUP BY d.symbol
            ORDER BY d.symbol
            """
        ).fetchall()
        if not rows:
            raise LookupError("本地库没有任何日线")
        buckets = {"可买": [], "观察池": [], "周线拦住": [], "不是候选": [], "历史不够": [], "失败": []}
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            if row is None or row[0] is None:
                continue
            code = str(row[0]).zfill(6)
            name = str(row[1] or "").strip()
            asof = row[3]
            try:
                view = pick_symbol(prepare(load_bars(conn, code)))
            except Exception as exc:
                buckets["失败"].append(f"{code} {name} 计算失败：{exc}")
                continue
            if view is None:
                buckets["失败"].append(f"{code} {name} 没有算出结果")
                continue
            line = f"{code} {name}  截止 {_day_text(asof)}  {view['text']}"
            bucket = view.get("bucket") or "失败"
            buckets.setdefault(bucket, []).append(line)
            if bucket == "可买":
                note = Journal()
                note.add(f"五线探戈 · 大盘选股 · {code} {name}")
                note.add(f"最后交易日 {_day_text(asof)}。这是空仓视角的下一交易日买入候选，不是已经下出去的单。")
                note.add(view["text"])
                note.save(output_dir(code) / "pick.txt")
            if index % 500 == 0 or index == total:
                print(f"选股进度 {index}/{total}，目前可买 {len(buckets['可买'])} 只，观察池 {len(buckets['观察池'])} 只")
    finally:
        conn.close()

    journal = Journal()
    journal.add("五线探戈 · 大盘选股")
    journal.add(f"数据 {DB_PATH}，共扫描 {len(rows)} 只。只使用本地日线，不连外网。")
    journal.add("可买 = 最后一天是 S3/S4，周线许可，并且走出了合格的回踩放量突破，追高和日线 MACD 也过了。")
    journal.add("观察池 = 最后一天是 S3/S4 且周线许可，但今天还没有可执行的买点。")
    journal.add(f"资金估算用 {INITIAL_CASH:.0f}，只为了看 1 手能不能放进 7.5% 的基础仓，不是下单。")
    journal.blank()
    for title in ("可买", "观察池", "周线拦住", "不是候选", "历史不够", "失败"):
        items = buckets.get(title) or []
        journal.add(f"========== {title}：{len(items)} ==========")
        if title in ("不是候选", "历史不够") and len(items) > 30:
            journal.add(f"（只列出前 30 条，其余 {len(items) - 30} 条省略，避免把日志淹没）")
            items = items[:30]
        if not items:
            journal.add("无")
        for item in items:
            journal.add(item)
        journal.blank()
    path = OUTPUT_ROOT / "market" / "pick.txt"
    journal.save(path)
    print(f"选股写完：可买 {len(buckets['可买'])}，观察池 {len(buckets['观察池'])}。汇总：{path}")
    return path


def self_check() -> None:
    """不连数据库。卡住的话说明回撤或费用口径被改坏了。"""
    # 先冲到 10 再跌到 8，后面又到 12。运行高点回撤是 20%，不是用后来的 12 倒算 8 的 33%。
    ok, text = window_verdict(
        np.array([10.0, 8.0, 12.0, 11.0]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([9.0, 9.0, 9.0, 9.0]),
        np.array([1.0, 0.5, 0.5, 1.0]),
    )
    if "20.00%" not in text and "20.0%" not in text:
        raise AssertionError(f"回撤幅度算错：{text}")
    if ok:
        raise AssertionError("最后收盘 11 虽然在均线上，但缩量只有评估内的根数，这个例子不该同时被别的条件放过而不提 20%")
    # 这个例子缩量有 2 根、回撤 20%、最后收盘高于 EMA，但 20% > max(2*0.4/10, 12%)=12%，必须失败。
    if "超过上限" not in text:
        raise AssertionError(text)

    good, good_text = window_verdict(
        np.array([10.0, 9.6, 9.5, 9.7]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([9.0, 9.0, 9.0, 9.0]),
        np.array([0.7, 0.7, 1.0, 1.0]),
    )
    if not good:
        raise AssertionError(good_text)

    volume = pd.Series([1.0] * 21 + [100.0])
    rvol = volume / volume.shift(1).rolling(20, min_periods=20).mean()
    if abs(float(rvol.iloc[-1]) - 100.0) > 1e-9:
        raise AssertionError(f"量比把当天算进去了：{rvol.iloc[-1]}")

    cost = trade_cost("sell", 10_000.0, 100, date(2024, 1, 2), "002432")
    # 佣金 max(5, 3)=5，印花税 5，过户费 0.1。
    if abs(cost - 10.1) > 1e-6:
        raise AssertionError(cost)

    # 网页传参：合法的要生效，未知键 / 小数窗口 / 上下限颠倒都要报错，跑完恢复原值。
    saved = current_params()
    apply_params({"RVOL_HIGH": 1.5, "window_max": 20})
    if RVOL_HIGH != 1.5 or WINDOW_MAX != 20:
        raise AssertionError("参数覆盖没有生效")
    for bad in ({"nope": 1}, {"window_min": 2.5}, {"window_min": 30}, {"lot": True}):
        try:
            apply_params(bad)
        except ValueError:
            continue
        raise AssertionError(f"非法参数没有被拦住：{bad}")
    apply_params(saved)
    try:
        apply_params({"ema_fast": 20, "ema_mid": 10})
    except ValueError:
        pass
    else:
        raise AssertionError("颠倒的 EMA 周期没有被拦住")
    apply_params(saved)
    if [span for _column, span, _color in ema_roles()] != [5, 13, 55, 125, 250]:
        raise AssertionError(f"EMA 默认周期被改坏了：{ema_roles()}")

    one = bad_close_span("2012-07-30", "2012-07-30", 1)
    many = bad_close_span("2012-07-30", "2012-10-16", 50)
    if "当天不产生信号" not in one or "共 50 天" not in many:
        raise AssertionError(f"无效收盘合并写错：{one} / {many}")
    print("selfcheck 通过")


def configure(cfg: dict | None) -> None:
    """把 config.yaml 里的路径和策略参数灌进模块全局。命令行 --params 会在这之后再覆盖一次。"""
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
    apply_params(section(cfg, "five_line"))


def main() -> None:
    global STREAM
    cfg = load_config()
    configure(cfg)
    backtest_cfg = section(cfg, "backtest")
    parser = argparse.ArgumentParser(description="五线探戈：本地数据的单股回测和大盘选股")
    parser.add_argument("command", nargs="?", default="backtest", choices=("backtest", "pick", "selfcheck"))
    parser.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL)
    parser.add_argument("--params", help='JSON，覆盖 config.yaml 的 five_line 参数，如 \'{"rvol_high": 1.5}\'')
    parser.add_argument("--start", default=backtest_cfg.get("start_date") or "", help="回测起始 YYYY-MM-DD")
    parser.add_argument("--end", default=backtest_cfg.get("end_date") or "", help="回测结束 YYYY-MM-DD")
    parser.add_argument("--stream", action="store_true", help="日志逐行打印且不自动打开报告（网页端用）")
    args = parser.parse_args()
    if args.command is None:
        raise ValueError("命令不能为空")
    if args.params:
        try:
            overrides = json.loads(args.params)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--params 不是合法 JSON：{exc}") from exc
        apply_params(overrides)
    STREAM = bool(args.stream)
    if args.command == "selfcheck":
        self_check()
        return
    if args.command == "pick":
        run_pick()
        return
    if args.symbol is None or not str(args.symbol).strip():
        raise ValueError("回测股票代码不能为空")
    run_backtest(args.symbol, start=args.start, end=args.end, show=not args.stream)


if __name__ == "__main__":
    main()
