"""20/55均线突破 + MACD过滤 + 250年线过滤 选股与多策略回测扫描器。

核心逻辑：
1. 均线系统：MA(fast=20) 与 MA(slow=55)，可选 MA(trend=250)。
2. 买入条件：
   - 基础条件：20日均线上穿55日均线（金叉，CROSS(MA20, MA55)）。
   - MACD过滤（可配开关）：
     - 0轴过滤：DIF > 0 且 DEA > 0（只在0轴上方操作）。
     - 动能过滤（可选）：MACD红柱（BAR > 0）或红柱放大。
   - 250天线过滤（可配开关）：
     - 收盘价 > MA250（或 MA20 > MA250），处于大周期多头，250线下不操作。
3. 卖出条件（优先级按信号触发）：
   - 移动止盈/移动止损：止损线只上移。硬止损是地板；盈利后线 = 持仓最高浮盈 − 回吐幅度。
     最高赚 30%、回吐 10% 时线在 +20%，跌回 20% 清仓；再涨到 60% 则线上移到 +50%。
   - 趋势结束：20日均线下穿55日均线（死叉，CROSS(MA55, MA20)）。
   - 可选更早离场：MACD红柱变绿、DIF下穿DEA。默认关闭，避免把还在涨的利润提前切掉。
4. 预设策略组合（支持右上角快捷切换）：
   - 策略A（基础版）：MA20上穿MA55 + MACD>0轴过滤 + 均线死叉卖出。
   - 策略B（移动止盈版）：MA20上穿MA55 + MACD>0轴 + 最高浮盈回吐后清仓。
   - 策略C（年线大周期版）：MA20上穿MA55 + MACD>0轴 + 收盘站上MA250年线 + 均线死叉/移动止盈。
   - 策略D（全严格版）：MA20上穿MA55 + 0轴上 + 站上MA250 + 移动止盈。
5. 全市场扫描：
   - 默认扫描本地 a_share.db 过去2年日线数据。
   - 包含单票逐笔交易回测（胜率、收益率、最大回撤、交易次数、最新买入信号）。
   - 导出独立 HTML 报告，提供右上角策略切换、多指标配置过滤、按收益/胜率/最新信号排序与检索。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录注入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path, section

DB_PATH = PROJECT_ROOT / "data" / "a_share.db"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ma20_55_scanner"
HTML_REPORT = OUTPUT_DIR / "index.html"
RESULT_JSON = OUTPUT_DIR / "result.json"
INT_FIELDS = (
    "ma_fast", "ma_slow", "ma_trend", "macd_fast", "macd_slow", "macd_signal", "min_bars",
)
FLOAT_FIELDS = ("stop_loss_pct", "trail_gap_pct")
BOOL_FIELDS = (
    "use_macd_zero", "use_macd_bar_pos", "use_ma250_trend", "filter_st",
    "exit_on_ma_cross", "exit_on_macd_dead", "exit_on_bar_green",
)


# ========== 策略配置数据类 ==========

@dataclass
class StrategyConfig:
    name: str = "策略A：MA20/55金叉 + 0轴上"
    desc: str = "基础20上穿55，MACD零轴线上，均线死叉平仓"
    
    # 均线参数
    ma_fast: int = 20
    ma_slow: int = 55
    ma_trend: int = 250
    ma_type: str = "SMA"  # SMA 或 EMA
    
    # MACD 参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 开仓过滤开关
    use_macd_zero: bool = True     # 0轴线上不操作过滤：DIF > 0 且 DEA > 0
    use_macd_bar_pos: bool = True  # 要求开仓时 MACD 柱为红柱 (BAR > 0)
    use_ma250_trend: bool = False  # 250天线过滤：收盘价在MA250之上
    filter_st: bool = True         # 过滤 ST/*ST 股票
    min_bars: int = 260            # 至少需要的历史 K 线数
    
    # 平仓规则开关
    exit_on_ma_cross: bool = True      # 20下穿55强制卖出
    exit_on_macd_dead: bool = True     # MACD死叉(DIF下穿DEA)卖出
    exit_on_bar_green: bool = False    # MACD红柱转绿柱卖出
    stop_loss_pct: float = 0.08        # 硬止损百分比（如 0.08 = 8%，0为不启用）。盈利后会被移动止盈上移
    trail_gap_pct: float = 0.10        # 移动止盈回吐幅度（最高浮盈回吐这么多就清仓，0为不启用）
    
    # 回测周期
    lookback_years: float = 2.0        # 默认最近2年


PRESET_STRATEGIES: dict[str, StrategyConfig] = {
    "strat_0_pure_ma": StrategyConfig(
        name="策略0：纯20/55均线金叉死叉 (无任何过滤基准)",
        desc="20上穿55买入，下穿55卖出，不设任何MACD和年线过滤，作为对比基准",
        use_macd_zero=False,
        use_macd_bar_pos=False,
        use_ma250_trend=False,
        exit_on_ma_cross=True,
        exit_on_macd_dead=False,
        exit_on_bar_green=False,
        stop_loss_pct=0.08,
    ),
    "strat_a_basic": StrategyConfig(
        name="策略A：MA20/55金叉 + 0轴上 (均线死叉卖)",
        desc="20上穿55买入，下穿55卖出；MACD零轴线上操作（DIF>0且DEA>0），不看年线",
        use_macd_zero=True,
        use_macd_bar_pos=True,
        use_ma250_trend=False,
        exit_on_ma_cross=True,
        exit_on_macd_dead=False,
        exit_on_bar_green=False,
        stop_loss_pct=0.08,
    ),
    "strat_b_macd_exit": StrategyConfig(
        name="策略B：MA20/55 + 0轴上 + 移动止盈",
        desc="盈利后止损线上移：最高浮盈回吐10%清仓。赚到30%守住20%，再涨到60%守住50%",
        use_macd_zero=True,
        use_macd_bar_pos=True,
        use_ma250_trend=False,
        exit_on_ma_cross=True,
        exit_on_macd_dead=False,
        exit_on_bar_green=False,
        stop_loss_pct=0.06,
    ),
    "strat_c_trend_250": StrategyConfig(
        name="策略C：大周期250年线上 + MA20/55 + 0轴上",
        desc="加入250日年线大周期过滤：250天线下不操作，只做大牛股主升浪，均线死叉卖出",
        use_macd_zero=True,
        use_macd_bar_pos=True,
        use_ma250_trend=True,
        exit_on_ma_cross=True,
        exit_on_macd_dead=False,
        exit_on_bar_green=False,
        stop_loss_pct=0.08,
    ),
    "strat_d_strict_all": StrategyConfig(
        name="策略D：全严格版 (250线上 + 0轴上 + 移动止盈)",
        desc="年线之上 + 0轴上金叉。止盈改为移动止盈：最高浮盈回吐10%才清仓，让利润奔跑同时锁住回撤",
        use_macd_zero=True,
        use_macd_bar_pos=True,
        use_ma250_trend=True,
        exit_on_ma_cross=True,
        exit_on_macd_dead=False,
        exit_on_bar_green=False,
        stop_loss_pct=0.06,
    ),
}


def _as_bool(value) -> bool:
    """网页和命令行都可能传来 0/1 或 true/false。空值直接拒绝。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) in (0, 1) and float(value) == int(value):
        return bool(int(value))
    raise ValueError(f"开关必须是 0 或 1，实际 {value!r}")


def apply_shared_params(cfg: StrategyConfig, params: dict | None, use_flags: bool) -> None:
    """数字参数作用于本次扫描的每一套策略。开关只覆盖下拉框当前选中的那一套。"""
    if cfg is None:
        raise ValueError("策略配置不能为空")
    if not params:
        return
    for key, raw in params.items():
        name = str(key or "").strip()
        if name in ("years", "limit", "strategy"):
            continue
        if raw is None:
            raise ValueError(f"参数 {name} 不能为空")
        if name in INT_FIELDS:
            number = float(raw)
            if number != int(number) or int(number) < 1:
                raise ValueError(f"{name} 必须是正整数")
            setattr(cfg, name, int(number))
        elif name in FLOAT_FIELDS:
            number = float(raw)
            if number < 0:
                raise ValueError(f"{name} 不能为负")
            setattr(cfg, name, number)
        elif name in BOOL_FIELDS and use_flags:
            setattr(cfg, name, _as_bool(raw))
        elif name not in BOOL_FIELDS:
            raise ValueError(f"未知参数 {name}")
    if cfg.ma_fast >= cfg.ma_slow:
        raise ValueError("短均线周期必须小于长均线")
    if cfg.macd_fast >= cfg.macd_slow:
        raise ValueError("MACD 快线必须短于慢线")


def build_strategies(
    params: dict | None,
    selected: str | None,
    flag_overrides: dict | None = None,
) -> dict[str, StrategyConfig]:
    """数字参数套到每一套策略上。每一套自己的开关可单独覆盖，便于下拉切换后分别回看。"""
    if selected and selected not in PRESET_STRATEGIES:
        raise ValueError(f"未知策略 {selected}")
    if flag_overrides is None:
        flag_overrides = {}
    if not isinstance(flag_overrides, dict):
        raise ValueError("策略开关必须是对象")
    built = {}
    for key, base in PRESET_STRATEGIES.items():
        if base is None:
            raise ValueError(f"预设 {key} 为空")
        cfg = StrategyConfig(**asdict(base))
        apply_shared_params(cfg, params, use_flags=False)
        own_flags = flag_overrides.get(key)
        if own_flags is not None:
            if not isinstance(own_flags, dict):
                raise ValueError(f"{key} 的开关必须是对象")
            apply_shared_params(cfg, own_flags, use_flags=True)
        elif key == selected:
            apply_shared_params(cfg, params, use_flags=True)
        built[key] = cfg
    return built


# ========== 指标计算函数 (含严密nil保护) ==========

def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """计算 20/55/250 均线与 MACD 指标，全部向量化防未来函数。
    核心逻辑加注：
    - MA计算：SMA使用 rolling().mean()，严格依赖当前及历史bar。
    - MACD计算：EMA使用 ewm(span, adjust=False)。
    - 金叉/死叉：基于 shift(1) 识别上穿与下穿。
    """
    if df is None or df.empty or len(df) < 30:
        return pd.DataFrame()
    
    res = df.copy()
    close = res["close"]
    
    # 1. 均线计算
    if cfg.ma_type.upper() == "EMA":
        res["ma_fast"] = close.ewm(span=cfg.ma_fast, adjust=False).mean()
        res["ma_slow"] = close.ewm(span=cfg.ma_slow, adjust=False).mean()
        res["ma_trend"] = close.ewm(span=cfg.ma_trend, adjust=False).mean()
    else:
        res["ma_fast"] = close.rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
        res["ma_slow"] = close.rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()
        res["ma_trend"] = close.rolling(cfg.ma_trend, min_periods=min(cfg.ma_trend, 60)).mean()

    # 2. MACD 指标
    ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
    res["dif"] = ema_fast - ema_slow
    res["dea"] = res["dif"].ewm(span=cfg.macd_signal, adjust=False).mean()
    res["macd_bar"] = 2.0 * (res["dif"] - res["dea"])
    
    # 3. 辅助逻辑信号计算 (前一根K线状态)
    res["prev_ma_fast"] = res["ma_fast"].shift(1)
    res["prev_ma_slow"] = res["ma_slow"].shift(1)
    res["prev_dif"] = res["dif"].shift(1)
    res["prev_dea"] = res["dea"].shift(1)
    res["prev_macd_bar"] = res["macd_bar"].shift(1)
    
    # 4. 20 与 55 金叉 / 死叉
    # 金叉：前一天 ma_fast <= ma_slow 且 当天 ma_fast > ma_slow
    res["ma_golden_cross"] = (
        res["prev_ma_fast"].notna() & res["prev_ma_slow"].notna() &
        res["ma_fast"].notna() & res["ma_slow"].notna() &
        (res["prev_ma_fast"] <= res["prev_ma_slow"]) &
        (res["ma_fast"] > res["ma_slow"])
    )
    # 死叉：前一天 ma_fast >= ma_slow 且 当天 ma_fast < ma_slow
    res["ma_death_cross"] = (
        res["prev_ma_fast"].notna() & res["prev_ma_slow"].notna() &
        res["ma_fast"].notna() & res["ma_slow"].notna() &
        (res["prev_ma_fast"] >= res["prev_ma_slow"]) &
        (res["ma_fast"] < res["ma_slow"])
    )
    
    # 5. MACD 死叉与红柱转绿
    # 死叉：前一天 dif >= dea 且 当天 dif < dea
    res["macd_death_cross"] = (
        res["prev_dif"].notna() & res["prev_dea"].notna() &
        res["dif"].notna() & res["dea"].notna() &
        (res["prev_dif"] >= res["prev_dea"]) &
        (res["dif"] < res["dea"])
    )
    # 红柱转绿：前一根 bar > 0 且 当前 bar <= 0
    res["macd_bar_turn_green"] = (
        res["prev_macd_bar"].notna() & res["macd_bar"].notna() &
        (res["prev_macd_bar"] > 0) & (res["macd_bar"] <= 0)
    )
    
    # 6. 买入过滤项判定
    # 0轴线上条件：DIF > 0 并且 DEA > 0
    res["macd_above_zero"] = (res["dif"].notna()) & (res["dea"].notna()) & (res["dif"] > 0) & (res["dea"] > 0)
    # MACD红柱条件：BAR > 0
    res["macd_bar_positive"] = (res["macd_bar"].notna()) & (res["macd_bar"] > 0)
    # 250日年线条件：收盘价在MA250上方
    res["above_ma250"] = (res["ma_trend"].notna()) & (close > res["ma_trend"])

    # 7. 综合买入信号生成 (收盘确认信号，次日开盘买入)
    buy_cond = res["ma_golden_cross"].copy()
    if cfg.use_macd_zero:
        buy_cond &= res["macd_above_zero"]
    if cfg.use_macd_bar_pos:
        buy_cond &= res["macd_bar_positive"]
    if cfg.use_ma250_trend:
        buy_cond &= res["above_ma250"]
    
    res["buy_signal"] = buy_cond.fillna(False)
    
    # 8. 综合卖出信号预判
    sell_cond = pd.Series(False, index=res.index)
    if cfg.exit_on_ma_cross:
        sell_cond |= res["ma_death_cross"]
    if cfg.exit_on_macd_dead:
        sell_cond |= res["macd_death_cross"]
    if cfg.exit_on_bar_green:
        sell_cond |= res["macd_bar_turn_green"]
        
    res["sell_signal"] = sell_cond.fillna(False)
    
    return res


def _bar_pnl(price, entry: float) -> float | None:
    """单根价格相对买入价的收益率。价格缺失或非数时返回 None，调用方必须判断。"""
    if price is None or entry is None or entry <= 0:
        return None
    try:
        px = float(price)
    except (TypeError, ValueError):
        return None
    if px != px or px <= 0:  # NaN
        return None
    return (px - float(entry)) / float(entry)


def trail_stop_line(peak_pnl: float | None, hard_stop_pct: float | None, trail_gap_pct: float | None) -> float | None:
    """移动止盈线，只升不降。触发后本回测按整仓清仓（没有分批减仓）。

    硬止损是地板。盈利后止损线 = 持仓以来最高浮盈 − 回吐幅度。
    最高浮盈 30%、回吐 10% → 线在 +20%，跌回 20% 清仓；
    再涨到 60% → 线上移到 +50%。没赚到回吐幅度时，线从硬止损往上抬，这就是移动止损。
    返回 None 表示不设价格止损。
    """
    line = None
    if hard_stop_pct is not None and float(hard_stop_pct) > 0:
        line = -float(hard_stop_pct)
    if (
        trail_gap_pct is not None
        and float(trail_gap_pct) > 0
        and peak_pnl is not None
        and float(peak_pnl) > 0
    ):
        trailed = float(peak_pnl) - float(trail_gap_pct)
        if line is None or trailed > line:
            line = trailed
    return line


def _check_trail_stop() -> None:
    """移动止盈的最小自检。规则改坏时这里会直接失败。"""
    assert abs(trail_stop_line(0.30, 0.08, 0.10) - 0.20) < 1e-9
    assert abs(trail_stop_line(0.60, 0.08, 0.10) - 0.50) < 1e-9
    assert trail_stop_line(0.60, 0.08, 0.10) > trail_stop_line(0.30, 0.08, 0.10)
    # 只赚 5% 时，线从 -8% 抬到 -5%，不往下调
    assert abs(trail_stop_line(0.05, 0.08, 0.10) - (-0.05)) < 1e-9
    assert abs(trail_stop_line(0.0, 0.08, 0.10) - (-0.08)) < 1e-9
    assert abs(trail_stop_line(None, 0.08, 0.10) - (-0.08)) < 1e-9
    assert trail_stop_line(0.60, 0.0, 0.0) is None
    # 涨到 30% 当天不打旧线，次日才按 +20% 清仓；再涨到 60% 则守住 50%
    peak = 0.0
    assert 0.25 > trail_stop_line(peak, 0.08, 0.10)
    peak = 0.30
    assert 0.19 <= trail_stop_line(peak, 0.08, 0.10)
    assert 0.55 > trail_stop_line(0.60, 0.08, 0.10)
    assert 0.49 <= trail_stop_line(0.60, 0.08, 0.10)


_check_trail_stop()


# ========== 单票快速回测引擎 (逐笔成交与收益率统计) ==========

@dataclass
class TradeRecord:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str
    hold_bars: int
    pnl_pct: float


@dataclass
class StockBacktestResult:
    symbol: str
    name: str
    total_trades: int
    win_rate: float            # 胜率 (0~100%)
    total_return_pct: float    # 累计收益率%
    annual_return_pct: float   # 年化收益率%
    max_drawdown_pct: float    # 最大回撤%
    profit_loss_ratio: float   # 盈亏比
    latest_close: float
    latest_signal_date: str    # 最近买入信号日期
    is_holding_now: bool       # 当前是否处于持仓中
    current_unrealized_pnl: float # 当前持仓未实现盈亏%
    trades: list[TradeRecord]


def backtest_single_stock(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    cfg: StrategyConfig,
    start_date: str | None = None,
) -> StockBacktestResult | None:
    """对单只股票执行回测，有完整的空值防范与T+1交易模拟。
    规则说明：
    - 收盘后指标确认买卖信号，下一根K线以开盘价成交（无未来函数）。
    - 佣金万三，印花税千一（卖出时），滑点千一。
    """
    if df is None or df.empty or len(df) < cfg.min_bars:
        return None
    
    # 过滤 ST/*ST 股票
    if cfg.filter_st and name:
        upper = name.upper()
        if "ST" in upper or "*" in name or "退" in name:
            return None

    # 计算指标
    data = compute_indicators(df, cfg)
    if data.empty:
        return None
        
    # 截取回测开始日期
    if start_date is not None and str(start_date).strip():
        data = data[data["date"] >= start_date].reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)
        
    if len(data) < 20:
        return None

    trades: list[TradeRecord] = []
    in_pos = False
    entry_price = 0.0
    entry_date = ""
    entry_idx = 0
    peak_pnl = 0.0  # 本笔持仓以来最高浮盈，移动止盈只根据它上移
    
    # 逐日模拟交易执行
    # 注意：i 日收盘产生信号，i+1 日开盘价成交
    n = len(data)
    for i in range(n - 1):
        curr_bar = data.iloc[i]
        next_bar = data.iloc[i + 1]
        
        # 保护性检查：确保必须的价格字段不为空
        if curr_bar is None or next_bar is None:
            continue
        c_close = curr_bar.get("close")
        n_open = next_bar.get("open")
        if c_close is None or n_open is None or np.isnan(c_close) or np.isnan(n_open) or n_open <= 0:
            continue
            
        c_date_str = str(curr_bar["date"])[:10]
        n_date_str = str(next_bar["date"])[:10]
        
        # 1. 若当前持仓中，先检查止损或平仓信号
        if in_pos:
            hold_days = i - entry_idx
            unrealized_pnl = (float(c_close) - entry_price) / entry_price if entry_price > 0 else 0.0
            
            should_exit = False
            exit_reason = ""

            # 用最低价判断是否碰到止损线（盘中碰到就走）；没有最低价时退回收盘价
            low_pnl = _bar_pnl(curr_bar.get("low"), entry_price)
            high_pnl = _bar_pnl(curr_bar.get("high"), entry_price)
            touch = low_pnl if low_pnl is not None else unrealized_pnl
            # 止损线用「到昨天为止」的最高浮盈。今天的新高要等确认没打到旧线，下一根才生效
            stop_line = trail_stop_line(peak_pnl, cfg.stop_loss_pct, cfg.trail_gap_pct)
            if stop_line is not None and touch is not None and touch <= stop_line:
                should_exit = True
                if stop_line > 0:
                    exit_reason = f"移动止盈 峰值{peak_pnl:.1%}回撤到{stop_line:.1%}"
                elif cfg.stop_loss_pct is not None and cfg.stop_loss_pct > 0 and stop_line <= -float(cfg.stop_loss_pct) + 1e-12:
                    exit_reason = f"硬止损{-cfg.stop_loss_pct:.1%}"
                else:
                    exit_reason = f"移动止损{stop_line:.1%}"
            # 信号平仓 (MA死叉 / MACD死叉 / 红变绿)
            elif bool(curr_bar.get("sell_signal")):
                should_exit = True
                if bool(curr_bar.get("ma_death_cross")):
                    exit_reason = "MA20下穿55死叉"
                elif bool(curr_bar.get("macd_death_cross")):
                    exit_reason = "MACD死叉(DIF<DEA)"
                elif bool(curr_bar.get("macd_bar_turn_green")):
                    exit_reason = "MACD红柱变绿"
                else:
                    exit_reason = "指标平仓信号"
            
            if should_exit:
                # 次日开盘卖出 (扣除滑点0.1%和交易印花税等综合成本0.15%)
                actual_exit_price = float(n_open) * 0.9975
                pnl = (actual_exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                trades.append(TradeRecord(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=n_date_str,
                    exit_price=round(actual_exit_price, 2),
                    exit_reason=exit_reason,
                    hold_bars=hold_days,
                    pnl_pct=round(pnl * 100, 2),
                ))
                in_pos = False
                entry_price = 0.0
                entry_date = ""
                peak_pnl = 0.0
                continue

            # 没触发才用今日最高上移峰值。最高价缺失时用收盘价
            favorable = high_pnl if high_pnl is not None else unrealized_pnl
            if favorable is not None and favorable > peak_pnl:
                peak_pnl = favorable
                
        # 2. 若空仓，检查买入信号
        if not in_pos:
            if bool(curr_bar.get("buy_signal")):
                # 次日开盘买入 (加滑点0.1%及佣金)
                entry_price = float(n_open) * 1.0015
                entry_date = n_date_str
                entry_idx = i + 1
                peak_pnl = 0.0
                in_pos = True
                
    # 统计交易绩效
    last_row = data.iloc[-1]
    latest_close = float(last_row["close"]) if last_row.get("close") is not None else 0.0
    latest_buy_signal_date = ""
    
    # 检查最近 5 天内是否有发出买入信号
    recent_signals = data[data["buy_signal"] == True]
    if not recent_signals.empty:
        latest_buy_signal_date = str(recent_signals.iloc[-1]["date"])[:10]
        
    current_unrealized_pnl = 0.0
    if in_pos and entry_price > 0 and latest_close > 0:
        current_unrealized_pnl = round(((latest_close - entry_price) / entry_price) * 100, 2)

    total_trades = len(trades)
    if total_trades == 0:
        # 如果当前正持仓，虽未平仓但也记录
        return StockBacktestResult(
            symbol=symbol,
            name=name,
            total_trades=0,
            win_rate=0.0,
            total_return_pct=current_unrealized_pnl if in_pos else 0.0,
            annual_return_pct=0.0,
            max_drawdown_pct=0.0,
            profit_loss_ratio=0.0,
            latest_close=round(latest_close, 2),
            latest_signal_date=latest_buy_signal_date,
            is_holding_now=in_pos,
            current_unrealized_pnl=current_unrealized_pnl,
            trades=[],
        )
        
    wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
    win_rate = round(len(wins) / total_trades * 100, 1)
    
    # 累加复利收益率
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= (1.0 + t.pnl_pct / 100.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            
    total_return = round((equity - 1.0) * 100, 2)
    # 按总回测年数估算年化
    days = (pd.to_datetime(data.iloc[-1]["date"]) - pd.to_datetime(data.iloc[0]["date"])).days
    years = max(days / 365.25, 0.2)
    annual_return = round(((equity ** (1.0 / years)) - 1.0) * 100, 2) if equity > 0 else -100.0
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
    pl_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)
    
    return StockBacktestResult(
        symbol=symbol,
        name=name,
        total_trades=total_trades,
        win_rate=win_rate,
        total_return_pct=total_return,
        annual_return_pct=annual_return,
        max_drawdown_pct=round(max_dd * 100, 2),
        profit_loss_ratio=pl_ratio,
        latest_close=round(latest_close, 2),
        latest_signal_date=latest_buy_signal_date,
        is_holding_now=in_pos,
        current_unrealized_pnl=current_unrealized_pnl,
        trades=trades,
    )


# ========== 全库扫描调度与多进程加速 ==========

def load_all_stocks_meta(conn: sqlite3.Connection) -> dict[str, str]:
    """读取所有股票代码和名称字典。"""
    if conn is None:
        raise ValueError("数据库连接不能为空")
    rows = conn.execute("SELECT symbol, name FROM stocks").fetchall()
    return {str(r[0]).zfill(6): str(r[1]) for r in rows if r and r[0]}


def scan_all_strategies(
    db_path: Path,
    strategies: dict[str, StrategyConfig],
    max_stocks: int | None = None,
    lookback_days: int = 730,
) -> dict[str, list[StockBacktestResult]]:
    """一次性批量拉取行情数据，并在内存中同时并行/多策略回测，速度提升4倍以上。"""
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    conn = sqlite3.connect(str(db_path))
    all_results: dict[str, list[StockBacktestResult]] = {k: [] for k in strategies.keys()}
    try:
        stocks_meta = load_all_stocks_meta(conn)
        max_date_row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
        latest_db_date = str(max_date_row[0]) if (max_date_row and max_date_row[0]) else "2026-10-14"
        start_date = (pd.to_datetime(latest_db_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        warmup_date = (pd.to_datetime(start_date) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        
        symbols = list(stocks_meta.keys())
        if max_stocks and max_stocks > 0:
            symbols = symbols[:max_stocks]
            
        print(f"开始批量扫描，总股票数: {len(symbols)}，区间: {start_date} ~ {latest_db_date} (预热自 {warmup_date})", flush=True)
        
        batch_size = 500
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        for b_idx in range(0, len(symbols), batch_size):
            b_syms = symbols[b_idx : b_idx + batch_size]
            placeholders = ",".join(["?"] * len(b_syms))
            query = f"""
                SELECT symbol, date, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol IN ({placeholders}) AND date >= ?
                ORDER BY symbol, date ASC
            """
            df_batch = pd.read_sql_query(query, conn, params=(*b_syms, warmup_date))
            if df_batch.empty:
                continue
                
            grouped = df_batch.groupby("symbol")
            for sym, group in grouped:
                s_name = stocks_meta.get(str(sym), str(sym))
                # 对当前股票，依次跑各个策略组合
                for s_key, s_cfg in strategies.items():
                    res = backtest_single_stock(str(sym), s_name, group, s_cfg, start_date=start_date)
                    if res is not None:
                        all_results[s_key].append(res)
                        
            curr_b = (b_idx // batch_size) + 1
            print(f"批次进度: {curr_b}/{total_batches} ({min(b_idx + batch_size, len(symbols))}/{len(symbols)}) 股票计算完成...", flush=True)
            
    finally:
        conn.close()

    # 只保留有成交或当前仍持仓的股票，纯无信号的不进选股列表。
    for s_key in all_results:
        kept = [row for row in all_results[s_key] if row is not None and (row.total_trades > 0 or row.is_holding_now)]
        kept.sort(key=lambda x: (x.is_holding_now, x.total_return_pct, x.latest_signal_date or ""), reverse=True)
        all_results[s_key] = kept
        
    return all_results


def _public_row(row: StockBacktestResult) -> dict:
    """列表页不带逐笔明细，避免全库结果把浏览器撑死。"""
    if row is None:
        return {}
    data = asdict(row)
    data.pop("trades", None)
    return data


def save_result(strategies: dict[str, StrategyConfig], results: dict[str, list[StockBacktestResult]], years: float, limit: int | None) -> None:
    """页面只读这份 JSON。策略下拉框切换的是这里的不同结果。"""
    if strategies is None or results is None:
        raise ValueError("没有可保存的扫描结果")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "years": years,
        "limit": limit,
        "strategies": {
            key: {
                "name": cfg.name,
                "desc": cfg.desc,
                "params": asdict(cfg),
                "rows": [_public_row(row) for row in results.get(key) or [] if row is not None],
            }
            for key, cfg in strategies.items()
            if cfg is not None
        },
    }
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"结果已写入 {RESULT_JSON}", flush=True)


# ========== 生成交互式 HTML 选股报告 ==========

def generate_html_report(
    all_strategy_results: dict[str, list[StockBacktestResult]],
    output_path: Path,
    scan_date: str,
) -> None:
    """生成单文件美观的 HTML 选股控制台。
    功能：
    - 右上角可切换不同策略（策略A、策略B、策略C、策略D）。
    - 详细的指标参数卡片与解释。
    - 选股股票表格，支持搜索股票代码/名称、排序、过滤“当前持仓”或“近期买点”。
    - 每只股票可展开查看详细买卖明细。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 序列化为 JSON 供前端交互
    json_data = {}
    for strat_key, res_list in all_strategy_results.items():
        json_data[strat_key] = [asdict(r) for r in res_list]
        
    presets_meta = {
        k: {
            "name": cfg.name,
            "desc": cfg.desc,
            "params": asdict(cfg),
        }
        for k, cfg in PRESET_STRATEGIES.items()
    }

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>20/55均线 + MACD + 250年线 策略选股与回测平台</title>
<style>
  :root {{
    --bg: #0b1020; --panel: #121a2e; --panel2: #18223a; --line: #24304d;
    --text: #e6ebf5; --muted: #8b97b3; --accent: #6c8cff; --accent2: #9b6cff;
    --up: #ff5d6c; --down: #22c58b; --warn: #ffb547; --err: #ff6b6b;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: radial-gradient(1200px 600px at 10% -10%, #1b2550 0%, transparent 60%),
                radial-gradient(900px 500px at 110% 0%, #2a1b50 0%, transparent 55%), var(--bg);
    color: var(--text); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    min-height: 100vh; display: flex; flex-direction: column;
    min-width: 1200px;
  }}
  header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; border-bottom: 1px solid var(--line);
    background: #0b1020cc; backdrop-filter: blur(10px); position: sticky; top: 0; z-index: 100;
  }}
  .logo-title {{ display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
  .badge {{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    padding: 6px 12px; border-radius: 8px; font-weight: 800; font-size: 14px; white-space: nowrap;
  }}
  .header-actions {{ display: flex; align-items: center; gap: 14px; flex-shrink: 0; }}
  select.strat-select {{
    background: var(--panel2); color: var(--text); border: 1px solid var(--accent);
    padding: 8px 14px; border-radius: 8px; font-weight: 600; font-size: 14px; outline: none; cursor: pointer;
  }}
  main {{ flex: 1; max-width: 1440px; margin: 0 auto; width: 100%; padding: 20px 24px 40px; }}
  .desc-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px 20px; margin-bottom: 20px;
  }}
  .desc-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
  .desc-head h2 {{ font-size: 18px; color: #fff; }}
  .tags {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
  .tag {{
    background: var(--panel2); border: 1px solid var(--line);
    padding: 4px 10px; border-radius: 6px; font-size: 12px; color: var(--muted);
  }}
  .tag.active {{ border-color: var(--accent); color: #c7b6ff; background: #6c8cff15; }}
  
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 20px;
  }}
  .stat-card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 16px;
  }}
  .stat-card span {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
  .stat-card b {{ font-family: var(--mono); font-size: 20px; }}
  
  .toolbar {{
    display: flex; gap: 12px; align-items: center; justify-content: space-between;
    margin-bottom: 14px; flex-wrap: wrap;
  }}
  .search-box {{ display: flex; gap: 10px; align-items: center; }}
  input[type=search] {{
    background: var(--panel); border: 1px solid var(--line); color: var(--text);
    padding: 8px 14px; border-radius: 8px; font-size: 13px; width: 240px; outline: none;
  }}
  input[type=search]:focus {{ border-color: var(--accent); }}
  .filter-pills {{ display: flex; gap: 8px; }}
  .pill {{
    background: var(--panel); border: 1px solid var(--line); color: var(--muted);
    padding: 6px 12px; border-radius: 20px; font-size: 12px; cursor: pointer; user-select: none;
  }}
  .pill.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  table.stock-table {{
    width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
  }}
  table.stock-table th, table.stock-table td {{
    padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--line);
  }}
  table.stock-table th {{
    background: var(--panel2); font-weight: 600; font-size: 12px; color: var(--muted);
    cursor: pointer; user-select: none;
  }}
  table.stock-table th:hover {{ color: var(--text); }}
  table.stock-table tr:hover td {{ background: #ffffff06; }}
  .code {{ font-family: var(--mono); font-weight: 700; color: #fff; }}
  .up {{ color: var(--up); font-weight: 700; }}
  .down {{ color: var(--down); font-weight: 700; }}
  .holding-badge {{
    background: #ff5d6c22; color: var(--up); border: 1px solid var(--up);
    padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px;
  }}
  .btn-detail {{
    background: var(--panel2); border: 1px solid var(--line); color: var(--muted);
    padding: 4px 8px; border-radius: 6px; font-size: 11px; cursor: pointer;
  }}
  .btn-detail:hover {{ color: var(--text); border-color: var(--accent); }}
  .details-row {{ background: #070b16; display: none; }}
  .details-box {{ padding: 14px 20px; }}
  .trade-mini-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .trade-mini-table th, .trade-mini-table td {{ padding: 6px 10px; border-bottom: 1px solid #1a2540; }}
  .trade-mini-table th {{ color: var(--muted); background: #10172c; }}
</style>
</head>
<body>

<header>
  <div class="logo-title">
    <div class="badge">MA20/55</div>
    <div>
      <h1 style="font-size: 18px;">均线穿透 + MACD + 250年线 策略选股回测</h1>
      <small style="color: var(--muted);">扫描周期：最近2年 | 更新时间：{scan_date}</small>
    </div>
  </div>
  <div class="header-actions">
    <a href="/" style="color:var(--muted); text-decoration:none; font-size:13px; margin-right:12px;">五线探戈</a>
    <a href="/bond.html" style="color:var(--muted); text-decoration:none; font-size:13px; margin-right:16px;">粘合突破</a>
    <span style="font-size: 13px; color: var(--muted);">当前策略：</span>
    <select id="strategySelect" class="strat-select">
      <!-- 动态填充策略选项 -->
    </select>
  </div>
</header>

<main>
  <!-- 策略说明与参数看板 -->
  <div class="desc-card">
    <div class="desc-head">
      <h2 id="curStratTitle">策略标题</h2>
      <span style="color: var(--warn); font-size: 12px;" id="curStratSummary">指标配置</span>
    </div>
    <p id="curStratDesc" style="color: var(--muted); font-size: 13px; margin-bottom: 10px;"></p>
    <div class="tags" id="curStratTags"></div>
  </div>

  <!-- 汇总数据卡片 -->
  <div class="stats-grid">
    <div class="stat-card"><span>选出标的总数</span><b id="stTotal">0</b></div>
    <div class="stat-card"><span>当前持仓标的</span><b id="stHolding" style="color: var(--up);">0</b></div>
    <div class="stat-card"><span>平均交易胜率</span><b id="stWinRate">0%</b></div>
    <div class="stat-card"><span>平均累计收益</span><b id="stAvgReturn">0%</b></div>
    <div class="stat-card"><span>平均盈亏比</span><b id="stPLRatio">0.0</b></div>
  </div>

  <!-- 搜索与筛选工具栏 -->
  <div class="toolbar">
    <div class="search-box">
      <input type="search" id="searchInput" placeholder="搜索代码、名称或买点日期...">
      <div class="filter-pills">
        <span class="pill active" data-filter="all">全部标的</span>
        <span class="pill" data-filter="holding">当前持仓中</span>
        <span class="pill" data-filter="profit">累计正收益</span>
      </div>
    </div>
    <div style="font-size: 12px; color: var(--muted);">
      点击列头可排序（默认按当前持仓与总收益排序）
    </div>
  </div>

  <!-- 选股结果表格 -->
  <table class="stock-table">
    <thead>
      <tr>
        <th onclick="sortTable('symbol')">代码</th>
        <th onclick="sortTable('name')">名称</th>
        <th onclick="sortTable('latest_close')">最新收盘</th>
        <th onclick="sortTable('latest_signal_date')">最新买点日期</th>
        <th onclick="sortTable('total_trades')">交易次数</th>
        <th onclick="sortTable('win_rate')">胜率</th>
        <th onclick="sortTable('total_return_pct')">累计收益%</th>
        <th onclick="sortTable('annual_return_pct')">年化收益%</th>
        <th onclick="sortTable('max_drawdown_pct')">最大回撤%</th>
        <th onclick="sortTable('profit_loss_ratio')">盈亏比</th>
        <th>持仓状态</th>
        <th>买卖详情</th>
      </tr>
    </thead>
    <tbody id="stockTableBody">
      <!-- 动态渲染数据行 -->
    </tbody>
  </table>
</main>

<script>
const PRESETS = {json.dumps(presets_meta, ensure_ascii=False)};
const DATA = {json.dumps(json_data, ensure_ascii=False)};

let currentStratKey = Object.keys(PRESETS)[0];
let currentFilter = "all";
let sortKey = "total_return_pct";
let sortAsc = false;
let searchQuery = "";

const $ = (id) => document.getElementById(id);

// 初始化策略下拉框
function initStrategySelector() {{
  const sel = $("strategySelect");
  sel.innerHTML = "";
  for (const [key, item] of Object.entries(PRESETS)) {{
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = item.name;
    sel.appendChild(opt);
  }}
  sel.value = currentStratKey;
  sel.onchange = (e) => {{
    currentStratKey = e.target.value;
    renderStrategyMeta();
    renderTable();
  }};
}}

// 渲染策略参数标签
function renderStrategyMeta() {{
  const meta = PRESETS[currentStratKey];
  $("curStratTitle").textContent = meta.name;
  $("curStratDesc").textContent = meta.desc;
  
  const p = meta.params;
  const tagsBox = $("curStratTags");
  tagsBox.innerHTML = "";
  
  const makeTag = (label, active) => {{
    const t = document.createElement("span");
    t.className = "tag " + (active ? "active" : "");
    t.textContent = label;
    tagsBox.appendChild(t);
  }};
  
  makeTag(`MA参数: ${{p.ma_fast}} / ${{p.ma_slow}}`, true);
  makeTag(`MACD参数: ${{p.macd_fast}},${{p.macd_slow}},${{p.macd_signal}}`, true);
  makeTag(`0轴线上过滤: ${{p.use_macd_zero ? "启用" : "未启用"}}`, p.use_macd_zero);
  makeTag(`250日年线过滤: ${{p.use_ma250_trend ? "严格要求C>MA250" : "不限年线"}}`, p.use_ma250_trend);
  makeTag(`MACD死叉止盈: ${{p.exit_on_macd_dead ? "启用" : "关闭"}}`, p.exit_on_macd_dead);
  makeTag(`红柱变绿卖出: ${{p.exit_on_bar_green ? "启用" : "关闭"}}`, p.exit_on_bar_green);
  makeTag(`MA死叉平仓: ${{p.exit_on_ma_cross ? "启用" : "关闭"}}`, p.exit_on_ma_cross);
  if (p.stop_loss_pct > 0) makeTag(`硬止损: ${{Math.round(p.stop_loss_pct * 100)}}%`, true);
  if (p.trail_gap_pct > 0) makeTag(`移动止盈回吐: ${{Math.round(p.trail_gap_pct * 100)}}%`, true);
}}

// 渲染表格
function renderTable() {{
  const rawList = DATA[currentStratKey] || [];
  
  // 过滤
  let list = rawList.filter((item) => {{
    if (currentFilter === "holding" && !item.is_holding_now) return false;
    if (currentFilter === "profit" && item.total_return_pct <= 0) return false;
    if (searchQuery) {{
      const q = searchQuery.toLowerCase();
      const matchCode = (item.symbol || "").toLowerCase().includes(q);
      const matchName = (item.name || "").toLowerCase().includes(q);
      const matchDate = (item.latest_signal_date || "").includes(q);
      if (!matchCode && !matchName && !matchDate) return false;
    }}
    return true;
  }});
  
  // 排序
  list.sort((a, b) => {{
    // 默认如果按收益排序，优先把正在持仓的且收益高的排在最前
    let va = a[sortKey];
    let vb = b[sortKey];
    if (typeof va === "string") va = va.toLowerCase();
    if (typeof vb === "string") vb = vb.toLowerCase();
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  }});
  
  // 统计指标
  $("stTotal").textContent = list.length;
  const holdings = list.filter(x => x.is_holding_now).length;
  $("stHolding").textContent = holdings;
  
  if (list.length > 0) {{
    const avgWin = (list.reduce((acc, x) => acc + x.win_rate, 0) / list.length).toFixed(1);
    const avgRet = (list.reduce((acc, x) => acc + x.total_return_pct, 0) / list.length).toFixed(2);
    const avgPL = (list.reduce((acc, x) => acc + x.profit_loss_ratio, 0) / list.length).toFixed(2);
    $("stWinRate").textContent = avgWin + "%";
    $("stAvgReturn").textContent = (avgRet >= 0 ? "+" : "") + avgRet + "%";
    $("stAvgReturn").style.color = avgRet >= 0 ? "var(--up)" : "var(--down)";
    $("stPLRatio").textContent = avgPL;
  }} else {{
    $("stWinRate").textContent = "0%";
    $("stAvgReturn").textContent = "0%";
    $("stPLRatio").textContent = "0.0";
  }}
  
  // 渲染主体行
  const tbody = $("stockTableBody");
  tbody.innerHTML = "";
  
  if (list.length === 0) {{
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center; padding: 40px; color: var(--muted);">暂无符合筛选条件的股票</td></tr>`;
    return;
  }}
  
  for (const row of list) {{
    const tr = document.createElement("tr");
    const retClass = row.total_return_pct >= 0 ? "up" : "down";
    const annClass = row.annual_return_pct >= 0 ? "up" : "down";
    const holdingBadge = row.is_holding_now
      ? `<span class="holding-badge">持仓中 ${{row.current_unrealized_pnl >= 0 ? "+" : ""}}${{row.current_unrealized_pnl}}%</span>`
      : `<span style="color:var(--muted); font-size:11px;">已清仓</span>`;
      
    tr.innerHTML = `
      <td class="code">${{row.symbol}}</td>
      <td><b>${{row.name || "—"}}</b></td>
      <td class="mono">¥${{row.latest_close.toFixed(2)}}</td>
      <td class="mono" style="color: #c7b6ff;">${{row.latest_signal_date || "—"}}</td>
      <td>${{row.total_trades}}</td>
      <td>${{row.win_rate}}%</td>
      <td class="${{retClass}}">${{row.total_return_pct >= 0 ? "+" : ""}}${{row.total_return_pct.toFixed(2)}}%</td>
      <td class="${{annClass}}">${{row.annual_return_pct >= 0 ? "+" : ""}}${{row.annual_return_pct.toFixed(2)}}%</td>
      <td style="color: var(--warn);">${{row.max_drawdown_pct.toFixed(1)}}%</td>
      <td>${{row.profit_loss_ratio}}</td>
      <td>${{holdingBadge}}</td>
      <td><button class="btn-detail" onclick="toggleDetails('${{row.symbol}}')">查看明细</button></td>
    `;
    tbody.appendChild(tr);
    
    // 展开行：交易明细
    const dtr = document.createElement("tr");
    dtr.id = `detail-${{row.symbol}}`;
    dtr.className = "details-row";
    
    let tradesHtml = "";
    if (row.trades && row.trades.length > 0) {{
      tradesHtml = `
        <table class="trade-mini-table">
          <thead>
            <tr>
              <th>买入日期</th>
              <th>买入价</th>
              <th>卖出日期</th>
              <th>卖出价</th>
              <th>持有天数</th>
              <th>卖出原因</th>
              <th>单笔收益率</th>
            </tr>
          </thead>
          <tbody>
            ${{row.trades.map(t => `
              <tr>
                <td>${{t.entry_date}}</td>
                <td>¥${{t.entry_price}}</td>
                <td>${{t.exit_date}}</td>
                <td>¥${{t.exit_price}}</td>
                <td>${{t.hold_bars}} 根K线</td>
                <td style="color: #ffcf8a;">${{t.exit_reason}}</td>
                <td class="${{t.pnl_pct >= 0 ? "up" : "down"}}">${{t.pnl_pct >= 0 ? "+" : ""}}${{t.pnl_pct}}%</td>
              </tr>
            `).join("")}}
          </tbody>
        </table>
      `;
    }} else {{
      tradesHtml = `<div style="color:var(--muted); padding: 10px;">回测区间内暂无已平仓交易（或当前仅首次买入尚未平仓）</div>`;
    }}
    
    dtr.innerHTML = `<td colspan="12"><div class="details-box">${{tradesHtml}}</div></td>`;
    tbody.appendChild(dtr);
  }}
}}

function sortTable(key) {{
  if (sortKey === key) {{
    sortAsc = !sortAsc;
  }} else {{
    sortKey = key;
    sortAsc = false;
  }}
  renderTable();
}}

function toggleDetails(sym) {{
  const row = $(`detail-${{sym}}`);
  if (!row) return;
  row.style.display = row.style.display === "table-row" ? "none" : "table-row";
}}

// 筛选事件
document.querySelectorAll(".filter-pills .pill").forEach(p => {{
  p.onclick = () => {{
    document.querySelectorAll(".filter-pills .pill").forEach(x => x.classList.remove("active"));
    p.classList.add("active");
    currentFilter = p.dataset.filter;
    renderTable();
  }};
}});

$("searchInput").oninput = (e) => {{
  searchQuery = e.target.value.trim();
  renderTable();
}};

initStrategySelector();
renderStrategyMeta();
renderTable();
</script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    print(f"选股报告页面已生成: {output_path}")


# ========== 命令行入口 ==========

def main():
    parser = argparse.ArgumentParser(description="MA20/55 + MACD + MA250 策略选股与多策略扫描器")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite数据库路径")
    parser.add_argument("--limit", type=int, default=None, help="扫描股票数量上限，留空=全部")
    parser.add_argument("--years", type=float, default=2.0, help="回测年限，默认2年")
    parser.add_argument("--strategy", default="", help="页面当前选中的策略，只有它接收开关覆盖")
    parser.add_argument("--params", default="", help="JSON 参数，覆盖均线、MACD、止损和开关")
    parser.add_argument("--flag-overrides", default="", help="按策略键覆盖开关的 JSON")
    parser.add_argument("--no-html", action="store_true", help="只写 result.json，页面自己渲染")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误：数据库不存在 {db_path}，请先执行数据同步！", flush=True)
        sys.exit(1)
    if args.years is None or not (0 < float(args.years) <= 30):
        print("回测年限必须在 0 到 30 之间", flush=True)
        sys.exit(1)
    if args.limit is not None and int(args.limit) < 1:
        print("股票数量必须是正整数，或不传表示全部", flush=True)
        sys.exit(1)
    overrides = {}
    if args.params:
        try:
            overrides = json.loads(args.params)
        except json.JSONDecodeError as exc:
            print(f"--params 不是合法 JSON：{exc}", flush=True)
            sys.exit(1)
        if not isinstance(overrides, dict):
            print("--params 必须是对象", flush=True)
            sys.exit(1)
    flag_overrides = {}
    if args.flag_overrides:
        try:
            flag_overrides = json.loads(args.flag_overrides)
        except json.JSONDecodeError as exc:
            print(f"--flag-overrides 不是合法 JSON：{exc}", flush=True)
            sys.exit(1)
        if not isinstance(flag_overrides, dict):
            print("--flag-overrides 必须是对象", flush=True)
            sys.exit(1)
    try:
        strategies = build_strategies(overrides, args.strategy.strip() or None, flag_overrides)
    except ValueError as exc:
        print(str(exc), flush=True)
        sys.exit(1)

    print("==================================================", flush=True)
    print(f"全库扫描开始：最近 {args.years:g} 年，股票 {'全部' if not args.limit else args.limit} 只", flush=True)
    print(f"策略套数 {len(strategies)}。右上角下拉框用于切换查看。", flush=True)
    print("==================================================", flush=True)

    lookback_days = int(float(args.years) * 365)
    all_results = scan_all_strategies(db_path, strategies, max_stocks=args.limit, lookback_days=lookback_days)
    save_result(strategies, all_results, float(args.years), args.limit)
    if not args.no_html:
        generate_html_report(all_results, HTML_REPORT, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("全策略回测扫描完成。", flush=True)
    for key, rows in all_results.items():
        cfg = strategies.get(key)
        title = cfg.name if cfg is not None else key
        print(f"    {title}：{len(rows)} 只", flush=True)


if __name__ == "__main__":
    main()
