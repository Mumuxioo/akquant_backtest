"""每个工作日增量同步全 A 股前复权日线。

直接运行会立刻增量同步一次。安装后由 macOS launchd 在登录状态下的工作日定时触发。
电脑在定时点休眠时，launchd 不会补跑，醒来后执行一次本脚本即可。

用法：
    python scripts/sync_a_share_daily.py
    python scripts/sync_a_share_daily.py --install-launchd --at 17:30
    python scripts/sync_a_share_daily.py --uninstall-launchd
"""

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a_share import DEFAULT_SYNC_WORKERS, SyncInterrupted, sync_all

LABEL = "com.quant-demo.a-share-sync"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量同步全 A 股前复权日线，并可安装工作日定时任务")
    parser.add_argument("--db", default="data/a_share.db", help="SQLite 文件路径")
    parser.add_argument("--start-date", default="19900101", help="新股补全历史的起点，格式 YYYYMMDD")
    parser.add_argument("--overlap-days", type=int, default=30, help="已有股票回补的自然日数")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_SYNC_WORKERS,
        help="并发请求数。腾讯日线 16 稳定；东财超过 4 会断连，仅北交所和失败回退会打到东财",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="仅 workers=1 时，每只股票请求后的间隔秒数",
    )
    parser.add_argument("--limit", type=int, help="仅同步前 N 只，用于试跑；不会写入定时任务")
    parser.add_argument("--at", default="17:30", help="定时时刻 HH:MM，仅配合 --install-launchd")
    parser.add_argument("--install-launchd", action="store_true", help="安装工作日定时任务")
    parser.add_argument("--uninstall-launchd", action="store_true", help="卸载工作日定时任务")
    return parser.parse_args()


def _parse_clock(value: str) -> tuple[int, int]:
    if value is None or not str(value).strip() or ":" not in str(value):
        raise ValueError(f"时刻格式必须是 HH:MM，实际为 {value!r}")
    hour_text, minute_text = str(value).strip().split(":", 1)
    if not hour_text.isdigit() or not minute_text.isdigit():
        raise ValueError(f"时刻格式必须是 HH:MM，实际为 {value!r}")
    hour, minute = int(hour_text), int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"时刻超出范围：{value}")
    return hour, minute


def _agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _db_path(raw: str) -> Path:
    if raw is None or not str(raw).strip():
        raise ValueError("db 路径不能为空")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def install_launchd(args: argparse.Namespace) -> Path:
    """写入并加载 launchd 配置。只覆盖本任务，不改其他代理。"""
    if not sys.executable:
        raise RuntimeError("找不到当前 Python 解释器")
    hour, minute = _parse_clock(args.at)
    script = Path(__file__).resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    db = _db_path(args.db)
    log_dir = PROJECT_ROOT / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Weekday 1–5 是周一到周五。定时任务固定走增量，不带 --limit。
    spec = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            str(script),
            "--db",
            str(db),
            "--start-date",
            args.start_date,
            "--overlap-days",
            str(args.overlap_days),
            "--workers",
            str(args.workers),
            "--delay",
            str(args.delay),
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": [
            {"Weekday": day, "Hour": hour, "Minute": minute} for day in range(1, 6)
        ],
        "StandardOutPath": str(log_dir / "a_share_sync.log"),
        "StandardErrorPath": str(log_dir / "a_share_sync.err"),
    }
    path = _agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(spec))

    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(path)], check=False)
    subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
    return path


def uninstall_launchd() -> None:
    path = _agent_path()
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(path)], check=False)
    if path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    if args.workers is None or args.workers < 1:
        raise SystemExit("workers 必须是大于 0 的整数")
    if args.install_launchd and args.uninstall_launchd:
        raise SystemExit("不能同时安装和卸载定时任务")
    if args.uninstall_launchd:
        uninstall_launchd()
        print("已卸载工作日增量同步")
        return
    if args.install_launchd:
        path = install_launchd(args)
        print(f"已安装：每个工作日 {args.at} 增量同步。配置：{path}")
        return

    try:
        result = sync_all(
            db_path=_db_path(args.db),
            start_date=args.start_date,
            delay=args.delay,
            limit=args.limit,
            incremental=True,
            overlap_days=args.overlap_days,
            workers=args.workers,
        )
    except SyncInterrupted as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"增量同步完成：股票 {result.total} 只，成功 {result.succeeded}，"
        f"失败 {result.failed}，写入 {result.rows} 行；数据库：{args.db}"
    )
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
