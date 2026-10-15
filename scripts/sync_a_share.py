"""把全 A 股前复权日线同步到本地 SQLite。

用法：
    python scripts/sync_a_share.py
    python scripts/sync_a_share.py --start-date 20200101 --limit 10
"""

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a_share import DEFAULT_SYNC_WORKERS, SyncInterrupted, sync_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步全 A 股前复权日线到 SQLite")
    parser.add_argument("--db", default="data/a_share.db", help="SQLite 文件路径")
    parser.add_argument("--start-date", default="19900101", help="开始日期，格式 YYYYMMDD")
    parser.add_argument(
        "--end-date",
        default=date.today().strftime("%Y%m%d"),
        help="结束日期，格式 YYYYMMDD",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_SYNC_WORKERS,
        help="并发请求数。腾讯日线 16 稳定；东财超过 4 会断连",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="仅 workers=1 时，每只股票请求后的间隔秒数",
    )
    parser.add_argument("--limit", type=int, help="仅同步前 N 只，用于试跑")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers is None or args.workers < 1:
        raise SystemExit("workers 必须是大于 0 的整数")
    try:
        result = sync_all(
            db_path=args.db,
            start_date=args.start_date,
            end_date=args.end_date,
            delay=args.delay,
            limit=args.limit,
            workers=args.workers,
        )
    except SyncInterrupted as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"同步完成：股票 {result.total} 只，成功 {result.succeeded}，"
        f"失败 {result.failed}，写入 {result.rows} 行；数据库：{args.db}"
    )
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
