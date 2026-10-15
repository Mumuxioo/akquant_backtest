"""本地网页控制台：数据同步 + 五线探戈 / 粘合突破回测 + 打开报告。只用标准库，只监听本机。

用法（在项目根目录）：
    python scripts/web_ui.py            # 启动后自动打开 http://127.0.0.1:8765
    python scripts/web_ui.py --no-open

五线探戈是 / ，粘合突破是 /bond.html。页面也可以直接双击用 file:// 打开，它会连本机这个服务。
浏览器不能直接执行 Python，所以同步和回测都由这里起子进程，把输出逐行流回页面。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, resolve_path, section

WEB_DIR = PROJECT_ROOT / "web"
SCRIPTS = PROJECT_ROOT / "scripts"
SYMBOL_RE = re.compile(r"^\d{6}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EXIT_MARK = "__EXIT__"     # 子进程结束后追加的最后一行，页面据此判断成功与否
MIME = {
    ".html": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def load_script(filename: str, module_name: str):
    """按文件路径加载策略脚本，只为拿到默认可调参数，不在本进程里跑回测。"""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"找不到 scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 在执行类定义时会回查 sys.modules，必须先登记。
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Task:
    """同一时间只跑一个子进程：同步会长时间写库，和回测并发读写没有意义。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.kind = ""

    def start(self, kind: str, argv: list[str]) -> subprocess.Popen | None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return None
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            extra = {}
            if os.name == "nt":
                # Windows 没有 SIGINT 可发给子进程，单独进程组后才能发 CTRL_BREAK。
                extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(
                [sys.executable, "-u", *argv],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                **extra,
            )
            self.kind = kind
            return self.proc

    def stop(self) -> bool:
        """发中断信号，让同步脚本走自己的回滚逻辑，而不是直接杀掉留下半截事务。"""
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
            return True

    def busy(self) -> str:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return self.kind
            return ""


class App:
    """启动时读一次配置；每次请求都用这份配置拼命令行。"""

    def __init__(self) -> None:
        self.cfg = load_config()
        self.five_line = load_script("five_line.py", "five_line_ui")
        self.five_line.configure(self.cfg)
        self.bond = load_script("bond_break.py", "bond_break_ui")
        self.bond.configure(self.cfg)
        self.ma = load_script("ma_cross_strategy.py", "ma_scan_ui")
        self.output_root = resolve_path(section(self.cfg, "output").get("dir") or "output")
        self.task = Task()

    def defaults(self) -> dict:
        backtest = section(self.cfg, "backtest")
        return {
            "params": self.five_line.current_params(),
            "symbol": str(backtest.get("default_symbol") or self.five_line.DEFAULT_SYMBOL),
            "start_date": str(backtest.get("start_date") or ""),
            "end_date": str(backtest.get("end_date") or ""),
            "db_path": str(self.five_line.DB_PATH),
            "db_exists": Path(self.five_line.DB_PATH).exists(),
            "busy": self.task.busy(),
        }

    def sync_argv(self) -> list[str]:
        data = section(self.cfg, "data")
        sync = section(self.cfg, "sync")
        argv = [str(SCRIPTS / "sync_a_share_daily.py")]
        if data.get("db_path"):
            argv += ["--db", str(data["db_path"])]
        for key, flag in (
            ("start_date", "--start-date"),
            ("overlap_days", "--overlap-days"),
            ("workers", "--workers"),
            ("delay", "--delay"),
        ):
            if sync.get(key) is not None and str(sync.get(key)).strip():
                argv += [flag, str(sync[key])]
        return argv

    def backtest_argv(self, body: dict) -> list[str]:
        """网页输入是不可信的：代码、日期、参数都在这里校验，非法直接拒绝。"""
        symbol = str(body.get("symbol") or "").strip()
        if not SYMBOL_RE.match(symbol):
            raise ValueError("股票代码必须是 6 位数字")
        argv = [str(SCRIPTS / "five_line.py"), "backtest", symbol, "--stream"]
        for key, flag in (("start_date", "--start"), ("end_date", "--end")):
            value = str(body.get(key) or "").strip()
            if value and not DATE_RE.match(value):
                raise ValueError(f"{key} 格式必须是 YYYY-MM-DD")
            argv += [flag, value]
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params 必须是对象")
        allowed = {key.lower() for key in self.five_line.TUNABLE}
        clean = {}
        for key, value in params.items():
            name = str(key or "").lower()
            if name not in allowed:
                raise ValueError(f"未知参数 {key}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"参数 {key} 必须是数字")
            clean[name] = value
        if clean:
            argv += ["--params", json.dumps(clean)]
        return argv

    def bond_defaults(self) -> dict:
        backtest = section(self.cfg, "backtest")
        return {
            "params": self.bond.current_params(),
            "exit_mode": self.bond.EXIT_MODE,
            "high_field": self.bond.HIGH_FIELD,
            "symbol": str(backtest.get("default_symbol") or self.bond.DEFAULT_SYMBOL),
            "start_date": str(backtest.get("start_date") or ""),
            "end_date": str(backtest.get("end_date") or ""),
            "db_path": str(self.bond.DB_PATH),
            "db_exists": Path(self.bond.DB_PATH).exists(),
            "busy": self.task.busy(),
        }

    def bond_backtest_argv(self, body: dict) -> list[str]:
        """和五线探戈同一套校验：代码、日期、数字参数不合法就拒绝，不交给子进程猜。"""
        if body is None:
            body = {}
        symbol = str(body.get("symbol") or "").strip()
        if not SYMBOL_RE.match(symbol):
            raise ValueError("股票代码必须是 6 位数字")
        mode = str(body.get("exit_mode") or self.bond.EXIT_MODE).strip().lower()
        field = str(body.get("high_field") or self.bond.HIGH_FIELD).strip().lower()
        if mode not in ("macd", "rr"):
            raise ValueError("止盈方案只能是 macd 或 rr")
        if field not in ("close", "high"):
            raise ValueError("高位口径只能是 close 或 high")
        argv = [
            str(SCRIPTS / "bond_break.py"), "backtest", symbol, "--stream",
            "--exit", mode, "--high", field,
        ]
        for key, flag in (("start_date", "--start"), ("end_date", "--end")):
            value = str(body.get(key) or "").strip()
            if value and not DATE_RE.match(value):
                raise ValueError(f"{key} 格式必须是 YYYY-MM-DD")
            argv += [flag, value]
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params 必须是对象")
        allowed = {key.lower() for key in self.bond.TUNABLE}
        clean = {}
        for key, value in params.items():
            name = str(key or "").lower()
            if name not in allowed:
                raise ValueError(f"未知参数 {key}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"参数 {key} 必须是数字")
            if value < 0:
                raise ValueError(f"参数 {key} 不能为负")
            clean[name] = value
        if clean:
            argv += ["--params", json.dumps(clean)]
        return argv

    def bond_reports(self, symbol: str) -> dict:
        if not SYMBOL_RE.match(symbol or ""):
            return {}
        folder = self.output_root / symbol
        names = (self.bond.REPORT_NAME, self.bond.REVIEW_NAME, self.bond.LOG_NAME)
        return {
            name: f"/output/{symbol}/{name}"
            for name in names
            if name and (folder / name).is_file()
        }

    def ma_defaults(self) -> dict:
        scan = section(self.cfg, "ma_cross_scan")
        presets = []
        for key, cfg in self.ma.PRESET_STRATEGIES.items():
            if cfg is None:
                continue
            presets.append({"key": key, "name": cfg.name, "desc": cfg.desc, "params": self.ma.asdict(cfg)})
        return {
            "years": float(scan.get("lookback_years") or 2),
            "limit": None,
            "db_exists": Path(self.ma.DB_PATH).exists(),
            "db_path": str(self.ma.DB_PATH),
            "busy": self.task.busy(),
            "presets": presets,
        }

    def ma_result(self) -> dict:
        path = self.output_root / "ma20_55_scanner" / "result.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def ma_scan_argv(self, body: dict) -> list[str]:
        """页面上的年限、数量和指标都在这里验过，再交给扫描脚本。"""
        if body is None:
            body = {}
        years = body.get("years", 2)
        if isinstance(years, bool) or not isinstance(years, (int, float)) or not (0 < float(years) <= 30):
            raise ValueError("回测年限必须在 0 到 30 之间")
        raw_limit = body.get("limit")
        limit = None
        if raw_limit not in (None, ""):
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)):
                raise ValueError("股票数量必须是正整数，留空表示全部")
            if float(raw_limit) != int(raw_limit) or int(raw_limit) < 1:
                raise ValueError("股票数量必须是正整数，留空表示全部")
            limit = int(raw_limit)
        strategy = str(body.get("strategy") or "").strip()
        if strategy and strategy not in self.ma.PRESET_STRATEGIES:
            raise ValueError("未知策略")
        params = body.get("params") or {}
        flags = body.get("flag_overrides") or {}
        if not isinstance(params, dict) or not isinstance(flags, dict):
            raise ValueError("参数必须是对象")
        clean = self._clean_ma_params(params, allow_flags=True)
        clean_flags = {}
        for key, value in flags.items():
            name = str(key or "").strip()
            if name not in self.ma.PRESET_STRATEGIES:
                raise ValueError(f"未知策略 {key}")
            if not isinstance(value, dict):
                raise ValueError(f"{key} 的开关必须是对象")
            clean_flags[name] = self._clean_ma_params(value, allow_flags=True)
        argv = [str(SCRIPTS / "ma_cross_strategy.py"), "--years", str(float(years)), "--no-html"]
        if limit is not None:
            argv += ["--limit", str(limit)]
        if strategy:
            argv += ["--strategy", strategy]
        if clean:
            argv += ["--params", json.dumps(clean, ensure_ascii=False)]
        if clean_flags:
            argv += ["--flag-overrides", json.dumps(clean_flags, ensure_ascii=False)]
        return argv

    def _clean_ma_params(self, params: dict, allow_flags: bool) -> dict:
        if params is None:
            return {}
        allowed = set(self.ma.INT_FIELDS) | set(self.ma.FLOAT_FIELDS)
        if allow_flags:
            allowed |= set(self.ma.BOOL_FIELDS)
        clean = {}
        for key, value in params.items():
            name = str(key or "").strip()
            if name not in allowed:
                raise ValueError(f"未知参数 {key}")
            if isinstance(value, bool):
                if name not in self.ma.BOOL_FIELDS:
                    raise ValueError(f"参数 {key} 必须是数字")
                clean[name] = 1 if value else 0
                continue
            if value is None or not isinstance(value, (int, float)):
                raise ValueError(f"参数 {key} 必须是数字")
            if name in self.ma.BOOL_FIELDS and value not in (0, 1):
                raise ValueError(f"参数 {key} 只能是 0 或 1")
            if value < 0:
                raise ValueError(f"参数 {key} 不能为负")
            clean[name] = value
        return clean

    def reports(self, symbol: str) -> dict:
        """某只股票已有哪些产物，页面据此点亮「打开报告」按钮。"""
        if not SYMBOL_RE.match(symbol or ""):
            return {}
        folder = self.output_root / symbol
        return {
            name: f"/output/{symbol}/{name}"
            for name in ("report.html", "review.html", "five_line_log.txt")
            if (folder / name).is_file()
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "QuantDemo/1.0"

        def log_request(self, code="-", size="-"):  # 安静一点，只打印出错的请求
            if str(getattr(code, "value", code)).startswith(("4", "5")):
                super().log_request(code, size)

        def _host_ok(self) -> bool:
            """Host 必须是本机名，挡住 DNS rebinding（外部域名解析到 127.0.0.1 来调接口）。"""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("127.0.0.1", "localhost", "::1")

        # ---- 跨域：允许 file:// 打开的页面（Origin: null）和本服务自己的页面 ----
        def _origin_ok(self) -> bool:
            if not self._host_ok():
                return False
            origin = self.headers.get("Origin")
            if origin is None or origin == "null":
                return True
            host = self.headers.get("Host") or ""
            return origin in (f"http://{host}",)

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin == "null":
                self.send_header("Access-Control-Allow-Origin", "null")

        def _json(self, data: dict, status: int = 200) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self._cors()
            self.end_headers()
            self.wfile.write(raw)

        def _file(self, path: Path) -> None:
            if path is None or not path.is_file():
                self._json({"error": "文件不存在"}, 404)
                return
            raw = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 64 * 1024:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        def _stream(self, kind: str, argv: list[str]) -> None:
            """起子进程，把输出原样逐行写回。客户端断开后子进程继续跑完，仍可点「停止」。"""
            proc = app.task.start(kind, argv)
            if proc is None:
                self._json({"error": f"已有任务在运行：{app.task.busy()}"}, 409)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            alive = True
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, b""):
                    if not alive:
                        continue
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        alive = False
            code = proc.wait()
            if alive:
                try:
                    self.wfile.write(f"\n{EXIT_MARK} {code}\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            self.close_connection = True

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if not self._host_ok():
                self._json({"error": "Host 不被允许"}, 403)
                return
            url = urlparse(self.path)
            path = unquote(url.path)
            if path in ("/", "/index.html"):
                self._file(WEB_DIR / "index.html")
            elif path in ("/bond", "/bond.html"):
                self._file(WEB_DIR / "bond.html")
            elif path in ("/scan", "/scan.html"):
                self._file(WEB_DIR / "scan.html")
            elif path == "/api/ma/config":
                self._json(app.ma_defaults())
            elif path == "/api/ma/result":
                self._json(app.ma_result())
            elif path == "/api/config":
                self._json(app.defaults())
            elif path == "/api/bond/config":
                self._json(app.bond_defaults())
            elif path == "/api/bond/reports":
                symbol = (parse_qs(url.query).get("symbol") or [""])[0]
                self._json(app.bond_reports(symbol))
            elif path == "/api/reports":
                symbol = (parse_qs(url.query).get("symbol") or [""])[0]
                self._json(app.reports(symbol))
            elif path.startswith("/output/"):
                # 只允许读 output 目录内的文件，防止 ../ 越界。
                root = app.output_root.resolve()
                target = (root / path[len("/output/"):]).resolve()
                if root != target and root not in target.parents:
                    self._json({"error": "禁止访问"}, 403)
                    return
                self._file(target)
            else:
                self._json({"error": "未找到"}, 404)

        def do_POST(self):
            if not self._origin_ok():
                self._json({"error": "来源不被允许"}, 403)
                return
            path = urlparse(self.path).path
            if path == "/api/sync":
                self._stream("数据同步", app.sync_argv())
            elif path == "/api/backtest":
                try:
                    argv = app.backtest_argv(self._body())
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._stream("回测", argv)
            elif path == "/api/ma/scan":
                try:
                    argv = app.ma_scan_argv(self._body())
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._stream("全库扫描", argv)
            elif path == "/api/bond/backtest":
                try:
                    argv = app.bond_backtest_argv(self._body())
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._stream("粘合突破回测", argv)
            elif path == "/api/stop":
                self._json({"stopped": app.task.stop()})
            else:
                self._json({"error": "未找到"}, 404)

    return Handler


def main() -> None:
    cfg = load_config()
    web = section(cfg, "web")
    parser = argparse.ArgumentParser(description="五线探戈本地网页控制台")
    parser.add_argument("--host", default=str(web.get("host") or "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(web.get("port") or 8765))
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    print("加载策略与配置……")
    app = App()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(f"控制台已启动：{url}  （Ctrl+C 退出）")
    print(f"粘合突破：{url}bond.html")
    print(f"20/55选股：{url}scan.html")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出……")
        app.task.stop()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
