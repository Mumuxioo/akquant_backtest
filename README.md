# quant-demo · 个人量化交易

A 股本地日线 + 五线探戈策略回测，带本地网页控制台。

## 目录

```
config/config.yaml        全部默认配置（数据库路径、同步参数、策略参数、端口）
scripts/web_ui.py         本地网页控制台服务（仅标准库，只监听本机）
scripts/five_line.py      五线探戈：单股回测 / 大盘选股
scripts/sync_a_share_daily.py  全 A 股日线增量同步
src/data/a_share.py       数据同步与 SQLite 存储
src/utils/                配置读取、日志
web/index.html            控制台页面（单文件、离线可用）
data/a_share.db           本地行情库（不入库，可同步生成）
output/<代码>/            回测产物：report.html / review.html / five_line_log.txt
```

## 使用

```bash
pip install -r requirements.txt
python scripts/web_ui.py          # 自动打开 http://127.0.0.1:8765
```

页面里可以：同步数据、可视化调整策略参数、输入股票代码回测并逐行查看每天的操作日志、打开回测报告和交易复盘。
`web/index.html` 也可以直接双击打开，前提是 `web_ui.py` 已在运行。

迁移到其他电脑：拷贝整个项目目录（含 `data/a_share.db`，或到新电脑上点「同步」重新生成），装好依赖后运行上面的命令即可。

命令行同样可用：

```bash
python scripts/five_line.py backtest 600519 --start 2020-01-01 --params '{"rvol_high": 1.5}'
python scripts/five_line.py selfcheck
python scripts/sync_a_share_daily.py
```
