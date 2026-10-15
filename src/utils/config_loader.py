"""配置加载：全项目唯一的配置读取入口。"""

from pathlib import Path

import yaml

# 项目根目录（src/utils/config_loader.py 往上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """读取 config.yaml，返回配置字典。文件不存在或内容为空时返回空字典，调用方走代码默认值。"""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def section(cfg: dict | None, name: str) -> dict:
    """取配置里的一节，缺失或类型不对都当空字典。"""
    if not cfg or not name:
        return {}
    value = cfg.get(name)
    return value if isinstance(value, dict) else {}


def resolve_path(rel: str | Path) -> Path:
    """把配置里的相对路径解析为项目根下的绝对路径；绝对路径原样返回。"""
    if rel is None or not str(rel).strip():
        raise ValueError("路径不能为空")
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path
