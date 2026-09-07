"""ezcal の設定処理。

優先順位 (低い -> 高い):

1. パッケージ同梱の既定値 (``ezcal/data/default_config.yaml``)
2. ``~/.config/ezcal/qe_config.yaml``
3. カレントディレクトリの ``./qe_config.yaml``
4. ``--config`` で指定したファイル
5. CLI で明示したオプション

設定はすべて素の入れ子辞書として保持する。こうしておけば、ezcal 側が事前に
知らないエンジン固有のキー (``vasp:``、``mlip:`` など) をユーザーが自由に
追加できる。
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

DEFAULT_CONFIG_NAME = "qe_config.yaml"
USER_CONFIG_PATH = Path.home() / ".config" / "ezcal" / DEFAULT_CONFIG_NAME
PACKAGE_DEFAULT = Path(__file__).with_name("data") / "default_config.yaml"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """``override`` を ``base`` に再帰的にマージする (``base`` は変更しない)。"""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 設定ファイルの最上位はマッピングでなければなりません")
    return data


@dataclass
class Config:
    """マージ済みの ezcal 設定。

    入れ子の値にはドット区切りのパスでアクセスする::

        cfg.get("dft.ecutwfc")
        cfg.set("run.nproc", 8)
    """

    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    # -- アクセス --------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        value = self.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def update(self, override: Mapping[str, Any]) -> "Config":
        return Config(_deep_merge(self.data, override), list(self.sources))

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """``{"dft.ecutwfc": 60, ...}`` 形式の上書きを適用する (``None`` は無視)。"""
        for dotted, value in overrides.items():
            if value is None:
                continue
            self.set(dotted, value)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def save(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    # -- パス的な値を扱うための補助 ---------------------------------------
    def path(self, dotted: str, default: Any = None) -> Path | None:
        value = self.get(dotted, default)
        if value is None:
            return None
        return Path(os.path.expandvars(str(value))).expanduser()


def default_config() -> Config:
    return Config(_read_yaml(PACKAGE_DEFAULT), [str(PACKAGE_DEFAULT)])


def config_search_path(explicit: str | os.PathLike | None = None) -> list[Path]:
    candidates: list[Path] = [USER_CONFIG_PATH, Path.cwd() / DEFAULT_CONFIG_NAME]
    if explicit:
        candidates.append(Path(explicit).expanduser())
    return candidates


def load_config(
    explicit: str | os.PathLike | None = None,
    extra: Mapping[str, Any] | None = None,
    search: bool = True,
) -> Config:
    """実際に適用される設定を組み立てる。"""
    cfg = default_config()
    if search:
        for candidate in config_search_path(explicit):
            if candidate.is_file():
                cfg = Config(_deep_merge(cfg.data, _read_yaml(candidate)),
                             cfg.sources + [str(candidate)])
    elif explicit:
        path = Path(explicit).expanduser()
        cfg = Config(_deep_merge(cfg.data, _read_yaml(path)), cfg.sources + [str(path)])
    if extra:
        cfg = cfg.update(extra)
    return cfg


def parse_set_options(pairs: Sequence[str] | None) -> dict:
    """``["dft.ecutwfc=60", "run.nproc=4"]`` をドット区切りの上書き辞書に変換する。"""
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set は key=value 形式で指定してください: {item!r}")
        key, _, raw = item.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def flatten(data: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            yield from flatten(value, prefix=f"{dotted}.")
        else:
            yield dotted, value
