"""Configuration handling for ezcal.

Precedence (low -> high):

1. packaged defaults (``ezcal/data/default_config.yaml``)
2. ``~/.config/ezcal/qe_config.yaml``
3. ``./qe_config.yaml`` in the current working directory
4. a file given with ``--config``
5. explicit CLI options

Everything is kept as plain nested dicts so that a user can add engine
specific keys (``vasp:``, ``mlip:`` ...) without ezcal having to know
about them in advance.
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
    """Recursively merge ``override`` into ``base`` (``base`` is not mutated)."""
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
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return data


@dataclass
class Config:
    """A merged ezcal configuration.

    Access nested values with dotted paths::

        cfg.get("dft.ecutwfc")
        cfg.set("run.nproc", 8)
    """

    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    # -- access ---------------------------------------------------------
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
        """Apply ``{"dft.ecutwfc": 60, ...}`` style overrides, skipping ``None``."""
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

    # -- convenience for path-like values --------------------------------
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
    """Build the effective configuration."""
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
    """Turn ``["dft.ecutwfc=60", "run.nproc=4"]`` into dotted overrides."""
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
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
