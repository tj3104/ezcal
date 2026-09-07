"""ASE calculator の解決層。

ezcal が機械学習ポテンシャル (MLIP / NNP) を扱うために必要なのは
「ASE の calculator オブジェクトを 1 つ作れること」だけである。したがって
ここでは *どうやって calculator を作るか* だけを外から差し替えられるように
してある。エンジン (:mod:`ezcal.engines.mlip`) や MD ドライバ
(:mod:`ezcal.md`) は、この層が返した calculator をそのまま使う。

指定方法は 3 通りあり、優先順位は上から順に高い:

``mlip.script``
    ユーザーが書いた Python ファイル。``build(**options)`` (または
    ``get_calculator`` / ``calculator``) を定義しておけばよい。
    ポテンシャルごとに設定方法がまったく違っても、このファイルの中で
    好きに書ける。雛形は ``ezcal mlip template`` で取り出せる。
``mlip.factory``
    ``module:attribute`` 形式の import パス。既存のライブラリ関数を
    そのまま呼びたい場合に使う。例:
    ``mace.calculators:mace_mp``。
``mlip.backend``
    同梱レシピの名前 (``sevennet``、``mace``、``chgnet``、``emt`` ...)。
    レシピは ``recipes/`` に 1 ファイル 1 ポテンシャルで置いてあり、
    ``mlip.recipe_dirs`` に自分のディレクトリを足せば、同じ形式の
    ファイルを置くだけで ``--mlip-backend`` から選べるようになる。

いずれの場合も、``mlip.options`` の中身がそのまま ``build()`` の
キーワード引数になる。``mlip.model`` と ``mlip.device`` は日常的に
切り替えるため、``options`` に自動で流し込まれる (レシピがその引数を
受け付ける場合のみ)。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

BUILTIN_DIR = Path(__file__).with_name("recipes")

#: ``mlip.model`` / ``mlip.device`` を流し込む先の引数名
CONVENIENCE_KEYS = ("model", "device")


class CalculatorError(RuntimeError):
    pass


@dataclass
class Recipe:
    """1 つのポテンシャル (calculator) の作り方。"""

    name: str
    build: Callable[..., Any]
    description: str = ""
    aliases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()          # import できないと使えないモジュール
    install: str = ""                       # 導入コマンド (エラーメッセージ用)
    defaults: dict = field(default_factory=dict)
    models: tuple[str, ...] = ()            # よく使うモデル名 (参考表示用)
    source: Path | None = None

    # -- 利用可否 --------------------------------------------------------
    def missing(self) -> list[str]:
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def check(self) -> list[str]:
        missing = self.missing()
        if not missing:
            return []
        hint = self.install or f"uv pip install {missing[0]}"
        return [f"{self.name}: {', '.join(missing)} が見つかりません  ->  {hint}"]

    def accepts(self, key: str) -> bool:
        """``model`` / ``device`` を自動で渡してよいレシピか。

        ``build()`` の引数として明示されているか、``DEFAULTS`` に入っている
        ものだけを対象にする。``**kwargs`` しか無いレシピ (EMT のように
        モデルの概念が無いもの) に無関係な引数を押し込まないための判定。
        """
        if key in self.defaults:
            return True
        try:
            signature = inspect.signature(self.build)
        except (TypeError, ValueError):      # pragma: no cover - C 実装など
            return False
        return key in signature.parameters

    def __call__(self, **options):
        merged = {**self.defaults, **options}
        problems = self.check()
        if problems:
            raise CalculatorError(problems[0])
        return self.build(**merged)


# ------------------------------------------------------------------ 読み込み
def _module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"ezcal_calc_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise CalculatorError(f"{path} を Python モジュールとして読み込めません")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe_from_module(module, source: Path | None = None) -> Recipe:
    """モジュールの決まった名前の属性から :class:`Recipe` を組み立てる。"""
    build = None
    for attribute in ("build", "get_calculator", "make_calculator", "calculator"):
        candidate = getattr(module, attribute, None)
        if candidate is None:
            continue
        build = candidate if callable(candidate) else (lambda _c=candidate, **_: _c)
        break
    if build is None:
        raise CalculatorError(
            f"{source or module}: build(**options) を定義してください "
            "(get_calculator / make_calculator / calculator でも可)")
    name = str(getattr(module, "NAME", None)
               or (source.stem if source else module.__name__))
    return Recipe(
        name=name,
        build=build,
        description=str(getattr(module, "DESCRIPTION", "") or ""),
        aliases=tuple(getattr(module, "ALIASES", ()) or ()),
        requires=tuple(getattr(module, "REQUIRES", ()) or ()),
        install=str(getattr(module, "INSTALL", "") or ""),
        defaults=dict(getattr(module, "DEFAULTS", {}) or {}),
        models=tuple(getattr(module, "MODELS", ()) or ()),
        source=source,
    )


def load_recipe(path: str | os.PathLike) -> Recipe:
    """Python ファイル 1 つを読み込んでレシピにする。"""
    path = Path(os.path.expandvars(str(path))).expanduser()
    if not path.is_file():
        raise CalculatorError(f"calculator スクリプトが見つかりません: {path}")
    return recipe_from_module(_module_from_path(path), source=path)


def recipe_dirs(extra: Sequence[str] | None = None) -> list[Path]:
    dirs = [BUILTIN_DIR]
    for item in extra or ():
        path = Path(os.path.expandvars(str(item))).expanduser()
        if path.is_dir():
            dirs.append(path)
    return dirs


def discover(extra_dirs: Sequence[str] | None = None) -> dict[str, Recipe]:
    """名前 (と別名) -> レシピ。後から見つかったものが優先される。"""
    found: dict[str, Recipe] = {}
    for directory in recipe_dirs(extra_dirs):
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                recipe = load_recipe(path)
            except Exception:                # 壊れたファイルで一覧を壊さない
                continue
            for key in (recipe.name, *recipe.aliases):
                found[str(key).lower()] = recipe
    return found


def available(extra_dirs: Sequence[str] | None = None) -> list[Recipe]:
    """重複を除いたレシピ一覧 (名前順)。"""
    seen: dict[str, Recipe] = {}
    for recipe in discover(extra_dirs).values():
        seen.setdefault(recipe.name, recipe)
    return [seen[name] for name in sorted(seen)]


def get_recipe(name: str, extra_dirs: Sequence[str] | None = None) -> Recipe:
    table = discover(extra_dirs)
    key = str(name).lower()
    if key not in table:
        raise CalculatorError(
            f"未知の MLIP バックエンドです: {name!r}  "
            f"(利用できるのは {', '.join(r.name for r in available(extra_dirs))}。"
            "任意のポテンシャルは mlip.script / mlip.factory で指定できます)")
    return table[key]


def _factory_from_path(spec: str) -> Callable[..., Any]:
    """``"module:attr"`` または ``"module.attr"`` を呼び出せる形にする。"""
    text = str(spec).strip()
    if ":" in text:
        module_name, _, attribute = text.partition(":")
    elif "." in text:
        module_name, _, attribute = text.rpartition(".")
    else:
        raise CalculatorError(
            f"mlip.factory は 'module:attribute' 形式で指定してください: {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CalculatorError(f"{module_name} を import できません: {exc}") from exc
    target = getattr(module, attribute, None)
    if target is None:
        raise CalculatorError(f"{module_name} に {attribute} がありません")
    return target


# ---------------------------------------------------------------- 解決処理
def describe(config) -> str:
    """いま何を使うことになっているかを 1 行で表す。

    実際に渡る引数から作るので、そのレシピが ``model`` を受け取らない場合
    (EMT など) はモデル名を出さない。
    """
    spec = _spec(config)
    if spec.get("script"):
        return f"script:{spec['script']}"
    if spec.get("factory"):
        return f"factory:{spec['factory']}"
    try:
        target, options = resolve(config)
    except CalculatorError:
        return str(spec.get("backend", "sevennet"))
    name = target.name if isinstance(target, Recipe) else str(target)
    model = options.get("model")
    if model is None and isinstance(target, Recipe):
        model = target.defaults.get("model")         # レシピ側の既定モデル
    return f"{name} ({model})" if model else name


def _spec(config) -> dict:
    if hasattr(config, "section"):
        return dict(config.section("mlip"))
    return dict(config or {})


def resolve(config) -> tuple[Callable[..., Any], dict]:
    """(呼び出し可能なファクトリ, 渡すキーワード引数) を返す。"""
    spec = _spec(config)
    options = dict(spec.get("options") or {})

    if spec.get("script"):
        target: Any = load_recipe(spec["script"])
    elif spec.get("factory"):
        target = _factory_from_path(spec["factory"])
    else:
        target = get_recipe(spec.get("backend") or "sevennet",
                            spec.get("recipe_dirs"))

    for key in CONVENIENCE_KEYS:
        value = spec.get(key)
        if value is None or key in options:
            continue
        if isinstance(target, Recipe) and not target.accepts(key):
            continue
        options[key] = value
    return target, options


def get_calculator(config):
    """設定から ASE calculator を作る。"""
    target, options = resolve(config)
    try:
        return target(**options)
    except CalculatorError:
        raise
    except TypeError as exc:
        raise CalculatorError(
            f"calculator を作れませんでした ({exc})。"
            f"渡した引数: {sorted(options)}  -  mlip.options を確認してください") from exc
    except Exception as exc:
        raise CalculatorError(f"calculator を作れませんでした: {exc}") from exc


def check(config) -> list[str]:
    """作る前に分かる問題 (依存の不足など) を列挙する。"""
    try:
        target, _ = resolve(config)
    except CalculatorError as exc:
        return [str(exc)]
    if isinstance(target, Recipe):
        return target.check()
    return []


__all__ = ["Recipe", "CalculatorError", "available", "check", "describe", "discover",
           "get_calculator", "get_recipe", "load_recipe", "resolve"]
