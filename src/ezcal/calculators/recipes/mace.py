"""MACE — 事前学習の汎用ポテンシャル (mace-mp / mace-off)。"""

NAME = "mace"
ALIASES = ("mace-mp",)
DESCRIPTION = "MACE foundation model (mace_mp / mace_off)"
REQUIRES = ("mace",)
INSTALL = "uv pip install mace-torch"
DEFAULTS = {"model": "medium", "device": "cpu", "default_dtype": "float64"}
MODELS = ("small", "medium", "large", "medium-mpa-0", "medium-omat-0")


def build(model: str = "medium", device: str = "cpu", flavour: str = "mp", **kwargs):
    if str(flavour).lower() in {"off", "mace-off", "organic"}:
        from mace.calculators import mace_off as factory
    else:
        from mace.calculators import mace_mp as factory
    return factory(model=model, device=device, **kwargs)
