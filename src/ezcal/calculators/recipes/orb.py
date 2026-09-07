"""Orb — Orbital Materials の汎用ポテンシャル。"""

NAME = "orb"
ALIASES = ("orb-models",)
DESCRIPTION = "Orb foundation model (orb-models)"
REQUIRES = ("orb_models",)
INSTALL = "uv pip install orb-models"
DEFAULTS = {"model": "orb-v3-conservative-inf-omat", "device": "cpu"}
MODELS = ("orb-v3-conservative-inf-omat", "orb-v3-direct-20-omat", "orb-v2")


def build(model: str = "orb-v3-conservative-inf-omat", device: str = "cpu", **kwargs):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    loader = getattr(pretrained, str(model).replace("-", "_"), None)
    if loader is None:
        raise ValueError(f"orb_models.forcefield.pretrained に {model} がありません")
    return ORBCalculator(loader(device=device), device=device, **kwargs)
