"""M3GNet / MatGL — pymatgen 系の汎用ポテンシャル。"""

NAME = "matgl"
ALIASES = ("m3gnet",)
DESCRIPTION = "MatGL (M3GNet 系の事前学習ポテンシャル)"
REQUIRES = ("matgl",)
INSTALL = "uv pip install matgl"
DEFAULTS = {"model": "M3GNet-MP-2021.2.8-PES"}
MODELS = ("M3GNet-MP-2021.2.8-PES", "M3GNet-MP-2021.2.8-DIRECT-PES")


def build(model: str = "M3GNet-MP-2021.2.8-PES", **kwargs):
    import matgl
    from matgl.ext.ase import PESCalculator

    return PESCalculator(matgl.load_model(model), **kwargs)
