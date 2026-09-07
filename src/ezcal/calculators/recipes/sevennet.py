"""SevenNet (7net) — GNN ポテンシャル。

モデルの切り替えは ``--model`` (= ``mlip.model``) で行う。事前学習モデル名
でも、自分で学習した checkpoint (.pth) のパスでも受け付ける。
多重忠実度モデル (7net-mf-ompa) は ``modal`` の指定が要るため、
``mlip.options.modal`` で渡す (既定は 'mpa')。
"""

NAME = "sevennet"
ALIASES = ("7net", "sevenn")
DESCRIPTION = "SevenNet (GNN, 事前学習モデル多数。自前 checkpoint も可)"
REQUIRES = ("sevenn",)
INSTALL = "uv pip install sevenn"
DEFAULTS = {"model": "7net-0", "device": "auto"}
MODELS = (
    "7net-0",            # 既定。軽量で速い (11Jul2024)
    "7net-l3i5",         # l=3 まで。7net-0 より高精度・低速
    "7net-mf-ompa",      # 多重忠実度 (modal='mpa' または 'omat24')
    "7net-omat",         # OMat24 で学習
    "7net-omni",
    "7net-omni-i8",
    "7net-omni-i12",
    "sevennet-0",
)

#: 多重忠実度モデルは modal を要求する
_MODAL_DEFAULT = {"7net-mf-ompa": "mpa", "sevennet-mf-ompa": "mpa"}


def build(model: str = "7net-0", device: str = "auto", modal=None, **kwargs):
    from sevenn.calculator import SevenNetCalculator

    if modal is None:
        modal = _MODAL_DEFAULT.get(str(model).lower())
    if modal is not None:
        kwargs["modal"] = modal
    return SevenNetCalculator(model=model, device=device, **kwargs)
