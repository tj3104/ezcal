"""CHGNet — 磁気モーメントも予測する汎用ポテンシャル。"""

NAME = "chgnet"
DESCRIPTION = "CHGNet (磁気モーメント予測つき)"
REQUIRES = ("chgnet",)
INSTALL = "uv pip install chgnet"
DEFAULTS = {"device": "cpu"}
MODELS = ("0.3.0", "0.2.0")


def build(model=None, device: str = "cpu", **kwargs):
    from chgnet.model.dynamics import CHGNetCalculator

    if model:
        from chgnet.model.model import CHGNet

        kwargs["model"] = CHGNet.load(model_name=str(model))
    return CHGNetCalculator(use_device=device, **kwargs)
