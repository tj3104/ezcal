"""ASE 内蔵の EMT。

追加のインストールが要らないので、ワークフローの配線を確認したいとき
(MD/MC モードの動作確認、CI、ドキュメントの例) に使う。対応元素は
Al, Cu, Ag, Au, Ni, Pd, Pt, C, N, O に限られ、精度も期待してはいけない。
"""

NAME = "emt"
DESCRIPTION = "ASE 内蔵 EMT (依存なし。配線確認・デモ用)"
REQUIRES = ()
DEFAULTS: dict = {}
MODELS = ()


def build(**kwargs):
    from ase.calculators.emt import EMT

    return EMT(**kwargs)
