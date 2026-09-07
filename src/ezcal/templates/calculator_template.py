"""ezcal 用 calculator スクリプトの雛形。

    ezcal mlip template -o my_potential.py       # このファイルを取り出す
    ezcal md POSCAR --calc-script my_potential.py

ezcal が MLIP に求めるのは「ASE の calculator を 1 つ返すこと」だけである。
このファイルの中では何をしてもよい (チェックポイントの探索、環境変数の
読み取り、複数モデルのアンサンブル、DFT calculator の設定 ...)。

``build(**options)`` の引数は、設定ファイルの ``mlip.options`` および
``--calc-option key=value`` の内容がそのまま渡ってくる。``mlip.model`` /
``mlip.device`` (= ``--model`` / ``--device``) も、ここで受け取れる名前に
なっていれば自動で渡される。
"""

# ezcal mlip list に表示される情報 (すべて省略可能)
NAME = "my_potential"
DESCRIPTION = "自前のポテンシャル"
REQUIRES = ()                      # 例: ("torch", "mypotential")
INSTALL = "uv pip install mypotential"
DEFAULTS = {"device": "cpu"}       # build() の既定引数
MODELS = ()                        # よく使うモデル名 (表示用)


def build(model=None, device="cpu", **options):
    """ASE calculator を返す。

    下は SevenNet を自前 checkpoint で読む例。使うポテンシャルに合わせて
    中身を書き換えること。
    """
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=model or "7net-0", device=device, **options)
