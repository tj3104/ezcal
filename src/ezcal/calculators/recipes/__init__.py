"""同梱の calculator レシピ (1 ファイル 1 ポテンシャル)。

各ファイルが定義するもの:

``NAME`` / ``ALIASES``
    ``--mlip-backend`` で指定する名前。
``DESCRIPTION``
    ``ezcal mlip list`` に出る 1 行説明。
``REQUIRES`` / ``INSTALL``
    必要な import 名と、足りないときに案内する導入コマンド。
``DEFAULTS``
    ``build()`` に渡す既定のキーワード引数。
``MODELS``
    よく使う事前学習モデルの名前 (一覧表示用。これ以外も渡せる)。
``build(**options)``
    ASE calculator を返す関数。

自分のポテンシャルを足す場合、このディレクトリと同じ形式のファイルを
好きな場所に置き、``mlip.recipe_dirs`` にそのディレクトリを追加する
(あるいは ``mlip.script`` でファイルを直接指定する)。
"""
