# ezcal の拡張方法

ezcal は「CLI / 設定 / ワークフロー / 作図」と「計算エンジン」を分けてあります。
エンジン側が守る約束は 2 つだけです。

1. `ezcal.engines.base.Engine` を継承し `run()` を実装する
2. 結果を `ezcal.engines.base.CalcResult` に詰めて返す

CLI・ワークフロー・作図は `CalcResult` しか見ないため、新しいコードを足しても
それらのファイルは 1 行も触りません。

```
ezcal/
├── cli.py            CLI (エンジン非依存)
├── config.py         設定の合成 (パッケージ既定 → user → cwd → --config → CLI)
├── structures.py     構造 I/O・対称化・k メッシュ・seekpath バンド経路
├── pseudo.py         擬ポテンシャル索引/取得/UPF ヘッダ解析 (QE 専用)
├── scheduler.py      Local / Qsub / dry-run  ← ジョブ投入はここだけが知っている
├── workflows.py      タスク連鎖 (bands なら scf を前に流す等)
├── md.py             MD / モンテカルロ (material-mc のラッパ)
├── charge.py         cube I/O・Bader 分割・Loewdin 電荷・原子電荷表
├── plotting.py       matplotlib / plotly
├── calculators/      ASE calculator の解決層
│   ├── __init__.py   レシピの探索と解決 (script > factory > backend)
│   └── recipes/      1 ファイル 1 ポテンシャル (sevennet.py, mace.py, emt.py, ...)
└── engines/
    ├── base.py       Engine ABC と CalcResult
    ├── __init__.py   レジストリ (register / get_engine)
    ├── qe/           Quantum ESPRESSO (inputs.py / outputs.py / engine.py)
    ├── vasp.py       VASP (骨組み)
    └── mlip.py       ASE calculator 系 (calculator の作り方は calculators/ が持つ)
```

## 1. 最小のエンジン

```python
from pathlib import Path
from ezcal.engines.base import CalcResult, Engine

class MyEngine(Engine):
    name = "mycode"
    supported = ("scf", "relax")          # 対応しないタスクは自動で拒否される

    def check(self):                       # 事前チェック (実行ファイルの有無など)
        return []                          # 問題があれば文字列のリストを返す

    def run(self, structure, task, workdir, prev=None, **kwargs):
        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        # 1. 入力を書く
        # 2. self.scheduler.execute(Stage(...)) で実行 (local/qsub の違いは意識しない)
        # 3. 出力を解析して CalcResult に詰める
        result = CalcResult(task=task, engine=self.name, workdir=workdir)
        result.energy = ...
        result.ok = True
        return result
```

`ezcal/engines/__init__.py` に 1 行足せば `--engine mycode` で使えます。

```python
register("mycode", lambda config, scheduler: MyEngine(config, scheduler))
```

設定は `config.get("mycode.command")` のように名前空間を切って読みます
(`qe_config.yaml` にセクションを足すだけで、config.py の変更は不要です)。

## 2. VASP

`engines/vasp.py` は実装済みです。入力生成 (INCAR/POSCAR/KPOINTS/POTCAR)、
実行、`vasprun.xml` と `OUTCAR` の解析、バンド・DOS・PDOS の抽出まで通っています。

### 2.1 mock-vasp で動かす (ライセンス不要)

aiida-vasp の `mock-vasp` は INCAR/KPOINTS/POSCAR をパースしてハッシュを取り、
レジストリに一致する計算があればその出力を複製する疑似 VASP です。

```bash
uv pip install aiida-vasp
```

```yaml
engine: vasp
vasp:
  command: mock-vasp
  mock:
    registry: ./vasp_mock_registry
```

レジストリの構造は `<base>/<名前>/calc-XXX/{inp,out}` で、`inp` に
INCAR・KPOINTS・POSCAR、`out` に vasprun.xml などの出力を置きます。
`ezcal vasp record <計算済みディレクトリ> --name <名前>` で登録できます。

**ハッシュはパース後の値で取られる**点に注意してください。`ENCUT = 200` (int) と
`ENCUT = 200.0` (float) は別物になります。ezcal が INCAR と KPOINTS を pymatgen 任せに
せず自前で書いているのはこのためです (`render_incar` / `render_kpoints_mesh`)。
一致しないときは `ezcal vasp hash <ディレクトリ>` で両者のハッシュを比べます。

`vasp.mock.vasp_cmd` に実 VASP を指定しておくと、未登録の入力は実 VASP で計算して
自動的にレジストリへ登録します。計算機センターで記録を作り、手元では再生する、
という運用ができます。

### 2.2 実 VASP で動かす

1. POTCAR ライブラリを用意し、`vasp.potcar_dir` を設定
   (pymatgen 側は `pmg config -p <PSP dir> <target>`)
2. `vasp.command` に MPI 実行ファイルを設定
3. 以上

`vasp.potcar_mode` の既定は `auto` で、`potcar_dir` があればライブラリから生成し、
無くてコマンドが mock のときだけ「これは擬ポテンシャルではない」と明記した
プレースホルダを書きます。実 VASP でライブラリが無い場合は黙って進まずエラーにします。

### 2.3 汎用オプションとの対応

`INCAR_MAP` (全タスク共通)、`RELAX_INCAR_MAP` (構造最適化のみ)、`TASK_INCAR`
(タスクが決める設定) の三段で組み立て、最後に `vasp.incar` のユーザー指定を適用します
(`null` を渡すとタグを削除、`incar_mode: replace` なら `vasp.incar` がそのまま INCAR)。
単位換算 (Ry→eV, Ry/Bohr→eV/Å, kBar→GPa) は同ファイルに集約してあります。

`--magmom` / `--afm` の値は変換せずそのまま渡します。QE は価電子数に対する割合、
VASP は μB と解釈が違いますが、いずれも初期値であり、意味を持つのは符号のパターンだからです。

## 3. 別の MLIP / NNP を足す

**ASE の calculator を 1 つ返せれば動きます。** ezcal 本体に手を入れる必要はありません。
`engines/mlip.py` は calculator の作り方を知らず、`ezcal.calculators` に任せています。
MD / MC (`ezcal md` / `ezcal mc`) も同じ解決層を使うので、一度足せば両方で使えます。

指定方法は 3 通りで、上のものが優先されます。

| 方法 | 設定キー | CLI |
|---|---|---|
| ユーザーの Python スクリプト | `mlip.script` | `--calc-script my_potential.py` |
| import パス | `mlip.factory` | `--calc-factory mace.calculators:mace_mp` |
| 同梱・追加のレシピ名 | `mlip.backend` | `--mlip-backend sevennet --model 7net-l3i5` |

いずれの場合も `mlip.options` (= `--calc-option key=value`) が `build()` の
キーワード引数になります。`mlip.model` / `mlip.device` は、そのレシピが引数として
受け取れる場合にだけ自動で渡されます (EMT のようにモデルの概念が無いレシピには渡らない)。

### 3.1 レシピを 1 ファイル書く

`src/ezcal/calculators/recipes/` と同じ形式のファイルです。雛形は
`ezcal mlip template -o my_potential.py` で取り出せます。

```python
NAME = "mypotential"
ALIASES = ("mypot",)
DESCRIPTION = "一覧に出る 1 行説明"
REQUIRES = ("mypotential",)            # import できなければ導入方法を案内する
INSTALL = "uv pip install mypotential"
DEFAULTS = {"device": "cpu"}           # build() の既定引数
MODELS = ("small", "large")            # 一覧表示用 (これ以外も渡せる)


def build(model=None, device="cpu", **options):
    from mypotential import MyCalculator
    return MyCalculator(model=model, device=device, **options)
```

置き場所は 2 通りです。

* `src/ezcal/calculators/recipes/` に置く → 同梱レシピとして常に見える
* 任意のディレクトリに置き、`mlip.recipe_dirs` にそのディレクトリを足す
  → `--mlip-backend mypotential` で選べる (パッケージは触らない)

`ezcal mlip list` に出るか、`ezcal mlip check --build` で実際に作れるかを確認できます。

### 3.2 設定ファイルの例

```yaml
engine: mlip                 # scf / relax / vc-relax を MLIP で回す場合
mlip:
  backend: mace
  model: medium
  device: cuda
  options: {default_dtype: float64}
```

```yaml
md:                          # ezcal md / mc は engine を見ず mlip: だけを見る
  mode: mcmd
mlip:
  script: ~/potentials/my_potential.py
  options: {checkpoint: ~/ckpt/best.pth}
```

## 4. スケジューラを足す

`scheduler.py` の `Scheduler` を継承して `execute(stage)` を実装し、
`get_scheduler()` に分岐を足します (例: Slurm)。テンプレート方式は qsub と同じで、
`{{COMMANDS}}` に実行行が展開されます。`sbatch` 用テンプレートを
`run.qsub.script` に指定し `submit_cmd`/`status_cmd` を変えるだけでも動きます。

## 5. MD / MC のモードを足す

`ezcal md` / `ezcal mc` は material-mc の `run_*` メソッドを 1 つのモードに対応させて
呼んでいるだけです。material-mc 側にメソッドが増えたら、`md.py` の `_MODE_LIST` に
1 行足し、必要なら `build_arguments()` に引数の組み立てを書きます。

```python
ModeSpec("my-mode", "run_my_mode", "説明", "mc", note="一覧に出る補足")
```

`kind` は引数の作り方の分類です (`md` / `mc` / `cycle` / `relax` / `event`)。
CLI・作図・レポートはモード名を知らないので、他のファイルは触りません。
後処理 (energy_log.csv の読み取り、時系列の作図、summary.json / report.md) は
全モード共通です。

## 6. タスクを足す

`workflows.CHAINS` に「そのタスクを実行するのに必要な前段」を書きます。

```python
CHAINS["phonon"] = ("scf", "phonon")
```

あとはエンジン側の `supported` に `"phonon"` を足し、`run()` で分岐すれば
`ezcal phonon Si.cif` が scf から自動で流れます (CLI のコマンド一覧は
`cli.TASKS` に 1 語足すだけ)。
