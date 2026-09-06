# ezcal (Easy Calculation)

初期構造ファイルを 1 つ渡すだけで第一原理計算が終わるようにする CLI パッケージです。
既定エンジンは Quantum ESPRESSO、計算エンジン層は差し替え可能で、SevenNet (MLIP) が
そのまま使え、VASP への拡張点も用意してあります。

**マニュアル: [`docs/manual.html`](docs/manual.html)**（ブラウザで開いてください。初心者向け／上級者向けの切り替えつき）

```bash
ezcal scf   Si.cif
ezcal relax Si.cif
ezcal bands Si.cif          # scf を前段として自動で流す
ezcal dos   Si.cif          # scf -> nscf -> dos.x/projwfc.x
ezcal auto  Si.cif          # vc-relax -> scf -> nscf -> dos -> bands まで一括

ezcal scf Si.cif --ecutwfc 60 --kmesh 8 8 8
```

---

## 1. インストール

```bash
source /home/tajimamainpc/.venv/ezcalenv312/bin/activate
uv pip install -e /home/tajimamainpc/06_claude_code/08_qez          # 本体
uv pip install -e '/home/tajimamainpc/06_claude_code/08_qez[all]'   # + mp-api, jupyter
uv pip install sevenn                                               # MLIP を使う場合
```

Quantum ESPRESSO の実行ファイルは `qe_config.yaml` の `qe.bin_dir`
(既定 `~/repos/q-e/bin`) から探します。

## 2. 何が自動で決まるか

「細かいことが分からなくても正しい計算が回る」ことを目的に、以下は指定しなければ
自動で決まります。すべて CLI オプションまたは設定ファイルで上書きできます。

| 項目 | 決め方 |
|---|---|
| 擬ポテンシャル | 元素と汎関数から `pseudopotentials.quantum-espresso.org` の索引を引き、PAW > USPP の順で選び自動ダウンロード (`~/.ezcal/pseudo`) |
| `ecutwfc` / `ecutrho` | ダウンロードした UPF ヘッダの "Suggested minimum cutoff" の全元素最大値。dual は PAW/USPP=8, NC=4 |
| k 点メッシュ | `dft.kspacing` (既定 0.25 Å⁻¹、2π 込み) から逆格子ベクトル長で決定 |
| nscf メッシュ | scf メッシュ × `nscf.kmesh_scale` (既定 2)、`occupations=tetrahedra` |
| バンド数 | 価電子数から占有バンド + 30〜40 % の空バンド |
| バンド経路 | seekpath の標準経路。**セルも seekpath の primitive cell に揃えてから scf を回す**ので経路と電荷密度が必ず整合する |
| 価電子数 | UPF の `z_valence` |
| バンドギャップ | 電子数から占有バンド数を数えて VBM/CBM を決定 (smearing で E_F が VBM より下に来ても金属と誤判定しない) |

## 3. コマンド

| コマンド | 内容 | 自動で流れる前段 |
|---|---|---|
| `ezcal scf` | SCF | - |
| `ezcal relax` | 原子位置最適化 | - |
| `ezcal vc-relax` | セル + 原子位置最適化 | - |
| `ezcal nscf` | 密メッシュ non-SCF | scf |
| `ezcal bands` | バンド構造 | scf |
| `ezcal dos` | 全 DOS + PDOS | scf, nscf |
| `ezcal auto` | 全部 | vc-relax → scf → nscf → dos → bands |
| `ezcal info` | 構造・対称性・k メッシュ・擬ポテンシャル・カットオフの確認 | - |
| `ezcal mp mp-149` | Materials Project から構造取得 | - |
| `ezcal pseudo list/fetch` | 擬ポテンシャルの一覧・取得 | - |
| `ezcal config init/show` | `qe_config.yaml` の生成・確認 | - |
| `ezcal plot RUNDIR` | 計算済みディレクトリから図を描き直す | - |
| `ezcal engines` | 利用可能なエンジンの確認 | - |

前段を流したくないときは `--only`、一部だけ飛ばすときは `--skip nscf,dos` を使います。

## 4. よく使うオプション

```
# DFT
--ecutwfc 60 --ecutrho 480     カットオフ (Ry)
--kmesh 8 8 8                  Monkhorst-Pack メッシュ
--kspacing 0.2                 メッシュを間隔で指定 (Å^-1)
-f pbe | pbesol | pz           汎関数 (擬ポテンシャルの選択も連動)
--input-dft vdw-df2-b86r       QE の input_dft を直接指定
--occupations smearing|fixed|tetrahedra
--smearing mv|gaussian|mp|fd  --degauss 0.02
--conv-thr 1e-8  --nbnd 20
--spin                         スピン分極 (nspin=2)
--magmom Fe=0.6,O=0            初期磁化
--magmom Ni=2.0/-2.0           反強磁性 (副格子ごとの初期磁化を "/" で並べる)
--afm Ni=2.0                   上と同じことの短縮形 (+m/-m を交互に配置)
--hubbard-u Fe=4.0             DFT+U (QE 7.x の HUBBARD カード)
--vdw grimme-d3

# 擬ポテンシャル
--pseudo-dir ~/pseudo
--pseudo Fe=Fe.pbe-spn-kjpaw_psl.1.0.0.UPF

# 実行
--np 8                         MPI プロセス数
--scheduler local|qsub  /  --qsub
--qsub-wait                    投入後ジョブ終了まで待って次の段へ進む
--queue batch --walltime 12:00:00 --nodes 1 --ppn 8
--qsub-script ./run_qe.sh
--dry-run                      入力とジョブスクリプトだけ書いて実行しない

# 出力
--plot matplotlib|plotly|both|none
--dpi 300  --emin -8 --emax 8  -o results --name run1
--keep-wfc

# その他
--engine qe|vasp|mlip
--config qe_config.yaml
--set dft.mixing_beta=0.2      任意の設定キーを直接上書き
--mp-api-key ...               (環境変数 MP_API_KEY でも可)
```

## 5. 磁性

```bash
# 強磁性
ezcal auto Fe.cif --spin --magmom Fe=0.6

# 反強磁性: 元素を副格子に分ける
ezcal auto NiO.cif --afm Ni=2.0 --hubbard-u Ni=6.0
ezcal scf  Cr.cif  --magmom Cr=1.0/-1.0        # --afm Cr=1.0 と同じ
```

`--afm Ni` を付けると、その元素のサイトを**出現順に** `Ni1`, `Ni2`, `Ni1`, ... と
ラベル付けし、`+m`, `-m` を交互に与えます。pw.x からは 2 つの species に見えるので
対称性も正しく下がります。`--hubbard-u Ni=6.0` のような元素指定は両方の副格子に展開されます。

- **磁気単位胞は自分で用意してください。** 反強磁性配置を収められる大きさのセルが必要です
  (例: NiO なら岩塩型の primitive cell ではなく AFM-II の菱面体 4 原子セル)。
  副格子を指定すると ezcal は自動で `--as-is` に切り替え、セルを縮約しません。
- サイト数が足りないときはその場でエラーになります。
- バンド計算では seekpath にも副格子を別 species として渡すため、
  磁気単位胞に対応した正しいブリルアンゾーンと経路が使われます。
- 結果には正味の磁化 `M` と絶対磁化 `|M|`、レポートにはサイトごとの磁気モーメントが出ます。

## 6. 設定ファイル `qe_config.yaml`

```bash
ezcal config init            # カレントに雛形を書き出す
ezcal config init --user     # ~/.config/ezcal/qe_config.yaml
ezcal config show            # 実際に効いている値と読んだ順番
```

優先順位は **パッケージ既定 → `~/.config/ezcal/qe_config.yaml` → `./qe_config.yaml`
→ `--config` → CLI オプション** です。実際に使われた設定は各実行ディレクトリに
`qe_config.used.yaml` として保存されるので、後から再現できます。

## 7. qsub での実行

```bash
ezcal auto Si.cif --qsub --queue batch --np 32 --qsub-wait
ezcal scf  Si.cif --qsub --dry-run          # 投入されるスクリプトを確認するだけ
```

- テンプレートは `./run_qe.sh` → `run.qsub.script` → パッケージ同梱 の順に探します。
- 置換されるプレースホルダは `{{JOB_NAME}} {{NODES}} {{PPN}} {{NPROC}} {{QUEUE}}
  {{WALLTIME}} {{WORKDIR}} {{MPIRUN}} {{OMP_NUM_THREADS}} {{COMMANDS}}`。
  module load などサイト固有の設定はテンプレート側に書き足してください。
- `--qsub-wait` を付けない場合、複数段のワークフローは 1 段目を投入した時点で止まります
  (段間で構造や電荷密度を受け渡す必要があるため)。

## 8. 出力

```
ezcal_out/Si_auto/
├── input_structure.cif / .json     入力構造
├── relaxed_structure.cif           緩和後
├── final_structure.cif
├── qe_config.used.yaml             実際に使った全設定
├── 00_vc-relax/  01_scf/  02_nscf/  03_dos/  04_bands/
│     ├── scf.in / scf.out          QE の入出力そのまま
│     └── scf_result.json           そのステップの解析結果
├── tmp/                            QE の outdir (全段で共有)
├── plots/  bands.png bands.html dos.png dos.html
│          bands_dos.png bands_dos.html convergence.png/html
│          bands.csv dos.csv
├── raw_data.json                   `ezcal plot` 用の生データ
├── summary.json                    機械可読サマリ
└── report.md                       人が読むサマリ
```

絶縁体ではプロットのゼロ点を VBM に、金属では E_F に取ります。

## 9. 機械学習ポテンシャル (SevenNet)

```bash
ezcal vc-relax Si.cif --engine mlip
ezcal auto     Si.cif --engine mlip     # 電子状態の段は自動でスキップし理由を表示
ezcal scf      Si.cif --engine mlip --set mlip.model=7net-l3i5
```

エネルギー・力・応力・構造最適化のみ対応します。バンドや DOS を要求された場合は
黙って別物を出さず、対応していない旨を返します。DFT の前段としての粗い緩和に有効です。

## 10. VASP

VASP バイナリでも、aiida-vasp の `mock-vasp` でも実行できます。
`mock-vasp` は INCAR/KPOINTS/POSCAR をパースしてハッシュを取り、レジストリに登録済みの
計算があればその出力を複製して返す疑似 VASP です。ライセンスの無いマシンでも
入力生成 → 実行 → 解析 → 作図の全経路をそのまま動かせます。

```bash
uv pip install aiida-vasp          # mock-vasp が入る

# 記録済み計算の再生
ezcal scf POSCAR --engine vasp --set vasp.command=mock-vasp \
      --set vasp.mock.registry=./vasp_mock_registry --as-is

# レジストリの中身
ezcal vasp registry --registry ./vasp_mock_registry
ezcal vasp hash <INCAR/KPOINTS/POSCAR のあるディレクトリ>

# 実 VASP のあるマシンで計算を記録 → レジストリを配って再生
ezcal vasp record <計算済みディレクトリ> --name si-scf/calc-000 --registry ./reg
```

設定例 (`vasp_mock.yaml`):

```yaml
engine: vasp
vasp:
  command: mock-vasp
  mock:
    registry: ./vasp_mock_registry
    vasp_cmd: null      # 実 VASP を指定すると、未登録の入力は実行して自動で記録する
  incar: {}             # INCAR タグの追加・上書き (null で削除)
  incar_mode: merge     # replace にすると vasp.incar がそのまま INCAR になる
  potcar_dir: null      # ライセンス版 POTCAR ライブラリ
  potcar_mode: auto     # potcar_dir があれば library、mock なら明示ラベル付きの
                        # プレースホルダ (実 VASP では書かずにエラーにする)
```

- **結果の出所は必ず明示されます。** mock で再生した結果には
  「どのレジストリ項目から複製したか」がレポートと `summary.json` に残ります。
  内蔵デモデータへのフォールバック (別の系のデータ) が起きた場合は警告になります。
- `ezcal` の汎用オプションは INCAR タグに変換されます (`--ecutwfc` → ENCUT (Ry→eV)、
  `--conv-thr` → EDIFF、`--spin`/`--magmom`/`--afm` → ISPIN + サイトごとの MAGMOM、
  `--hubbard-u` → LDAU 一式)。`--magmom` の値は QE では価電子数に対する割合、
  VASP では μB としてそのまま渡されます。
- バンド計算の KPOINTS は seekpath の経路をそのまま explicit 形式で書きます。
- DOS/PDOS・バンドの作図は QE と同じ経路を通るので、図も CSV も同じ形式で出ます。

VASP を実際に走らせるには `vasp.command` に実行ファイル、`vasp.potcar_dir` に
POTCAR ライブラリを設定してください (pymatgen の `pmg config -p <PSP dir> <target>`)。

## 11. 電荷密度と原子電荷 (`ezcal charge`)

```bash
ezcal charge MgO.cif --np 8                       # scf → dos → charge
ezcal charge NiO.cif --afm Ni=2.0 --np 8          # スピン密度も自動で出る
ezcal auto Si.cif --charge                        # auto の最後に付ける
ezcal charge X.cif --charge-kinds density,spin,ae_valence
```

`pp.x` で電荷密度を cube に落とし、**Bader** と **Löwdin** の 2 通りで原子電荷を出します。

### 出力

| ファイル | 内容 |
|---|---|
| `02_charge/density.cube` / `spin.cube` | 電荷密度・スピン密度（Gaussian cube） |
| `02_charge/atomic_charges.csv` | 原子ごとの Löwdin/Bader 電子数・電荷・体積・軌道分解・磁気モーメント |
| `properties.md` / `properties.json` | ギャップ・E_F・原子ごとの磁気モーメントと電荷を1か所に集約 |
| `plots/density_profile.png` | 各軸方向の平面平均（原子位置つき） |
| `plots/density_slice.png` | 3面の2D断面 |
| `plots/density_isosurface.html` | 3D等値面（plotly、原子位置つき） |

スピン分極計算では `spin_*` も同じ3種類が出ます。

### 原子電荷の2つの見方

- **Löwdin** — `projwfc.x` の出力に元から入っているので**追加計算ゼロ**。軌道分解（s/p/d/f）と
  原子ごとの磁気モーメントも同時に得られます。基底への射影なので定義依存が強め。
- **Bader** — 密度をグリッド上で原子の basin に分割します
  （Henkelman, Arnaldsson, Jónsson 2006 の on-grid 法を内蔵実装、**外部バイナリ不要**）。
  グリッド由来の不確かさが 0.01〜0.05 e 程度あります。

MgO の例: Bader が Mg +1.68 / O −1.68、Löwdin が Mg +1.52 / O −1.36。
値が違うのは定義が違うからで、**同じ手法どうしで比較**してください。

### どの密度を使うか

| `--charge-kinds` | pp.x | 用途 |
|---|---|---|
| `density`（既定） | plot_num=0 | 擬電荷密度。積分が厳密に価電子数と一致し、Bader が安定 |
| `spin` | 6 | ρ↑−ρ↓。nspin=2 なら自動で追加 |
| `ae_valence` / `ae_total` | 17 / 21 | PAW の全電子密度。**scf の FFT グリッドでは粗すぎて破綻します**（QE のドキュメントも「非常に密なグリッドが必要」と明記）。ezcal は積分値をチェックして警告します |

## 12. ベンチマーク (`ezcal bench`)

原子数の少ない金属 20 種・酸化物 20 種を `auto` で流し、実験値と Materials Project と
突き合わせます。**別の環境で ezcal が正しく動いているかを 1 コマンドで確認する**ためのものです。

```bash
ezcal bench list --set metals          # 対象と参照値の一覧
ezcal bench run  --set all -o 03_qe_bench --np 8      # 本番 (数時間)
ezcal bench run  --only Al,MgO -o /tmp/chk --np 8     # 動作確認だけなら数十秒
ezcal bench report 03_qe_bench --mp-api-key <KEY>     # 予実比較
```

構造はプロトタイプ (fcc/bcc/hcp/岩塩型/蛍石型/ウルツ鉱型/ルチル型/アナターゼ型/
ペロブスカイト型/赤銅鉱型/岩塩型 AFM-II) と実験格子定数から生成するので、
ネットワークも API キーも無しで走ります。Materials Project の値はレポート時に
**組成 + 空間群で照合して取得**します (material id を埋め込まないので、MP の再採番に影響されません)。

### 比較する量

| 量 | 計算側の出所 | 比較先 |
|---|---|---|
| 格子定数 a, c, c/a | vc-relax 後の従来型セル | 実験・MP |
| 原子あたり体積 V₀ | vc-relax | 実験・MP |
| 密度 | vc-relax | 実験・MP |
| 体積弾性率 B₀, B₀′ | E(V) の Birch–Murnaghan フィット | 実験 |
| バンドギャップ (直接/間接も) | 密メッシュ nscf | 実験・MP |
| 磁気モーメント | scf (サイト分解) | 実験・MP |
| N(E_F) | dos.x | 金属の文献値 |
| d バンド中心 | projwfc.x | 定性比較 |

- **E(V) は全体積で同じ k 点メッシュを使います。** `kspacing` に任せると体積ごとに
  メッシュが変わって E(V) が段差になり、B₀ が意味を失います。
- 凝集エネルギー・生成エネルギーは含めていません。孤立原子や O₂ の基準計算と
  アニオン補正が必要で、「インストールが正しいか」の確認とは別の作業だからです。
- 出力: `bench_results.json` (生データ)、`bench_comparison.csv` (量ごとの予実表)、
  `bench_summary.json` (量ごとの ME/MAE/MARE)、`bench_report.md` (人が読む表)、
  パリティ図 `plots/parity_vs_experiment.png|html` と `plots/parity_vs_mp.png|html`
  (実験用と Materials Project 用の 2 枚。MP の図は API キーを渡したときだけ作られます)。
- 途中で止めても `--resume` (既定) で続きから流せます。

## 13. テスト

```bash
cd 02_ezcal_test
./run_tests.sh          # 8 コア、緩い収束条件での一通りの動作確認
```
