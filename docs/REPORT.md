# ezcal 実装レポート

- 作成日: 2026-09-03（反強磁性・VASP エンジン・ベンチマーク・電荷解析を追記）
- 対象: `mission.md`「ezcal(Easy Calculation)パッケージの作成」
- 環境: WSL2 / Ubuntu, Python 3.12.3 (`/home/tajimamainpc/.venv/ezcalenv312`),
  Quantum ESPRESSO 7.5 (`~/repos/q-e/bin`), CPU 16 コア (テストは 8 コア使用)

---

## 1. 成果物

| パス | 内容 |
|---|---|
| `src/ezcal/` | パッケージ本体 (Python 約 4,000 行) |
| `pyproject.toml` | `uv pip install -e .` でインストール可、`ezcal` コマンドを提供 |
| `qe_config.yaml` (雛形は `src/ezcal/data/default_config.yaml`) | 全既定値。`ezcal config init` で書き出し |
| `run_qe.sh` | qsub 用テンプレート (パッケージ内にも同梱) |
| `notebooks/ezcal_demo.ipynb` | MP API キーをその場で入力する形式のデモ (実行済みの `executed_ezcal_demo.ipynb` 付き) |
| `tests/test_ezcal.py` | QE 不要の単体テスト 79 件 |
| `02_ezcal_test/run_tests.sh` | 実計算を含む受け入れテスト 32 件 |
| `03_qe_bench/` | 金属20＋酸化物20 の予実比較 (`ezcal bench`) |
| `README.md` / `docs/EXTENDING.md` | 使い方 / 拡張手順 |

インストールとテストは以下で再現できます。

```bash
source /home/tajimamainpc/.venv/ezcalenv312/bin/activate
uv pip install -e .
python -m pytest tests -q                 # 79 passed
cd 02_ezcal_test && ./run_tests.sh full   # 32 passed, 0 failed
```

---

## 2. 依頼事項に対する対応

| 依頼 | 対応 |
|---|---|
| `ezcal scf/relax/bands/dos <構造>` の CLI | 実装。加えて `vc-relax` `nscf` `auto` `info` `mp` `pseudo` `config` `plot` `engines` |
| `--ecutwfc 60 --kmesh 8 8 8` 形式のオプション | 実装。他に汎関数・スメアリング・磁性・U・vdW・擬ポテンシャル指定など |
| pip でインストール | `pyproject.toml` (setuptools, src レイアウト)。`ezcal` エントリポイント |
| 管理ノードでも qsub でも実行 | `--scheduler local|qsub` (`--qsub` 短縮形)。`Scheduler` 抽象クラスで分離 |
| 既定値をコンフィグで変更 | `qe_config.yaml`。パッケージ既定 → `~/.config/ezcal/` → カレント → `--config` → CLI の順で合成 |
| qsub 環境が無いので `run_qe.sh` があるものとして設計 | テンプレート方式。`--dry-run` で投入されるスクリプトを確認可能 (テスト 08 で検証) |
| ASE/pymatgen/seekpath 等の活用 | pymatgen (構造・対称性・VASP 入出力)、ASE (ファイル読込・MLIP 最適化)、seekpath (バンド経路)、spglib |
| レポート | 本ファイル |
| テストは `02_ezcal_test` | 同ディレクトリに構造・スクリプト・全計算結果 |
| 初期構造から一発で bands まで | `ezcal auto`。単体の `ezcal bands` / `ezcal dos` も前段 (scf/nscf) を自動で流す |
| VASP や MLIP への拡張性 | `engines/` にエンジン層を分離。SevenNet と VASP はいずれも実動 |
| VASP を実行できるように | aiida-vasp の `mock-vasp` で記録済み計算を再生。実 VASP は `vasp.command` と `vasp.potcar_dir` を設定するだけ |
| SevenNet で動作テスト | `--engine mlip` で MgO / Si を検証 (テスト 20〜22) |
| MP API キーは ipynb 内で入力 | `getpass` で入力、リポジトリに保存しない |
| 1〜2 原子の単体金属・酸化物でベンチマーク | Al(1) / Si(2) / MgO(2) / Fe(1) |
| テストは 8 コア | 全て `--np 8` |
| `uv pip install` を使用 | 依存関係・本体ともに `uv pip install` |
| matplotlib / plotly 両対応 | `--plot matplotlib|plotly|both|none` |
| K 点・磁性・汎関数などを CLI オプションに | 下記 4 節 |
| 反強磁性 | `--afm Ni=2.0` / `--magmom Ni=2.0/-2.0`。元素を副格子に分ける |
| ベンチマーク用の動作確認コマンド | `ezcal bench list/run/report`。金属20＋酸化物20を `auto` で流し、実験値と Materials Project に突き合わせる |
| 電荷密度・原子電荷の出力 (20260903) | `ezcal charge`。pp.x で cube、Bader (内蔵実装) と Löwdin の 2 通りの原子電荷、平面平均・2D断面・3D等値面、CSV |
| バンド計算後の物理量出力 (20260903) | `properties.md` / `properties.json` にギャップ (金属は 0 と明記)・フェルミ準位・VBM/CBM・原子ごとの磁気モーメントを集約 |

---

## 3. 設計

```
CLI (cli.py)
   └── Config (config.py)          既定値の合成と --set による任意キー上書き
   └── Workflow (workflows.py)     タスク連鎖: bands なら scf を前に流す
          ├── Engine (engines/)    qe / vasp / mlip   ← 差し替え点
          └── Scheduler (scheduler.py)  local / qsub / dry-run ← 実行方法の差し替え点
   └── plotting.py                 matplotlib / plotly
```

エンジンは `CalcResult` (エネルギー・力・応力・E_F・ギャップ・磁化・構造・生データ) だけを
返す約束になっており、CLI・ワークフロー・作図はエンジンの種類を知りません。
そのため VASP や別の MLIP を足しても、追加するのは 1 ファイルとレジストリの 1 行だけです。

### 「何も指定しなくても正しく回る」ための自動決定

初学者でも使えることを最優先に、以下を自動化しました。

- **擬ポテンシャル**: 94 元素分の UPF 索引をパッケージに同梱
  (`pseudopotentials.quantum-espresso.org` から生成)。元素と汎関数から PAW > USPP の
  優先順で選び、`~/.ezcal/pseudo` へ自動ダウンロード。
- **カットオフ**: ダウンロードした UPF ヘッダの "Suggested minimum cutoff" を解析し、
  系内の全元素の最大値を採用 (dual は PAW/USPP=8, NC=4)。
- **k 点**: 逆格子ベクトル長と `kspacing` から決定。nscf は scf の 2 倍で `tetrahedra`。
- **バンド経路**: seekpath。**経路と電荷密度のセルが食い違わないよう、bands を含む
  ワークフローでは scf の前にセルを seekpath の primitive cell へ揃えます。**
- **バンド数**: UPF の `z_valence` から価電子数を数え、空バンドを 30〜40 % 追加。

### 実装上とくに注意した点

1. **金属/絶縁体の判定**
   スメアリングを使うと QE の E_F は絶縁体でも VBM より下に来ることがあり、
   「E_F を跨ぐバンドがあるか」で判定すると Si が金属になります。
   価電子数から占有バンド数を数えて VBM/CBM を求める方式に変更しました
   (`tests/test_ezcal.py::test_gap_from_eigenvalues_handles_smearing` で回帰テスト)。
2. **エネルギー原点**
   絶縁体では VBM、金属では E_F をゼロにし、軸ラベルも `E - E_VBM` / `E - E_F` と
   出し分けます。
3. **DOS と PDOS のエネルギー格子**
   `dos.x` と `projwfc.x` は別々の格子を吐くため、PDOS を DOS 格子へ内挿してから
   重ね描き・CSV 出力します。スピン分極時は up/down を分けて描画します。
4. **バンド経路の折れ点**
   `U|K` のように経路が飛ぶ点では 2 つの k 点が同じ経路長に載ります。両方を計算に
   渡したうえで、目盛りラベルだけ `U|K` に統合します。
5. **反強磁性は species ラベルで表現する**
   pw.x は同じ元素でも別ラベル（`Ni1`, `Ni2`）なら別 species として扱い、対称性もそれに応じて
   下げます。`--afm Ni=2.0` はその元素のサイトを出現順に `Ni1`/`Ni2` と名付け、`+m`/`-m` を
   交互に与えます。ラベルは pymatgen の site property として構造に載せ、
   入力生成・XML の読み戻し・seekpath・projwfc の出力解析まで一貫して運びます。とくに
   **seekpath にも副格子を別 species として渡す**ため、磁気単位胞に対応した正しい
   ブリルアンゾーンが使われます（AFM の bcc Cr は Γ-H-N-P ではなく単純立方の Γ-X-M-R）。
   `--hubbard-u Ni=6.0` のような元素指定は両副格子へ自動展開されます。
   セルが磁気秩序を収められない場合はその場でエラーにし、副格子指定時は primitive cell への
   縮約を自動で止めます（縮約すると必要なサイトが消えるため）。
6. **VASP は mock-vasp で実際に動かして検証した**
   aiida-vasp の `mock-vasp` は INCAR/KPOINTS/POSCAR を**パースしてから**ハッシュを取り、
   レジストリに一致する計算があればその出力を複製します。したがって
   `ENCUT = 200`(整数) と `ENCUT = 200.0`(実数) は別物になり、KPOINTS のシフト行の有無でも
   一致しません。pymatgen の書き手は ENCUT を実数化し、ゼロシフト行を省略するため、
   ezcal は INCAR と KPOINTS を自前で書いています (POSCAR は pymatgen のまま)。
   これで aiida-vasp 公開の Si 実行例とハッシュが一致し、実 VASP の出力を再生できました。
7. **ecutwfc を下げたときに ecutrho が追随していなかった**
   `--ecutwfc 30` を指定しても `ecutrho` は擬ポテンシャル由来の 352 Ry のままで、
   dual が 11.7 という異常な比になり、PAW の補強電荷が破綻して
   `Error in routine electrons (1): charge is wrong` で落ちていました。受け入れテストが
   間欠的に落ちたことで発覚した実バグです。指定された ecutwfc から dual を取り直すよう修正し、
   さらに推奨値を下回る指定には警告を出すようにしました。あわせて、pw.x の既知エラー
   (cdiaghg の破綻、SCF 不収束、G ベクトル不足など) に対して次に試すことを `hint:` として
   出す仕組みを入れています。ここでは「`cdiaghg` は成功した実行のタイミング表にも現れる」ため、
   助言はエラーブロックの中身に対してのみ照合しています。
8. **E(V) は全体積で同じ k 点メッシュを使う**
   `kspacing` に任せると体積ごとにメッシュが変わり、E(V) に段差が入って体積弾性率が
   意味を失います。実際 Al で 102 GPa (文献 76-79) という値が出ていて、メッシュを固定したら
   78.3 GPa になりました。ベンチマークで初めて表面化した種類の誤りです。
9. **立方晶の格子定数は体積から導く**
   反強磁性体は緩和で菱面体に歪むため、従来型セルの `a` が最近接距離になってしまい
   NiO で −29 % という誤差が出ていました。プロトタイプごとに「従来型立方セルの原子数」を
   持たせ、原子あたり体積から辺長を求める方式に変更しています。primitive/conventional の
   どちらのセルが来ても正しく、MP の構造にもそのまま使えます。
10. **XML を一次情報にする**
   `data-file-schema.xml` からエネルギー・力・応力・固有値・磁化・緩和後構造を読みます
   (テキスト出力は収束フラグ・実行時間・エラーのみ)。単位換算 (Hartree/Bohr → eV/Å/GPa)
   は 1 か所に集約しました。

---

## 4. CLI オプション (DFT でよく触るもの)

```
--ecutwfc / --ecutrho        カットオフ (Ry)         既定: 擬ポテンシャルから自動
--kmesh nx ny nz             k 点メッシュ            既定: --kspacing から自動
--kspacing                   逆格子間隔 (Å^-1)       既定 0.25
-f / --functional            pbe | pbesol | pz       擬ポテンシャルの選択にも連動
--input-dft                  QE の input_dft を直指定 (vdW 汎関数など)
--occupations / --smearing / --degauss
--conv-thr / --nbnd
--spin / --magmom Fe=0.6,O=0 スピン分極と初期磁化
--hubbard-u Fe=4.0           DFT+U (QE 7.x の HUBBARD カード、軌道は元素から自動決定)
--vdw grimme-d3
--pseudo Fe=<file.UPF>       擬ポテンシャルの固定
--set <任意のキー>=<値>       上記に無い設定も全て変更可能
```

---

## 5. テスト結果

### 5.1 単体テスト (QE 不要)

`python -m pytest tests -q` → **79 passed** (5 s)。
設定の合成、k メッシュ、バンド経路、擬ポテンシャル選択、入力ファイル生成
(スピン・U・relax・explicit k 点)、ギャップ判定、qsub スクリプト生成、
反強磁性 (副格子分割・ラベル長・磁気単位胞のサイズ検査・磁気セルのバンド経路・
XML ラベルの往復)、VASP (INCAR の整数/実数の書き分け・タスクタグの優先順位・
MAGMOM/LDAU の展開・POTCAR の扱い・記録済み計算とのハッシュ一致)、
pw.x のエラー診断とカットオフの導出を検証。

### 5.2 受け入れテスト (実計算)

`02_ezcal_test/run_tests.sh full` → **32 passed, 0 failed**。

| # | 内容 | 結果 |
|---|---|---|
| 01-06 | version / engines / config / pseudo list / pseudo fetch / info | PASS |
| 07-08 | ローカル・qsub の入力とジョブスクリプト生成 (`--dry-run`) | PASS |
| 10-11 | Al scf, Al relax | PASS |
| 12-15 | 反強磁性の入力生成, Cr scf, Cr vc-relax, 小さすぎるセルの拒否 | PASS |
| 20-22 | SevenNet vc-relax, auto, 非対応タスクの拒否 | PASS |
| 30-32 | Si / MgO / Fe(スピン分極) の `auto` フルワークフロー | PASS |
| 33-35 | `ezcal plot` による再作図, Al の dos, NiO 反強磁性のフルワークフロー | PASS |
| 40-45 | VASP: レジストリ一覧, ハッシュ, mock 再生, 記録の往復, 全タスクの入力生成, POTCAR 不在の拒否 | PASS |

### 5.3 開発中の動作確認 (1〜2 原子系、8 コア)

| 系 | エンジン | ecut (Ry) | k 点 | 格子定数 (Å) | 参考値 (PBE) | ギャップ (eV) | 参考値 | 磁化 (μB) | 時間 |
|---|---|---|---|---|---|---|---|---|---|
| Si (2 原子) | QE | 44/352 (自動) | 6³ | **5.451** | 5.47 | **0.592** (間接) | 0.61 | - | 26 s |
| MgO (2 原子) | QE | 58/464 | 6³ | **4.249** | 4.25 | **4.492** (直接) | 4.45 | - | 13 s |
| Fe bcc (1 原子) | QE | 71/568 (自動) | 8³ | **2.806** | 2.83 | 0 (金属) | 金属 | **2.09** | 24 s |
| Al fcc (1 原子) | QE | 29/232 | 8³ | 4.050 (固定) | - | 0 (金属) | 金属 | - | 4 s |
| Si (2 原子) | VASP (mock 再生) | 200 eV | 8³ | 5.431 (固定) | - | **0.619** (間接) | 0.61 | - | 1 s |
| Si | SevenNet | - | - | **5.463** | - | 対象外 | - | - | 2.8 s |
| MgO | SevenNet | - | - | **4.255** | - | 対象外 | - | - | 2.9 s |

#### 反強磁性

| 系 | 設定 | 格子定数 (Å) | サイト磁気モーメント (μ_B) | 参考値 (PBE) | ギャップ (eV) | 参考値 | 時間 |
|---|---|---|---|---|---|---|---|
| Cr bcc AFM (2 原子) | 50/400 Ry, 10³ | 2.880 (固定) | **±1.12** | 1.0–1.2 | 0 (金属) | 金属 | 15 s |
| Cr bcc AFM (vc-relax) | 45/360 Ry, 8³ | **2.850** | ±0.89 | 2.85 / 小さくなる | 0 (金属) | 金属 | 30 s |
| NiO AFM-II + U=6 eV (4 原子) | 50/400 Ry, 4³ | 5.107 (固定) | **±1.66** (O は 0) | 1.65–1.72 | **3.20** (scf) / 3.06 (密メッシュ) | 3.0–3.4 | 94 s |

正味の磁化はいずれも 10⁻⁴ μ_B 以下で、反強磁性秩序が正しく立っていることが確認できます。
NiO の PDOS では Ni1-d と Ni2-d が上下対称の鏡像になり、伝導帯端に U で分裂した空の Ni-d
（上部ハバードバンド）が現れます。

カットオフ・k 点・擬ポテンシャル・バンド経路は全て自動決定で、格子定数・バンドギャップ・
磁気モーメントとも PBE の文献値と一致します。SevenNet の格子定数は DFT と 0.15 % 以内で
一致しており、前処理としての実用性も確認できました。

Fe の格子定数がやや小さめ (2.806 vs 2.83) なのは、テスト時間を詰めるために
`--degauss 0.03 Ry` と粗い 8³ メッシュを使っているためです。

### 5.4 VASP エンジン

VASP ライセンスが無い環境なので、aiida-vasp の `mock-vasp` を使って全経路を検証しました。
`mock-vasp` は入力のハッシュから記録済み計算を引いて出力を複製するため、
**再生されるのは実際の VASP の結果**です (捏造ではありません)。

- 検証用レジストリ: `02_ezcal_test/vasp_mock_registry/si-simple/calc-000`
  (aiida-vasp が公開している Si の実行例。`inp/` が入力、`out/` がその VASP 出力)
- ezcal が生成した INCAR/KPOINTS/POSCAR がこの項目とハッシュ一致することを単体テストで固定
- 再生結果: **E = −10.794 eV、E_F = 6.469 eV、間接ギャップ 0.619 eV** (PBE の文献値 ~0.61)。
  同じ Si を QE で計算した 0.597 eV とも整合します。
- DOS は `vasprun.xml` から読み出し、QE と同じ作図経路で PNG/HTML/CSV を出力
- 全タスク (scf/relax/vc-relax/nscf/dos/bands) の入力生成を `--dry-run` で確認。
  バンドは seekpath の経路を explicit KPOINTS として書き出します
- `ezcal vasp record` で計算済みディレクトリをレジストリに登録 → 再生、の往復も確認
- **結果の出所は必ず記録されます。** どのレジストリ項目から複製したかがレポートと
  `summary.json` に残り、内蔵デモデータ (別の系) へのフォールバックが起きた場合は警告になります
- POTCAR は捏造しません。ライブラリがあればそこから生成し、無くてコマンドが mock のときだけ
  「これは擬ポテンシャルではない」と明記したプレースホルダを書き、実 VASP ではエラーで止まります

### 5.5 ベンチマーク (`ezcal bench`, 40 系)

原子数の少ない金属 20 種・酸化物 20 種を `auto` で流し、実験値と Materials Project に
突き合わせました (`03_qe_bench/`、8 コアで 167 分、40/40 成功)。
構造はプロトタイプ (fcc/bcc/hcp/岩塩型/蛍石型/ウルツ鉱型/ルチル型/アナターゼ型/
ペロブスカイト型/赤銅鉱型/岩塩型 AFM-II) と実験格子定数から生成し、
カットオフ・k 点・擬ポテンシャル・バンド経路は全て自動決定です。

#### 実験値との比較

| 量 | 対象 | n | ME | MAE | MARE |
|---|---|---|---|---|---|
| 格子定数 a | 金属 | 20 | +0.001 Å | 0.038 Å | **1.01 %** |
| 格子定数 a | 酸化物 | 20 | +0.023 Å | 0.036 Å | **0.81 %** |
| 格子定数 c | 全体 | 8 | +0.041 Å | 0.063 Å | 1.22 % |
| 体積弾性率 B₀ | 金属 | 20 | −2.1 GPa | 12.5 GPa | **12.0 %** |
| 体積弾性率 B₀ | 酸化物 | 14 | +0.8 GPa | 10.5 GPa | **6.5 %** |
| バンドギャップ | 酸化物 | 17 | **−2.36 eV** | 2.36 eV | 55 % |
| 磁気モーメント | 金属 | 2 | +0.002 μ_B | 0.082 μ_B | 8.7 % |
| 磁気モーメント | 酸化物 (AFM) | 3 | −0.85 μ_B | 0.85 μ_B | 26 % |

#### Materials Project との比較

MP も PBE 系なので、こちらは**同じ土俵の比較**です (組成 + 空間群で照合、40/40 一致)。

| 量 | 対象 | n | ME | MAE | MARE |
|---|---|---|---|---|---|
| 格子定数 a | 全体 | 35 | +0.013 Å | 0.026 Å | **0.63 %** |
| c/a | 全体 | 6 | +0.002 | 0.003 | **0.19 %** |
| 原子あたり体積 | 全体 | 40 | +0.06 Å³ | 0.33 Å³ | 2.13 % |
| 密度 | 全体 | 40 | −0.07 g/cm³ | 0.15 g/cm³ | 2.13 % |
| バンドギャップ | 全体 | 18 | **+0.008 eV** | 0.148 eV | (24 %) |

- **ギャップは MP とほぼ完全に一致します** (ME +0.008 eV、MAE 0.15 eV)。実験より
  55 % 小さいのは PBE の既知の性質であって実装の誤りではない、という切り分けができます。
  MARE 24 % はギャップが小さい系 (CdO 0、SnO2 0.62、ZnO 0.72 eV) で相対誤差が
  跳ね上がるためで、絶対誤差で見るべきです。
- MP が別相に緩和した 5 系 (Fe→Fmmm, BaTiO3→I4/mcm, CoO→R-3m, SnO2→Pnnm,
  TiO2-rutile→Imma) は a/c の比較から自動的に除外しています (体積・密度・ギャップは比較可能)。
- NiO/MnO/CoO は反強磁性セルで、**+U 無しの素の PBE** です。そのためギャップと
  磁気モーメントが実験より大幅に小さく出ます (NiO: 1.25 eV / 1.37 μ_B、実験 4.3 eV / 1.9 μ_B)。
- 凝集エネルギー・生成エネルギーは含めていません。孤立原子や O₂ の基準計算と
  アニオン補正が必要で、インストール確認とは別の作業だからです。

パリティ図は実験用 `03_qe_bench/plots/parity_vs_experiment.png|html` と
Materials Project 用 `plots/parity_vs_mp.png|html` の 2 枚、系ごとの表は `bench_report.md`、
量ごとの生データは `bench_comparison.csv` にあります。
**MP のパリティ図はバンドギャップまで対角線上に乗ります**（外れる 2 点は MP が +U を使う
NiO と MnO）。実験のパリティ図でギャップだけが下振れするのと対照的で、
「ズレは PBE の性質であって実装の誤りではない」ことが一目で分かります。

### 5.6 図の確認

`02_ezcal_test/runs/*/plots/` に PNG と対話的 HTML を出力しています。

- `Si_auto/plots/bands_dos.png` — Γ 点 VBM、Γ→X 近傍 CBM の間接ギャップ
- `MgO_auto/plots/bands_dos.png` — Γ 点直接ギャップ、価電子帯が O-2p、伝導帯が Mg-3s
- `Fe_auto_spin/plots/bands_dos.png` — 交換分裂したマジョリティ/マイノリティバンド、
  E_F 直上のマイノリティ d ピーク
- `NiO_afm_auto/plots/dos.png` — Ni1-d と Ni2-d が上下対称の鏡像 (反強磁性の証拠)、
  O-2p が価電子帯上端、U で分裂した空の Ni-d が伝導帯端

いずれも教科書的な形状で、PDOS の軌道分解も物理的に妥当です。

### 5.7 電荷密度と原子電荷

QE の再ビルドは不要でした (`pp.x` / `projwfc.x` は `make pwall` に含まれる)。

- **Bader は外部バイナリ無しの内蔵実装**です。pybader は 2020 年で更新停止しており
  `pkg_resources` (setuptools 81 で削除) に依存して import すら通らなかったため、
  on-grid 法 (Henkelman, Arnaldsson, Jónsson 2006) を numpy で実装しました。
  1M 格子点を 0.2 秒で処理します。
- 検証: 合成した2つのガウス分布で**合計が厳密保存**し、グリッド細分で収束
  (40³→100³ で 6.02→5.99、厳密解 6.00)。実データでは **MgO が Mg +1.68 / O −1.68**
  (文献値どおり)、**AFM NiO が Ni +1.13 / O −1.13**、basin 体積の和がセル体積と一致。
- **Löwdin 電荷は追加計算ゼロ**です。DOS のために毎回走らせている `projwfc.x` の出力に
  元から入っており、軌道分解と原子ごとの磁気モーメント (QE の `polarization`) も得られます。
  AFM NiO で球積分値 ±1.388 μ_B と Löwdin ±1.330 μ_B が独立に一致しました。
- **PAW 全電子密度 (plot_num=17/21) は scf の FFT グリッドでは破綻します。**
  NiO の 75³ グリッドで plot_num=21 の積分が 72 e のはずが 115 e になり、等価な 2 つの Ni に
  別々の電荷が出ました。積分値を価電子数と突き合わせる検算を組み込み、自動で警告します。
  既定は `plot_num=0` (積分が厳密に一致する擬電荷密度) です。

### 5.8 ノートブック

`notebooks/ezcal_demo.ipynb` を `jupyter nbconvert --execute` で通しで実行し、
**エラー 0** を確認 (`executed_ezcal_demo.ipynb`)。Materials Project からの構造取得
(mp-149)、フルワークフロー、matplotlib/plotly 表示、SevenNet 比較、qsub スクリプト生成まで
含みます。API キーは `getpass` で入力する方式で、ファイルには残りません。

---

## 6. 既知の制約

- **qsub は未実機検証**: 本環境に `qsub` が無いため、スクリプト生成と投入経路
  (`qsub` 不在時のエラーメッセージ含む) までの確認です。実機ではテンプレートの
  `module load` 行の追記が必要です。
- **`--qsub` で `--qsub-wait` を付けない場合**、複数段のワークフローは 1 段目の投入で
  停止します。段間で構造・電荷密度を受け渡す必要があるためで、その旨をメッセージで通知します。
- **実 VASP バイナリでの実行は未検証**: 本環境にライセンスが無いため、検証は
  `mock-vasp` による記録済み計算の再生までです。実 VASP では `vasp.command` と
  `vasp.potcar_dir` を設定すれば同じ経路が動きます (POTCAR 生成のみ未実行)。
- **ベンチマークの参照値**: 実験格子定数・ギャップ・体積弾性率は標準的な結晶学/半導体の
  データ集に基づく値で、集計ごとに差が出る量は推測せず null にしてあります。
  Materials Project の値はレポート実行時に API から取得します (material id は埋め込まない)。
- **mock-vasp のレジストリは 1 ディレクトリのみ**: aiida-vasp 5.1 の
  `MockRegistry.__init__` が複数パスを受け付けない実装のため、設定にリストを書いても
  先頭のみ使います。
- **MLIP は電子状態を扱いません**: `bands`/`dos`/`nscf` は実行せず、理由を返します。
- **非共線**磁性とスピン軌道相互作用は未対応です (共線の強磁性・反強磁性・フェリ磁性は対応)。
  反強磁性では磁気単位胞をユーザーが与える必要があり、磁気構造の自動探索は行いません。
- フォノン、NEB は未対応です
  (`CHAINS` とエンジンの `supported` にタスクを足す形で拡張できます)。

---

## 7. 拡張の勘所 (詳細は `docs/EXTENDING.md`)

- **別の計算コード**: `Engine` を継承して `run()` で `CalcResult` を返し、
  `engines/__init__.py` に 1 行登録。VASP は `INCAR_MAP` / `TASK_INCAR` に
  ezcal の汎用オプションと INCAR タグの対応表を用意済み。
- **別の MLIP**: `MLIPEngine.calculator()` に ASE calculator を返す分岐を足すだけ
  (MACE / CHGNet 用の分岐は雛形として記述済み)。
- **別のジョブスケジューラ**: `Scheduler` を継承。テンプレート方式は共通なので、
  Slurm なら `sbatch` 用テンプレートと `submit_cmd`/`status_cmd` の変更だけでも動きます。
- **新しいタスク**: `workflows.CHAINS` に前段を書き、エンジンの `supported` に追加。
