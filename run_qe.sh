#!/bin/bash
#PBS -N {{JOB_NAME}}
#PBS -l nodes={{NODES}}:ppn={{PPN}}
#PBS -l walltime={{WALLTIME}}
#PBS -q {{QUEUE}}
#PBS -j oe
# ------------------------------------------------------------------
# ezcal の qsub 用テンプレート。ezcal が二重波括弧のプレースホルダ
# JOB_NAME, NODES, PPN, NPROC, QUEUE, WALLTIME, WORKDIR, MPIRUN,
# OMP_NUM_THREADS, COMMANDS を置換する (ezcal/scheduler.py を参照)。
# このファイルを構造ファイルの隣にコピーし、自分の計算環境に合わせて
# 編集して使う (module load、スクラッチ領域の扱いなど)。ezcal は
# ./run_qe.sh を最優先で読み込む。
# ------------------------------------------------------------------

cd "${PBS_O_WORKDIR:-{{WORKDIR}}}" || exit 1

export OMP_NUM_THREADS={{OMP_NUM_THREADS}}

# --- 環境ごとの設定 (ここを編集する) ------------------------------
# module load openmpi
# module load quantum-espresso
# source ~/.venv/ezcalenv312/bin/activate

echo "ホスト        : $(hostname)"
echo "作業ディレクトリ: $(pwd)"
echo "開始          : $(date)"

set -o pipefail

# --- ezcal が生成したコマンド --------------------------------------
{{COMMANDS}}
status=$?

echo "終了          : $(date)"
exit $status
