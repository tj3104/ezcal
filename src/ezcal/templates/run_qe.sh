#!/bin/bash
#PBS -N {{JOB_NAME}}
#PBS -l nodes={{NODES}}:ppn={{PPN}}
#PBS -l walltime={{WALLTIME}}
#PBS -q {{QUEUE}}
#PBS -j oe
# ------------------------------------------------------------------
# ezcal qsub template.  ezcal substitutes the double-brace placeholders
# JOB_NAME, NODES, PPN, NPROC, QUEUE, WALLTIME, WORKDIR, MPIRUN,
# OMP_NUM_THREADS and COMMANDS (see ezcal/scheduler.py).
# Copy this file next to your structure and edit it for your site
# (module load, scratch handling, ...); ezcal picks up ./run_qe.sh first.
# ------------------------------------------------------------------

cd "${PBS_O_WORKDIR:-{{WORKDIR}}}" || exit 1

export OMP_NUM_THREADS={{OMP_NUM_THREADS}}

# --- site specific setup (edit me) --------------------------------
# module load openmpi
# module load quantum-espresso
# source ~/.venv/ezcalenv312/bin/activate

echo "host      : $(hostname)"
echo "workdir   : $(pwd)"
echo "started   : $(date)"

set -o pipefail

# --- generated commands -------------------------------------------
{{COMMANDS}}
status=$?

echo "finished  : $(date)"
exit $status
