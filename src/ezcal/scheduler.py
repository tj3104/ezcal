"""Execution back-ends: run locally with mpirun, or submit with qsub.

A *stage* is an ordered list of :class:`Command` objects that must run one
after another (for example ``pw.x`` then ``dos.x``).  The scheduler is the
only place that knows how a stage becomes actual work, which keeps the
engines and the workflows free of any queueing-system detail.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

TEMPLATE_DIR = Path(__file__).with_name("templates")


class SchedulerError(RuntimeError):
    pass


@dataclass
class Command:
    """One executable invocation."""

    executable: str
    args: list[str] = field(default_factory=list)
    stdout: Path | None = None
    stdin: Path | None = None
    parallel: bool = True          # wrap with mpirun?
    label: str = ""

    def render(self, mpirun: str | None = None, nproc: int = 1,
               mpirun_flags: Sequence[str] = ()) -> list[str]:
        parts: list[str] = []
        if self.parallel and mpirun and nproc > 1:
            parts += [mpirun, "-np", str(nproc), *mpirun_flags]
        parts += [self.executable, *self.args]
        return parts

    def shell(self, mpirun: str | None = None, nproc: int = 1,
              mpirun_flags: Sequence[str] = ()) -> str:
        line = " ".join(shlex.quote(p) for p in self.render(mpirun, nproc, mpirun_flags))
        if self.stdin:
            line += f" < {shlex.quote(str(self.stdin))}"
        if self.stdout:
            line += f" > {shlex.quote(str(self.stdout))}"
        return line


@dataclass
class Stage:
    name: str
    commands: list[Command]
    workdir: Path
    env: dict[str, str] = field(default_factory=dict)   # extra environment variables


@dataclass
class JobResult:
    ok: bool
    returncode: int = 0
    job_id: str | None = None
    submitted_only: bool = False
    elapsed: float = 0.0
    log: list[str] = field(default_factory=list)
    script: Path | None = None


class Scheduler:
    """Base class."""

    name = "base"

    def __init__(self, config) -> None:
        self.config = config
        self.nproc = int(config.get("run.nproc", 1) or 1)
        self.mpirun = config.get("run.mpirun", "mpirun")
        self.mpirun_flags = list(config.get("run.mpirun_flags", []) or [])
        self.omp = int(config.get("run.omp_num_threads", 1) or 1)
        self.dry_run = bool(config.get("run.dry_run", False))

    def blocking(self) -> bool:
        raise NotImplementedError

    def execute(self, stage: Stage) -> JobResult:
        raise NotImplementedError

    def write_command_script(self, stage: Stage) -> JobResult:
        """Dry run: record the exact command line without executing it."""
        lines = [f"export {key}={shlex.quote(str(value))}"
                 for key, value in sorted(stage.env.items())]
        lines += [cmd.shell(self.mpirun, self.nproc, self.mpirun_flags)
                  for cmd in stage.commands]
        script = stage.workdir / f"{stage.name}.commands.sh"
        script.write_text("#!/bin/bash\ncd " + str(stage.workdir) + "\n"
                          + "\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        return JobResult(ok=True, submitted_only=True, script=script, log=lines)

    def env(self, stage: "Stage | None" = None) -> dict:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self.omp)
        if stage is not None and stage.env:
            env.update({k: str(v) for k, v in stage.env.items()})
        return env


class LocalScheduler(Scheduler):
    """Run the commands right here, one after another."""

    name = "local"

    def blocking(self) -> bool:
        return True

    def execute(self, stage: Stage) -> JobResult:
        stage.workdir.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return self.write_command_script(stage)
        timeout = self.config.get("run.timeout")
        start = time.time()
        log: list[str] = []

        for cmd in stage.commands:
            argv = cmd.render(self.mpirun, self.nproc, self.mpirun_flags)
            if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
                raise SchedulerError(
                    f"executable not found: {argv[0]}\n"
                    "  set it in qe_config.yaml (qe.bin_dir / qe.commands.*)"
                )
            log.append(cmd.shell(self.mpirun, self.nproc, self.mpirun_flags))
            out_handle = cmd.stdout.open("w", encoding="utf-8") if cmd.stdout else None
            in_handle = cmd.stdin.open("r", encoding="utf-8") if cmd.stdin else None
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(stage.workdir),
                    stdout=out_handle or subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=in_handle,
                    env=self.env(stage),
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SchedulerError(f"{cmd.label or argv[0]} timed out after {timeout}s") from exc
            finally:
                if out_handle:
                    out_handle.close()
                if in_handle:
                    in_handle.close()

            if proc.returncode != 0:
                stderr = (proc.stderr or b"").decode("utf-8", "replace")
                return JobResult(
                    ok=False,
                    returncode=proc.returncode,
                    elapsed=time.time() - start,
                    log=log + [stderr.strip()[-4000:]],
                )
        return JobResult(ok=True, elapsed=time.time() - start, log=log)


class QsubScheduler(Scheduler):
    """Write a job script from ``run_qe.sh`` and submit it with ``qsub``."""

    name = "qsub"

    def __init__(self, config) -> None:
        super().__init__(config)
        qsub = config.section("run").get("qsub", {}) if config.get("run.qsub") else {}
        self.settings = qsub or config.get("run.qsub", {}) or {}
        self.submit_cmd = self.settings.get("submit_cmd", "qsub")
        self.status_cmd = self.settings.get("status_cmd", "qstat")
        self.wait = bool(self.settings.get("wait", False))
        self.poll = float(self.settings.get("poll_interval", 30))

    def blocking(self) -> bool:
        return self.wait

    # -- script generation ----------------------------------------------
    def template_text(self) -> str:
        raw = self.settings.get("script", "run_qe.sh")
        for candidate in (Path(raw).expanduser(), Path.cwd() / raw, TEMPLATE_DIR / "run_qe.sh"):
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        raise SchedulerError(
            f"qsub template {raw!r} not found (looked in ., $PWD and {TEMPLATE_DIR})"
        )

    def render_script(self, stage: Stage) -> str:
        exports = [f"export {key}={shlex.quote(str(value))}"
                   for key, value in sorted(stage.env.items())]
        body = "\n".join(
            exports + [cmd.shell(self.mpirun, self.nproc, self.mpirun_flags)
                       for cmd in stage.commands]
        )
        job_name = f"{self.settings.get('job_prefix', 'ezcal')}_{stage.name}"
        mapping = {
            "JOB_NAME": job_name,
            "NODES": str(self.settings.get("nodes", 1)),
            "PPN": str(self.settings.get("ppn") or self.nproc),
            "NPROC": str(self.nproc),
            "QUEUE": str(self.settings.get("queue") or ""),
            "WALLTIME": str(self.settings.get("walltime", "24:00:00")),
            "WORKDIR": str(stage.workdir),
            "OMP_NUM_THREADS": str(self.omp),
            "MPIRUN": str(self.mpirun),
            "COMMANDS": body,
        }
        text = self.template_text()
        for key, value in mapping.items():
            text = text.replace("{{" + key + "}}", value)
        # a queue directive with an empty value would break the script
        if not mapping["QUEUE"]:
            text = "\n".join(
                line for line in text.splitlines()
                if not re.match(r"^\s*#(PBS -q|SBATCH -p)\s*$", line)
            ) + "\n"
        return text

    # -- submission ------------------------------------------------------
    def execute(self, stage: Stage) -> JobResult:
        stage.workdir.mkdir(parents=True, exist_ok=True)
        script_path = stage.workdir / f"{stage.name}.qsub.sh"
        script_path.write_text(self.render_script(stage), encoding="utf-8")
        script_path.chmod(0o755)

        if self.dry_run:
            return JobResult(ok=True, submitted_only=True, script=script_path,
                             log=[f"dry run: {script_path} written, not submitted"])

        if shutil.which(self.submit_cmd) is None:
            raise SchedulerError(
                f"'{self.submit_cmd}' is not available on this machine.\n"
                f"  The job script was still written to {script_path}\n"
                "  Submit it by hand, or use --scheduler local."
            )

        start = time.time()
        proc = subprocess.run(
            [self.submit_cmd, str(script_path.name)],
            cwd=str(stage.workdir),
            capture_output=True,
            text=True,
            env=self.env(stage),
            check=False,
        )
        if proc.returncode != 0:
            return JobResult(ok=False, returncode=proc.returncode,
                             log=[proc.stdout, proc.stderr], script=script_path)
        job_id = (proc.stdout or "").strip().splitlines()[-1].strip() if proc.stdout else None

        if not self.wait:
            return JobResult(ok=True, job_id=job_id, submitted_only=True,
                             elapsed=time.time() - start, script=script_path,
                             log=[f"submitted {job_id}"])

        self._wait_for(job_id)
        return JobResult(ok=True, job_id=job_id, elapsed=time.time() - start,
                         script=script_path, log=[f"finished {job_id}"])

    def _wait_for(self, job_id: str | None) -> None:
        if not job_id:
            return
        short = job_id.split(".")[0]
        while True:
            time.sleep(self.poll)
            proc = subprocess.run([self.status_cmd, short], capture_output=True,
                                  text=True, check=False)
            if proc.returncode != 0 or short not in (proc.stdout or ""):
                return
            state = re.search(rf"^{re.escape(short)}\S*\s+\S+\s+\S+\s+\S+\s+(\S+)",
                              proc.stdout, re.MULTILINE)
            if state and state.group(1) in {"C", "F"}:
                return


def get_scheduler(config) -> Scheduler:
    name = str(config.get("run.scheduler", "local")).lower()
    if name in {"dryrun", "dry-run", "dry"}:
        config.set("run.dry_run", True)
        name = "local"
    if name in {"local", "direct", "shell"}:
        return LocalScheduler(config)
    if name in {"qsub", "pbs", "torque"}:
        return QsubScheduler(config)
    raise SchedulerError(f"unknown scheduler {name!r} (use 'local' or 'qsub')")
