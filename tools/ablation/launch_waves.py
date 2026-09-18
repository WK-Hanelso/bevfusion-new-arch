#!/usr/bin/env python3
"""Launch the two-stage ablation matrix on one eight-GPU node."""

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Sequence

try:
    from .experiments import EXPERIMENTS, Experiment, select_experiments
except ImportError:  # Direct execution from the repository root.
    from experiments import EXPERIMENTS, Experiment, select_experiments


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = ROOT / "experiments/runs"
CONDA_ENV = "bevfusion-b200"
FAILURE_RE = re.compile(
    r"(?i)(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])|CUDA error|"
    r"CUDA out of memory|out of memory|CUBLAS_STATUS|CUDNN_STATUS"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def shell_command(command: Sequence[str], cuda_devices: Optional[str] = None) -> str:
    rendered = shlex.join([str(part) for part in command])
    if cuda_devices is not None:
        return f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)} {rendered}"
    return rendered


def git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return revision


def normalize_dataroot(value: str) -> str:
    # Existing recursive configs concatenate dataset_root with filenames.
    return value if value.endswith(os.sep) else value + os.sep


def build_command(
    experiment: Experiment,
    run_dir: Path,
    epochs: int,
    gpus_per_job: int,
    dataroot: str,
    seed: int,
    master_port: int,
    load_from_dsvt: Optional[str],
) -> List[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "torchrun",
        f"--master_port={master_port}",
        f"--nproc_per_node={gpus_per_job}",
        "tools/train_torchrun.py",
        str(experiment.config_path),
        "--run-dir",
        str(run_dir),
        "--max_epochs",
        str(epochs),
        "--dataset_root",
        normalize_dataroot(dataroot),
        "--seed",
        str(seed),
        "--fp16",
        "None",
        "--find_unused_parameters",
        "True",
        "--checkpoint_config.out_dir",
        str(run_dir / "checkpoints"),
    ]
    if experiment.uses_dsvt and load_from_dsvt:
        command.extend(["--load_from", load_from_dsvt])
    return command


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _capture_command(path: Path, command: Sequence[str]) -> None:
    lines = [f"$ {shell_command(command)}\n"]
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        lines.append(f"exit_code={completed.returncode}\n")
        lines.append(completed.stdout)
    except (OSError, subprocess.SubprocessError) as error:
        lines.append(f"capture_error={error}\n")
    path.write_text("".join(lines))


def capture_phase_environment(phase_dir: Path) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    _capture_command(
        phase_dir / "env_nvidia_smi.txt",
        ["conda", "run", "-n", CONDA_ENV, "nvidia-smi"],
    )
    _capture_command(
        phase_dir / "env_nvcc.txt",
        ["conda", "run", "-n", CONDA_ENV, "nvcc", "--version"],
    )
    _capture_command(
        phase_dir / "env_pip_freeze.txt",
        ["conda", "run", "-n", CONDA_ENV, "python", "-m", "pip", "freeze"],
    )


def _failure_matches(log_path: Path) -> List[str]:
    if not log_path.is_file():
        return []
    matches = []
    for line_number, line in enumerate(
        log_path.read_text(errors="replace").splitlines(), start=1
    ):
        if FAILURE_RE.search(line):
            matches.append(f"line {line_number}: {line.strip()[:500]}")
    return matches


def _copy_resolved_config(run_dir: Path) -> bool:
    source = run_dir / "configs.yaml"
    target = run_dir / "config.yaml"
    if source.is_file():
        shutil.copyfile(str(source), str(target))
        return True
    return False


@dataclass
class RunningJob:
    experiment: Experiment
    slot: int
    wave: int
    devices: str
    run_dir: Path
    command: List[str]
    process: Optional[subprocess.Popen]
    log_handle: Optional[IO[str]]
    started_at: str
    started_monotonic: float
    metadata: dict
    startup_error: Optional[str] = None
    scan_offset: int = 0
    scan_buffer: str = ""
    live_failures: List[str] = field(default_factory=list)


def _start_job(
    experiment: Experiment,
    slot: int,
    wave: int,
    devices: str,
    command: List[str],
    run_dir: Path,
    phase: str,
    epochs: int,
    seed: int,
    revision: str,
) -> RunningJob:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    rendered = shell_command(command, devices)
    (run_dir / "command.txt").write_text(rendered + "\n")
    started_at = utc_now()
    env_lines = [
        f"started_at_utc={started_at}",
        f"hostname={platform.node()}",
        f"platform={platform.platform()}",
        f"git_commit={revision}",
        f"phase={phase}",
        f"experiment_id={experiment.experiment_id}",
        f"bits={experiment.bits}",
        f"seed={seed}",
        f"epochs={epochs}",
        f"CUDA_VISIBLE_DEVICES={devices}",
        f"conda_env={CONDA_ENV}",
    ]
    (run_dir / "env.txt").write_text("\n".join(env_lines) + "\n")
    metadata = {
        "bits": experiment.bits,
        "command": rendered,
        "config": str(experiment.config_path),
        "epochs": epochs,
        "experiment_id": experiment.experiment_id,
        "git_commit": revision,
        "phase": phase,
        "seed": seed,
        "slot": slot,
        "started_at_utc": started_at,
        "status": "running",
        "wave": wave,
    }
    _write_json(run_dir / "metrics.json", metadata)

    log_handle = (run_dir / "train.log").open("w")
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = devices
    process = None
    startup_error = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=child_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        startup_error = str(error)
        log_handle.write(f"launcher startup error: {error}\n")
        log_handle.flush()

    return RunningJob(
        experiment=experiment,
        slot=slot,
        wave=wave,
        devices=devices,
        run_dir=run_dir,
        command=command,
        process=process,
        log_handle=log_handle,
        started_at=started_at,
        started_monotonic=time.monotonic(),
        metadata=metadata,
        startup_error=startup_error,
    )


def _finish_job(job: RunningJob) -> bool:
    if job.process is None:
        return_code = 127
    else:
        return_code = job.process.wait()
    if job.log_handle is not None:
        job.log_handle.close()

    wall_time = time.monotonic() - job.started_monotonic
    failures = _failure_matches(job.run_dir / "train.log")
    if job.startup_error:
        failures.insert(0, f"startup: {job.startup_error}")
    if return_code != 0:
        failures.insert(0, f"process exit code {return_code}")
    config_copied = _copy_resolved_config(job.run_dir)
    if not config_copied:
        failures.append("resolved config configs.yaml was not produced")

    status = "failed" if failures else "completed"
    job.metadata.update(
        {
            "ended_at_utc": utc_now(),
            "failure_matches": failures,
            "return_code": return_code,
            "status": status,
            "wall_time_seconds": round(wall_time, 3),
        }
    )
    _write_json(job.run_dir / "metrics.json", job.metadata)
    print(
        f"[{status.upper()}] wave={job.wave} slot={job.slot} "
        f"id={job.experiment.experiment_id} rc={return_code} "
        f"wall={wall_time:.1f}s",
        flush=True,
    )
    return status == "completed"


def _scan_new_log(job: RunningJob, final: bool = False) -> List[str]:
    path = job.run_dir / "train.log"
    if not path.is_file():
        return []
    with path.open(errors="replace") as handle:
        handle.seek(job.scan_offset)
        chunk = handle.read()
        job.scan_offset = handle.tell()
    text = job.scan_buffer + chunk
    if final:
        lines = text.splitlines()
        job.scan_buffer = ""
    else:
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            job.scan_buffer = lines.pop()
        else:
            job.scan_buffer = ""
    matches = []
    for line in lines:
        if FAILURE_RE.search(line):
            matches.append(line.strip()[:500])
    return matches


def _monitor_wave(jobs: Sequence[RunningJob]) -> None:
    """Stop only the affected slot when its log reports a fatal condition."""

    active = [job for job in jobs if job.process is not None]
    while active:
        remaining = []
        for job in active:
            new_failures = _scan_new_log(job)
            if new_failures and job.process is not None and job.process.poll() is None:
                job.live_failures.extend(new_failures)
                print(
                    f"[STOP] wave={job.wave} slot={job.slot} "
                    f"id={job.experiment.experiment_id} fatal log pattern: "
                    f"{new_failures[0]}",
                    flush=True,
                )
                job.process.terminate()
            if job.process is not None and job.process.poll() is None:
                remaining.append(job)
            else:
                job.live_failures.extend(_scan_new_log(job, final=True))
        active = remaining
        if active:
            time.sleep(1.0)


def _is_completed(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        return json.loads(metrics_path.read_text()).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _chunks(values: Sequence[Experiment], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("screening", "final"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gpus-per-job", type=int, choices=(2, 4))
    parser.add_argument("--parallel", type=int, choices=(2, 4))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--dataroot", default="data/nuscenes/")
    parser.add_argument("--load-from-dsvt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--master-port", type=int, default=29500, help=argparse.SUPPRESS
    )
    return parser


def resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.epochs is None:
        args.epochs = 6 if args.phase == "screening" else 20
    if args.gpus_per_job is None:
        args.gpus_per_job = 2 if args.phase == "screening" else 4
    if args.parallel is None:
        args.parallel = 4 if args.phase == "screening" else 2
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.gpus_per_job * args.parallel > 8:
        parser.error("--gpus-per-job * --parallel cannot exceed 8 GPUs")
    if args.ids is None:
        if args.phase == "final":
            parser.error("--ids is required for final phase (pass aggregate Top-K IDs)")
        selected = EXPERIMENTS
    else:
        try:
            selected = select_experiments(args.ids)
        except ValueError as error:
            parser.error(str(error))
    return args, tuple(selected)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args, selected = resolve_args(parser, parser.parse_args(argv))
    phase_dir = args.runs_root.resolve() / args.phase

    pending = []
    for experiment in selected:
        run_dir = phase_dir / experiment.experiment_id
        if args.resume and _is_completed(run_dir):
            print(f"[SKIP] completed id={experiment.experiment_id}")
        else:
            pending.append(experiment)

    print(
        f"phase={args.phase} jobs={len(pending)} epochs={args.epochs} "
        f"gpus_per_job={args.gpus_per_job} parallel={args.parallel}"
    )
    for wave_number, wave in enumerate(_chunks(pending, args.parallel), start=1):
        for slot, experiment in enumerate(wave):
            devices = ",".join(
                str(slot * args.gpus_per_job + offset)
                for offset in range(args.gpus_per_job)
            )
            run_dir = phase_dir / experiment.experiment_id
            command = build_command(
                experiment=experiment,
                run_dir=run_dir,
                epochs=args.epochs,
                gpus_per_job=args.gpus_per_job,
                dataroot=args.dataroot,
                seed=args.seed,
                master_port=args.master_port + slot,
                load_from_dsvt=args.load_from_dsvt,
            )
            print(
                f"[PLAN] wave={wave_number} slot={slot} "
                f"id={experiment.experiment_id} bits={experiment.bits} "
                f"gpus={devices}\n  {shell_command(command, devices)}"
            )

    if args.dry_run:
        return 0
    capture_phase_environment(phase_dir)
    revision = git_revision()
    if not pending:
        return 0
    all_completed = True
    active_jobs: List[RunningJob] = []
    try:
        for wave_number, wave in enumerate(_chunks(pending, args.parallel), start=1):
            active_jobs = []
            for slot, experiment in enumerate(wave):
                devices = ",".join(
                    str(slot * args.gpus_per_job + offset)
                    for offset in range(args.gpus_per_job)
                )
                run_dir = phase_dir / experiment.experiment_id
                command = build_command(
                    experiment,
                    run_dir,
                    args.epochs,
                    args.gpus_per_job,
                    args.dataroot,
                    args.seed,
                    args.master_port + slot,
                    args.load_from_dsvt,
                )
                active_jobs.append(
                    _start_job(
                        experiment,
                        slot,
                        wave_number,
                        devices,
                        command,
                        run_dir,
                        args.phase,
                        args.epochs,
                        args.seed,
                        revision,
                    )
                )
            _monitor_wave(active_jobs)
            for job in active_jobs:
                all_completed = _finish_job(job) and all_completed
            active_jobs = []
    except KeyboardInterrupt:
        for job in active_jobs:
            if job.process is not None and job.process.poll() is None:
                job.process.terminate()
        for job in active_jobs:
            if job.process is not None:
                job.process.wait()
            if job.log_handle is not None and not job.log_handle.closed:
                job.log_handle.close()
        print("launcher interrupted; active jobs were terminated", file=sys.stderr)
        return 130
    return 0 if all_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
