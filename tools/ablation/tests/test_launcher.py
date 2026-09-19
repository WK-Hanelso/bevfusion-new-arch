"""CPU-only tests for the ablation registry and launcher planning."""

import json
import os
from pathlib import Path
import sys

ABLATION_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABLATION_TOOLS))

from experiments import (  # noqa: E402
    STAGE_EXPERIMENTS,
    BY_BITS,
    BY_ID,
    EXPERIMENTS,
    LEGACY_CONFIG_ALIASES,
)
import launch_waves  # noqa: E402
from launch_waves import build_command, main, make_gpu_groups  # noqa: E402


def test_registry_is_bijective_and_keeps_legacy_aliases():
    assert len(EXPERIMENTS) == len(BY_BITS) == 16 and len(BY_ID) == 16 + len(STAGE_EXPERIMENTS)
    assert BY_ID["B0"].bits == "0000"
    assert BY_ID["FINAL"].bits == "1111"
    assert BY_ID["A3"].bits == "0001"
    assert BY_ID["A4"].bits == "0010"
    assert LEGACY_CONFIG_ALIASES["c2_dsvt_widthformer.yaml"] == (
        "dsvt1_wf1_gf0_dal0.yaml"
    )


def test_dsvt_checkpoint_is_only_added_to_dsvt_commands(tmp_path):
    common = dict(
        run_dir=tmp_path / "run",
        epochs=6,
        gpus_per_job=2,
        dataroot="/data/nuscenes",
        seed=7,
        master_port=29500,
        load_from_dsvt="/weights/dsvt.pth",
    )
    baseline = build_command(BY_ID["B0"], **common)
    dsvt = build_command(BY_ID["A1"], **common)
    assert "--load_from" not in baseline
    assert dsvt[-2:] == ["--load_from", "/weights/dsvt.pth"]
    assert baseline[baseline.index("--max_epochs") + 1] == "6"
    assert baseline[baseline.index("--dataset_root") + 1].endswith("/")


def test_screening_dry_run_plans_all_16_without_writing(tmp_path, capsys):
    assert (
        main(["--phase", "screening", "--dry-run", "--runs-root", str(tmp_path)])
        == 0
    )
    output = capsys.readouterr().out
    assert output.count("[PLAN]") == 16
    assert "wave=16 slot=3 id=FINAL" in output
    assert "gpus=6,7" in output
    assert "--nproc_per_node=2" in output
    assert "--max_epochs 6" in output
    assert not list(tmp_path.iterdir())


def test_resume_skips_only_completed_runs(tmp_path, capsys):
    completed = tmp_path / "screening" / "B0"
    completed.mkdir(parents=True)
    (completed / "metrics.json").write_text(json.dumps({"status": "completed"}))
    assert (
        main(
            [
                "--phase",
                "screening",
                "--ids",
                "B0",
                "A1",
                "--resume",
                "--dry-run",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "[SKIP] completed id=B0" in output
    assert output.count("[PLAN]") == 1
    assert "id=A1" in output


def test_resume_skips_live_running_pid_and_requeues_stale_pid(tmp_path, capsys):
    live = tmp_path / "screening" / "B0"
    live.mkdir(parents=True)
    (live / "metrics.json").write_text(json.dumps({"status": "running"}))
    (live / "launcher.pid").write_text(f"{os.getpid()}\n")
    stale = tmp_path / "screening" / "A1"
    stale.mkdir(parents=True)
    (stale / "metrics.json").write_text(json.dumps({"status": "running"}))
    (stale / "launcher.pid").write_text("999999999\n")

    assert (
        main(
            [
                "--phase",
                "screening",
                "--ids",
                "B0",
                "A1",
                "--resume",
                "--dry-run",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"[SKIP] running id=B0 pid={os.getpid()}" in output
    assert output.count("[PLAN]") == 1
    assert "id=A1" in output


def test_custom_gpu_list_is_split_into_three_groups():
    assert make_gpu_groups("0,1,2,3,6,7", 2) == ("0,1", "2,3", "6,7")


def test_custom_gpu_dry_run_uses_pool_plan_format(tmp_path, capsys):
    assert (
        main(
            [
                "--phase",
                "screening",
                "--ids",
                "B0",
                "A1",
                "A2",
                "A3",
                "--gpus",
                "0,1,2,3,6,7",
                "--dry-run",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "parallel=3" in output
    assert "[PLAN] wave=1 slot=0 id=B0 bits=0000 gpus=0,1" in output
    assert "[PLAN] wave=2 slot=1 id=A1 bits=1000 gpus=2,3" in output
    assert "[PLAN] wave=3 slot=2 id=A2 bits=0100 gpus=6,7" in output
    assert "[PLAN] wave=4 slot=0 id=A3 bits=0001 gpus=0,1" in output
    assert output.count("[PLAN]") == 4
    assert not list(tmp_path.iterdir())


def test_failed_job_refills_its_slot_without_a_wave_barrier(
    tmp_path, capsys, monkeypatch
):
    fake_runner = tmp_path / "fake_runner.py"
    fake_runner.write_text(
        """\
import pathlib
import sys
import time

experiment_id = sys.argv[1]
run_dir = pathlib.Path(sys.argv[2])
if experiment_id == "B0":
    raise SystemExit(9)
time.sleep(0.5)
(run_dir / "configs.yaml").write_text("resolved: true\\n")
"""
    )

    def fake_build_command(experiment, run_dir, *args, **kwargs):
        return [sys.executable, str(fake_runner), experiment.experiment_id, str(run_dir)]

    monkeypatch.setattr(launch_waves, "build_command", fake_build_command)
    monkeypatch.setattr(launch_waves, "capture_phase_environment", lambda path: None)
    monkeypatch.setattr(launch_waves, "git_revision", lambda: "test-revision")

    result = main(
        [
            "--phase",
            "screening",
            "--ids",
            "B0",
            "A1",
            "A2",
            "A3",
            "A4",
            "--gpus",
            "0,1,2,3,6,7",
            "--runs-root",
            str(tmp_path / "runs"),
            "--skip-gate",
        ]
    )
    assert result == 1
    output = capsys.readouterr().out
    failed = output.index("[FAILED] wave=1 slot=0 id=B0")
    refill = output.index("[START] slot=0 gpus=0,1 id=A3")
    assert failed < refill
    metrics = json.loads(
        (tmp_path / "runs" / "screening" / "A3" / "metrics.json").read_text()
    )
    assert metrics["wave"] == 4
    assert metrics["slot"] == 0
    assert metrics["status"] == "completed"


def test_launch_is_refused_without_gate_marker(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(launch_waves, "git_revision", lambda: "no-such-revision")
    monkeypatch.setattr(launch_waves, "capture_phase_environment", lambda path: None)
    result = main(
        ["--phase", "screening", "--ids", "B0", "--runs-root", str(tmp_path / "runs")]
    )
    assert result == 2
    assert "[GATE] refusing to launch" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_jobs_per_gpu_two_duplicates_groups_and_ports(tmp_path, capsys):
    assert (
        main(
            ["--phase", "screening", "--jobs-per-gpu", "2", "--dry-run",
             "--runs-root", str(tmp_path)]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "parallel=8" in output
    assert "slot=4 id=A2" not in output  # first 8 launches fill 8 slots in order
    assert "wave=5 slot=4 id=A4" in output and "gpus=0,1" in output
    assert "--master_port=29504" in output


def test_failure_regex_ignores_eval_table_nan():
    regex = launch_waves.FAILURE_RE
    assert not regex.search("traffic_cone  0.136  1.061  0.450  nan  nan  nan")
    assert not regex.search("object/barrier_vel_err: nan, object/nds: 0.0689")
    assert regex.search("loss: nan, grad_norm: 12.0")
    assert regex.search("loss/object/loss_bbox: inf")
    assert regex.search("grad_norm: nan")
    assert regex.search("RuntimeError: CUDA error: device-side assert triggered")


def test_one_gpu_job_uses_batch16_with_gradient_accumulation(tmp_path, capsys):
    assert main(["--phase", "screening", "--ids", "B0", "--gpus-per-job", "1",
                 "--dry-run", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "--data.samples_per_gpu 16" in out
    assert "--optimizer_config.type GradientCumulativeOptimizerHook" in out
    assert "--optimizer_config.cumulative_iters 2" in out
    capsys.readouterr()
    assert main(["--phase", "screening", "--ids", "B0", "--gpus-per-job", "2",
                 "--dry-run", "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "--data.samples_per_gpu 16" in out and "cumulative_iters" not in out


def test_spconv_v2_lifts_per_gpu_batch_cap(monkeypatch):
    monkeypatch.setenv("BEVFUSION_SPCONV", "v2")
    assert launch_waves.batch_plan(1) == (32, 1)
    monkeypatch.setenv("BEVFUSION_SPCONV", "legacy")
    assert launch_waves.batch_plan(1) == (16, 2)
