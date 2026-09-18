"""CPU-only tests for the ablation registry and launcher planning."""

import json
from pathlib import Path
import sys

ABLATION_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABLATION_TOOLS))

from experiments import (  # noqa: E402
    BY_BITS,
    BY_ID,
    EXPERIMENTS,
    LEGACY_CONFIG_ALIASES,
)
from launch_waves import build_command, main  # noqa: E402


def test_registry_is_bijective_and_keeps_legacy_aliases():
    assert len(EXPERIMENTS) == len(BY_ID) == len(BY_BITS) == 16
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
    assert "wave=4 slot=3 id=FINAL" in output
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
