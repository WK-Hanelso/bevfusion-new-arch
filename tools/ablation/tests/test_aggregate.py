"""Fixture-based tests for mmdet3d ablation log aggregation."""

import json
from pathlib import Path
import sys

ABLATION_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABLATION_TOOLS))

from aggregate import (  # noqa: E402
    collect_rows,
    main,
    parse_log,
    select_top_k,
)


FIXTURE = Path(__file__).parent / "fixtures/mmdet3d_train.log"


def test_parse_mmdet3d_epoch_metrics_fixture():
    parsed = parse_log(FIXTURE)
    assert len(parsed.epochs) == 2
    first, second = parsed.epochs
    assert first.epoch == 1
    assert first.mAP == 0.401
    assert first.NDS == 0.451
    assert first.s_per_iter == 1.1
    assert first.peak_vram_mb == 11000
    # heatmap loss must not be mistaken for the mAP metric.
    assert second.mAP == 0.42
    assert second.NDS == 0.47
    assert second.nan_detected
    assert parsed.nan_detected
    assert parsed.wall_time_seconds == 12.5


def _make_run(runs_root, experiment_id, nds_offset=0.0):
    run_dir = runs_root / "screening" / experiment_id
    (run_dir / "checkpoints").mkdir(parents=True)
    text = FIXTURE.read_text().replace("0.4700", f"{0.47 + nds_offset:.4f}")
    (run_dir / "train.log").write_text(text)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "wall_time_seconds": 99.5,
                "git_commit": "abc123",
                "seed": 7,
            }
        )
    )
    (run_dir / "checkpoints" / "epoch_2.pth").touch()


def test_collect_csv_markdown_and_forced_top_k(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "B0", 0.00)
    _make_run(runs_root, "A1", 0.08)
    _make_run(runs_root, "A2", 0.04)
    _make_run(runs_root, "FINAL", 0.01)
    rows = collect_rows(runs_root / "screening")
    assert len(rows) == 8
    assert all(row["peak_vram_mb"] == 12288 for row in rows)
    best = select_top_k(rows, "NDS", 3, force_baselines=True)
    assert [row["experiment_id"] for row in best] == ["B0", "FINAL", "A1"]

    output_dir = tmp_path / "results"
    assert (
        main(
            [
                "--phase",
                "screening",
                "--runs-root",
                str(runs_root),
                "--output-dir",
                str(output_dir),
                "--top-k",
                "3",
                "--by",
                "NDS",
                "--force-include-b0-final",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "TOP_K_IDS=B0 FINAL A1" in output
    csv_text = (output_dir / "screening_summary.csv").read_text()
    markdown = (output_dir / "screening_summary.md").read_text()
    assert "git_commit,seed" in csv_text
    assert "ΔmAP" in markdown
    assert "Top-3 by NDS: B0, FINAL, A1" in markdown
