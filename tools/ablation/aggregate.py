#!/usr/bin/env python3
"""Aggregate mmdet3d ablation logs into CSV and Markdown summaries."""

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from .experiments import BY_ID, EXPERIMENTS
except ImportError:  # Direct execution from the repository root.
    from experiments import BY_ID, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[2]
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
EPOCH_RE = re.compile(r"Epoch(?:\(val\))?\s*\[(\d+)\]", re.IGNORECASE)
JSON_EPOCH_RE = re.compile(r'["\']epoch["\']\s*:\s*(\d+)', re.IGNORECASE)
TIME_RE = re.compile(rf"(?<!data_)\btime[\"']?\s*:\s*({FLOAT})", re.IGNORECASE)
S_PER_ITER_RE = re.compile(rf"({FLOAT})\s*s\s*/\s*iter", re.IGNORECASE)
MEMORY_RE = re.compile(
    rf"\b(?:memory|peak[ _-]?(?:vram|memory))[\"']?\s*:\s*({FLOAT})\s*(GiB|GB|MiB|MB)?",
    re.IGNORECASE,
)
NAN_RE = re.compile(r"(?i)(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])")
TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)")

METRIC_PATTERNS = {
    "mAP": re.compile(
        rf"(?<![A-Za-z0-9_])[\"']?(?:object/)?(?:mAP|map)[\"']?\s*[:=]\s*({FLOAT})",
        re.IGNORECASE,
    ),
    "NDS": re.compile(
        rf"(?<![A-Za-z0-9_])[\"']?(?:object/)?nds[\"']?\s*[:=]\s*({FLOAT})",
        re.IGNORECASE,
    ),
}
for _name in ("mATE", "mASE", "mAOE", "mAVE", "mAAE"):
    METRIC_PATTERNS[_name] = re.compile(
        rf"(?<![A-Za-z0-9_])[\"']?(?:object/)?{_name}[\"']?\s*[:=]\s*({FLOAT})",
        re.IGNORECASE,
    )


def _float(value: str) -> float:
    return float(value)


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return value


@dataclass
class EpochMetrics:
    epoch: Optional[int]
    mAP: Optional[float] = None
    NDS: Optional[float] = None
    mATE: Optional[float] = None
    mASE: Optional[float] = None
    mAOE: Optional[float] = None
    mAVE: Optional[float] = None
    mAAE: Optional[float] = None
    s_per_iter: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    nan_detected: bool = False
    _times: List[float] = field(default_factory=list, repr=False)

    def finish(self) -> None:
        finite_times = [value for value in self._times if math.isfinite(value)]
        if finite_times:
            self.s_per_iter = statistics.mean(finite_times)


@dataclass
class ParsedLog:
    epochs: List[EpochMetrics]
    nan_detected: bool
    wall_time_seconds: Optional[float]


def _parse_timestamp(value: str) -> Optional[dt.datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def parse_log_text(text: str) -> ParsedLog:
    """Parse standard MMCV text logs and their JSON-like metric lines."""

    by_epoch: Dict[int, EpochMetrics] = {}
    current_epoch: Optional[int] = None
    global_nan = False
    first_timestamp = None
    last_timestamp = None

    def record(epoch: int) -> EpochMetrics:
        return by_epoch.setdefault(epoch, EpochMetrics(epoch=epoch))

    for line in text.splitlines():
        timestamp_match = TIMESTAMP_RE.search(line)
        if timestamp_match:
            timestamp = _parse_timestamp(timestamp_match.group(1))
            if timestamp is not None:
                first_timestamp = first_timestamp or timestamp
                last_timestamp = timestamp

        epoch_match = EPOCH_RE.search(line) or JSON_EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        has_nan = bool(NAN_RE.search(line))
        global_nan = global_nan or has_nan
        metric_values = {}
        for name, pattern in METRIC_PATTERNS.items():
            matches = pattern.findall(line)
            if matches:
                metric_values[name] = _float(matches[-1])

        time_matches = S_PER_ITER_RE.findall(line)
        if not time_matches and "Epoch" in line:
            time_matches = TIME_RE.findall(line)
        memory_matches = MEMORY_RE.findall(line)

        if current_epoch is not None and (
            metric_values or time_matches or memory_matches or has_nan
        ):
            item = record(current_epoch)
            for name, value in metric_values.items():
                setattr(item, name, value)
                if not math.isfinite(value):
                    item.nan_detected = True
            for value in time_matches:
                item._times.append(_float(value))
            for value, unit in memory_matches:
                memory_mb = _float(value)
                if unit.lower() in ("gib", "gb"):
                    memory_mb *= 1024.0
                if item.peak_vram_mb is None or memory_mb > item.peak_vram_mb:
                    item.peak_vram_mb = memory_mb
            item.nan_detected = item.nan_detected or has_nan

    for item in by_epoch.values():
        item.finish()
    wall_time = None
    if first_timestamp is not None and last_timestamp is not None:
        wall_time = max(0.0, (last_timestamp - first_timestamp).total_seconds())
    return ParsedLog(
        epochs=[by_epoch[key] for key in sorted(by_epoch)],
        nan_detected=global_nan,
        wall_time_seconds=wall_time,
    )


def parse_log(path: Path) -> ParsedLog:
    return parse_log_text(path.read_text(errors="replace"))


def _read_metadata(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _checkpoint_for_epoch(run_dir: Path, epoch: Optional[int]) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    if epoch is not None:
        exact = checkpoint_dir / f"epoch_{epoch}.pth"
        if exact.exists():
            return str(exact)
        candidates = sorted(checkpoint_dir.rglob(f"*epoch_{epoch}.pth"))
        if candidates:
            preferred = [path for path in candidates if "best" in str(path).lower()]
            return str((preferred or candidates)[0])
    latest = checkpoint_dir / "latest.pth"
    return str(latest) if latest.exists() else ""


CSV_FIELDS = [
    "experiment_id",
    "bits",
    "config",
    "status",
    "epoch",
    "mAP",
    "NDS",
    "mATE",
    "mASE",
    "mAOE",
    "mAVE",
    "mAAE",
    "s_per_iter",
    "peak_vram_mb",
    "wall_time_seconds",
    "nan_detected",
    "ckpt_path",
    "git_commit",
    "seed",
]


def collect_rows(phase_dir: Path) -> List[dict]:
    rows = []
    for experiment in EXPERIMENTS:
        run_dir = phase_dir / experiment.experiment_id
        if not run_dir.is_dir():
            continue
        metadata = _read_metadata(run_dir)
        log_path = run_dir / "train.log"
        parsed = (
            parse_log(log_path) if log_path.is_file() else ParsedLog([], False, None)
        )
        wall_time = metadata.get("wall_time_seconds", parsed.wall_time_seconds)
        epoch_items = parsed.epochs or [EpochMetrics(epoch=None)]
        run_peak_vram = max(
            (
                item.peak_vram_mb
                for item in parsed.epochs
                if item.peak_vram_mb is not None
            ),
            default=None,
        )
        failure_nan = any(
            NAN_RE.search(str(value))
            for value in metadata.get("failure_matches", [])
        )
        for item in epoch_items:
            row = {
                "experiment_id": experiment.experiment_id,
                "bits": experiment.bits,
                "config": str(experiment.config_path),
                "status": metadata.get("status", "unknown"),
                "epoch": item.epoch,
                "mAP": _finite_or_none(item.mAP),
                "NDS": _finite_or_none(item.NDS),
                "mATE": _finite_or_none(item.mATE),
                "mASE": _finite_or_none(item.mASE),
                "mAOE": _finite_or_none(item.mAOE),
                "mAVE": _finite_or_none(item.mAVE),
                "mAAE": _finite_or_none(item.mAAE),
                "s_per_iter": _finite_or_none(item.s_per_iter),
                "peak_vram_mb": _finite_or_none(run_peak_vram),
                "wall_time_seconds": wall_time,
                "nan_detected": bool(
                    parsed.nan_detected or item.nan_detected or failure_nan
                ),
                "ckpt_path": _checkpoint_for_epoch(run_dir, item.epoch),
                "git_commit": metadata.get("git_commit", ""),
                "seed": metadata.get("seed", ""),
            }
            rows.append(row)
    return rows


def best_rows(rows: Iterable[dict], by: str) -> List[dict]:
    best: Dict[str, dict] = {}
    for row in rows:
        value = row.get(by)
        if value is None:
            continue
        old = best.get(row["experiment_id"])
        if old is None or value > old[by]:
            best[row["experiment_id"]] = row
    order = {item.experiment_id: index for index, item in enumerate(EXPERIMENTS)}
    return sorted(best.values(), key=lambda row: order[row["experiment_id"]])


def select_top_k(
    rows: Iterable[dict], by: str, top_k: int, force_baselines: bool
) -> List[dict]:
    ranked = sorted(
        best_rows(rows, by),
        key=lambda row: (-row[by], list(BY_ID).index(row["experiment_id"])),
    )
    if not force_baselines:
        return ranked[:top_k]
    forced_ids = ("B0", "FINAL")
    forced = [
        row
        for name in forced_ids
        for row in ranked
        if row["experiment_id"] == name
    ]
    selected = forced[:top_k]
    for row in ranked:
        if len(selected) >= top_k:
            break
        if row["experiment_id"] not in {item["experiment_id"] for item in selected}:
            selected.append(row)
    return selected


def _format(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(value, baseline, digits: int = 4) -> str:
    if value is None or baseline is None:
        return "—"
    return f"{value - baseline:+.{digits}f}"


def render_markdown(rows: List[dict], by: str, selected: List[dict]) -> str:
    best = best_rows(rows, by)
    baseline = next((row for row in best if row["experiment_id"] == "B0"), None)
    lines = [
        f"# Ablation summary (best epoch by {by})",
        "",
        "| ID | bits | epoch | mAP | NDS | ΔmAP | ΔNDS | s/iter | Δs/iter | "
        "peak VRAM MB | ΔVRAM MB | NaN | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best:
        checkpoint = row["ckpt_path"] or "—"
        base_map = baseline["mAP"] if baseline else None
        base_nds = baseline["NDS"] if baseline else None
        base_time = baseline["s_per_iter"] if baseline else None
        base_vram = baseline["peak_vram_mb"] if baseline else None
        lines.append(
            "| {experiment_id} | {bits} | {epoch} | {mAP} | {NDS} | {dmap} | "
            "{dnds} | {time} | {dtime} | {vram} | {dvram} | {nan} | `{ckpt}` |".format(
                experiment_id=row["experiment_id"],
                bits=row["bits"],
                epoch=_format(row["epoch"], 0),
                mAP=_format(row["mAP"]),
                NDS=_format(row["NDS"]),
                dmap=_delta(row["mAP"], base_map),
                dnds=_delta(row["NDS"], base_nds),
                time=_format(row["s_per_iter"]),
                dtime=_delta(row["s_per_iter"], base_time),
                vram=_format(row["peak_vram_mb"], 1),
                dvram=_delta(row["peak_vram_mb"], base_vram, 1),
                nan=_format(row["nan_detected"]),
                ckpt=checkpoint,
            )
        )
    lines.extend(
        [
            "",
            f"Top-{len(selected)} by {by}: "
            + (", ".join(row["experiment_id"] for row in selected) or "none"),
            "",
        ]
    )
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("screening", "final"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--by", choices=("NDS", "mAP"), default="NDS")
    parser.add_argument("--force-include-b0-final", action="store_true")
    parser.add_argument(
        "--runs-root", type=Path, default=ROOT / "experiments/runs"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    phase_dir = args.runs_root.resolve() / args.phase
    rows = collect_rows(phase_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "screening_summary" if args.phase == "screening" else "final_ablation"
    csv_path = args.output_dir / f"{stem}.csv"
    markdown_path = args.output_dir / f"{stem}.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    selected = select_top_k(
        rows, args.by, args.top_k, args.force_include_b0_final
    )
    markdown_path.write_text(render_markdown(rows, args.by, selected))
    print(f"wrote {csv_path} ({len(rows)} epoch rows)")
    print(f"wrote {markdown_path}")
    print("TOP_K_IDS=" + " ".join(row["experiment_id"] for row in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
