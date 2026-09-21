"""Canonical experiment registry shared by ablation tooling.

The bit order is DSVT, WidthFormer, GFusion, DAL.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Tuple


CONFIG_DIR = Path("configs/nuscenes/det/ablation")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    bits: str

    config_override: str = ""  # stage experiments point at a hand-written config

    @property
    def config_name(self) -> str:
        if self.config_override:
            return self.config_override
        dsvt, widthformer, gfusion, dal = self.bits
        return f"dsvt{dsvt}_wf{widthformer}_gf{gfusion}_dal{dal}.yaml"

    @property
    def config_path(self) -> Path:
        return CONFIG_DIR / self.config_name

    @property
    def uses_dsvt(self) -> bool:
        return self.bits[0] == "1"


# This is the one authoritative ID <-> bits <-> config ordering.  Keep the
# human-oriented order: baseline, single changes, pairs, triples, full model.
EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment("B0", "0000"),
    Experiment("A1", "1000"),
    Experiment("A2", "0100"),
    Experiment("A3", "0001"),
    Experiment("A4", "0010"),
    Experiment("C1", "1100"),
    Experiment("C2", "1001"),
    Experiment("C3", "1010"),
    Experiment("C4", "0101"),
    Experiment("C5", "0110"),
    Experiment("C6", "0011"),
    Experiment("P1", "1101"),
    Experiment("P2", "1110"),
    Experiment("P3", "1011"),
    Experiment("P4", "0111"),
    Experiment("FINAL", "1111"),
)

# Stage experiments (BEVFusion-style staged training).  Not part of the 16-way
# matrix: S1 = LiDAR-only DSVT + TransFusion head (init from the official DSVT
# backbone) whose checkpoint initialises the DSVT fusion arms, exactly as
# BEVFusion initialises its fusion model from lidar-only.pth.
STAGE_EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment("S1", "1---", config_override="s1_dsvt_lidar.yaml"),
    # FINAL trained "properly": 10 sweeps + GT-Aug (fade last epoch) + ResNet-50.
    Experiment("FINALP", "1111", config_override="final_perf.yaml"),
    Experiment("B0P", "0000", config_override="b0_perf.yaml"),
    Experiment("P2P", "1110", config_override="p2_perf.yaml"),
    Experiment("A1P", "1000", config_override="a1_perf.yaml"),
)

BY_ID = MappingProxyType(
    {item.experiment_id: item for item in EXPERIMENTS + STAGE_EXPERIMENTS}
)
BY_BITS = MappingProxyType({item.bits: item for item in EXPERIMENTS})

if len(BY_ID) != 16 + len(STAGE_EXPERIMENTS) or len(BY_BITS) != 16:
    raise RuntimeError("ablation experiment IDs and bit patterns must be unique")


# Compatibility filenames used by earlier experiment documents.  These are
# generated copies, not additional experiment identities.  In particular, the
# old lowercase c2/c3/c4 filenames do not have the same numbering convention
# as the new uppercase C2/C3/C4 experiment IDs.
LEGACY_CONFIG_ALIASES = MappingProxyType(
    {
        "b0_legacy.yaml": BY_BITS["0000"].config_name,
        "e1_dsvt.yaml": BY_BITS["1000"].config_name,
        "e2_widthformer.yaml": BY_BITS["0100"].config_name,
        "e3_gfusion.yaml": BY_BITS["0010"].config_name,
        "e4_dal.yaml": BY_BITS["0001"].config_name,
        "c2_dsvt_widthformer.yaml": BY_BITS["1100"].config_name,
        "c3_dsvt_widthformer_gfusion.yaml": BY_BITS["1110"].config_name,
        "c4_full.yaml": BY_BITS["1111"].config_name,
    }
)


def select_experiments(ids: Iterable[str]) -> Tuple[Experiment, ...]:
    """Resolve IDs case-insensitively while rejecting unknowns and repeats."""

    selected = []
    seen = set()
    for raw_id in ids:
        experiment_id = raw_id.upper()
        if experiment_id not in BY_ID:
            choices = ", ".join(BY_ID)
            raise ValueError(f"unknown experiment ID {raw_id!r}; choose from {choices}")
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment ID {raw_id!r}")
        selected.append(BY_ID[experiment_id])
        seen.add(experiment_id)
    return tuple(selected)


def canonical_names() -> Tuple[str, ...]:
    return tuple(item.config_name for item in EXPERIMENTS)
