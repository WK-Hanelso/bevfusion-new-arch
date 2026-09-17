"""CPU test for manifest-driven point feature zeroing before H2D."""

from pathlib import Path
import subprocess


RUNTIME = Path(__file__).resolve().parents[1]


def test_cpp_point_feature_zeroing(tmp_path):
    executable = tmp_path / "point_features_test"
    manifest = tmp_path / "lidar.manifest.json"
    manifest.write_text('{"official_layout": true, "zero_feature_channels": [4]}')
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-I",
            str(RUNTIME),
            str(RUNTIME / "point_features.cpp"),
            str(Path(__file__).with_name("point_features_test.cpp")),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable), str(manifest)], check=True)
