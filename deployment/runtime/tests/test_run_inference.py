"""CPU-only bundle contract tests. Fixture bytes are NOT TensorRT engines."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from deployment.runtime import run_inference as run


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bevfusion-bundle-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "relocated bundle"
        self.root.mkdir()
        self.bundle_path = self.root / "bundle_fp16.manifest.json"
        self.model = {
            "config_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "random_init_diagnostic": False,
            "seed": None,
        }
        self.bundle = {
            "schema_version": 2, "precision": "fp16",
            "model": copy.deepcopy(self.model), "engines": {},
        }
        (self.root / "plugins").mkdir()
        plugins = []
        for name in run.PLUGIN_FILES:
            path = self.root / "plugins" / name
            path.write_bytes(b"plugin hash fixture only " + name.encode())
            plugins.append({"path": "/old/mount/plugins/" + name,
                            "sha256": run.sha256_file(path)})
        self.manifests = {}
        for name in run.ENGINE_OPTIONS:
            path = self.root / (name + ".engine")
            path.write_bytes(b"engine hash fixture only " + name.encode())
            manifest = {
                "schema_version": 1, "engine_name": name,
                "engine": "/old/mount/" + path.name,
                "engine_sha256": run.sha256_file(path),
                "model": copy.deepcopy(self.model), "precision": "fp16",
                "tensorrt_version": "10.13.2.6", "machine": "aarch64",
                "plugins": plugins if name == "lidar_raw" else [],
            }
            if name == "lidar_raw":
                manifest.update(
                    point_profile={"min": [1, 5], "opt": [34688, 5], "max": [1000000, 5]},
                    capacity={"max_points": 1000000, "max_pillars": 10000,
                              "max_sets_per_shift": 512},
                )
            self.manifests[name] = manifest
            self.bundle["engines"][name] = {
                "path": manifest["engine"], "sha256": manifest["engine_sha256"],
                "build_manifest": "/old/mount/" + name + ".manifest.json",
            }
        self.save()

    def save(self):
        for name, manifest in self.manifests.items():
            path = self.root / (name + ".manifest.json")
            path.write_text(json.dumps(manifest))
            self.bundle["engines"][name]["build_manifest_sha256"] = run.sha256_file(path)
        self.bundle_path.write_text(json.dumps(self.bundle))

    def test_relocated_bundle_and_default_optimum(self):
        result = run.verify_bundle(self.bundle_path)
        self.assertEqual(result["points"], 34688)
        self.assertEqual(result["check"], "artifact_integrity_only")
        self.assertEqual(len(result["plugins"]), 5)
        for path in result["engines"].values():
            self.assertEqual(Path(path).parent, self.root)

    def test_min_and_million_point_boundaries(self):
        for points in (1, 34688, 1000000):
            with self.subTest(points=points):
                self.assertEqual(run.verify_bundle(self.bundle_path, points)["points"], points)

    def test_outside_profile_rejected(self):
        for points in (-1, 0, 1000001, True):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "outside"):
                run.verify_bundle(self.bundle_path, points)

    def test_modified_engine(self):
        (self.root / "camera_bev.engine").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            run.verify_bundle(self.bundle_path)

    def test_modified_build_manifest(self):
        with (self.root / "camera_bev.manifest.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            run.verify_bundle(self.bundle_path)

    def test_modified_plugin(self):
        (self.root / "plugins" / run.PLUGIN_FILES[0]).write_bytes(b"different capacity build")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            run.verify_bundle(self.bundle_path)

    def test_missing_plugin(self):
        (self.root / "plugins" / run.PLUGIN_FILES[0]).unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            run.verify_bundle(self.bundle_path)

    def test_duplicate_plugin(self):
        specs = self.manifests["lidar_raw"]["plugins"]
        specs[-1] = copy.deepcopy(specs[0])
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate plugin"):
            run.verify_bundle(self.bundle_path)

    def test_mixed_checkpoint(self):
        self.manifests["fusion_dal"]["model"]["checkpoint_sha256"] = "c" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "model identity mismatch"):
            run.verify_bundle(self.bundle_path)

    def test_missing_provenance(self):
        self.bundle["model"].pop("checkpoint_sha256")
        self.save()
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            run.verify_bundle(self.bundle_path)

    def test_mixed_build_targets(self):
        self.manifests["camera_bev"]["machine"] = "x86_64"
        self.save()
        with self.assertRaisesRegex(ValueError, "different TensorRT versions/machines"):
            run.verify_bundle(self.bundle_path)

    def test_mixed_precision(self):
        self.manifests["camera_bev"]["precision"] = "fp32"
        self.save()
        with self.assertRaisesRegex(ValueError, "precision mismatch"):
            run.verify_bundle(self.bundle_path)

    def test_profile_exceeds_capacity(self):
        self.manifests["lidar_raw"]["capacity"]["max_points"] = 100000
        self.save()
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            run.verify_bundle(self.bundle_path)

    def test_random_init_requires_explicit_flag(self):
        models = [self.bundle["model"]] + [m["model"] for m in self.manifests.values()]
        for model in models:
            model.update(checkpoint_sha256=None, random_init_diagnostic=True, seed=0)
        self.save()
        with self.assertRaisesRegex(ValueError, "allow-random-init"):
            run.verify_bundle(self.bundle_path)
        self.assertTrue(run.verify_bundle(self.bundle_path, allow_random_init=True)
                        ["model"]["random_init_diagnostic"])

    def test_check_only_with_no_site_packages_or_runtime(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", run.__file__, "--bundle", str(self.bundle_path),
             "--check-only", "--runtime", "/missing/runtime"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["check"], "artifact_integrity_only")

    def test_corrupt_bundle_does_not_start_runtime(self):
        (self.root / "camera_bev.engine").write_bytes(b"modified")
        with mock.patch.object(run.subprocess, "run") as launch, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(["--bundle", str(self.bundle_path)]), 2)
        launch.assert_not_called()

    def test_verified_argv_and_child_failure_propagation(self):
        args = ["--bundle", str(self.bundle_path), "--runtime", sys.executable,
                "--points", "1000000", "--warmup", "0", "--latency", "--memory",
                "--memory-sample-ms", "5", "--single-thread-submit"]
        with mock.patch.object(run.subprocess, "run") as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            launch.return_value.returncode = 7
            self.assertEqual(run.main(args), 7)
        command = launch.call_args.args[0]
        self.assertEqual(command.count("--plugin"), 5)
        self.assertEqual(command[command.index("--points") + 1], "1000000")
        self.assertIn(str(self.root / "camera_bev.engine"), command)
        self.assertIn("--single-thread-submit", command)
        self.assertNotIn("shell", launch.call_args.kwargs)

    def test_cannot_override_verified_engine(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            run.parse_args(["--bundle", str(self.bundle_path), "--camera-engine", "/other"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
