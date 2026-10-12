"""Security and installation failure tests using temporary, isolated fixtures."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.download_artifacts import (
    download_file, extract_verified_zip, install_artifacts, load_descriptor,
    relative_path, verify_artifacts,
)
from scripts.setup_runtime import child_environment, healthy, require_inside, setup
from scripts.download_data import extract_canonical_dataset, install_dataset, verify_dataset
from scripts import setup_kaggle_runtime as kaggle_runtime


def entry(path: str, contents: bytes) -> dict:
    return {"path": path, "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}


class ArtifactSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def descriptor(self, artifacts):
        path = self.root / "descriptor.json"
        path.write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}), encoding="utf-8")
        return path

    def test_paths_reject_windows_and_posix_escape(self):
        for value in ("../weights.keras", "/weights.keras", "C:/weights.keras", "model\\weights.keras", "model/../weights.keras", "model//weights.keras", "model/./weights.keras"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                relative_path(value)

    def test_unverified_or_credentialed_urls_rejected(self):
        for url in ("http://example.org/weights.keras", "file:///weights.keras", "https://user:password@example.org/weights.keras"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                load_descriptor(self.descriptor([{**entry("weights.keras", b"weights"), "url": url}]))

    def test_null_url_has_actionable_failure_without_network(self):
        with self.assertRaisesRegex(ValueError, "No verified public download URL"):
            download_file({**entry("weights.keras", b"weights"), "url": None}, self.root / "download")
        self.assertFalse((self.root / "download").exists())

    def test_atomic_install_and_corruption_detection(self):
        contents = b"verified-model-bytes"
        descriptor = self.descriptor([{**entry("weights.keras", contents), "url": "https://example.org/weights.keras"}])
        output = self.root / "artifacts"
        with patch("scripts.download_artifacts.download_file", side_effect=lambda artifact, destination: destination.write_bytes(contents)):
            result = install_artifacts(descriptor, output)
        self.assertEqual(result["files"], 1)
        self.assertEqual((output / "weights.keras").read_bytes(), contents)
        self.assertTrue(verify_artifacts(descriptor, output)["verified"])
        (output / "weights.keras").write_bytes(contents[:-1] + b"X")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_artifacts(descriptor, output)
        self.assertFalse((self.root / ".artifacts.install.lock").exists())

    def test_bad_download_is_not_installed(self):
        descriptor = self.descriptor([{**entry("weights.keras", b"expected"), "url": "https://example.org/weights.keras"}])
        output = self.root / "artifacts"
        with patch("scripts.download_artifacts.download_file", side_effect=lambda artifact, destination: destination.write_bytes(b"corrupt!")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                install_artifacts(descriptor, output)
        self.assertFalse(output.exists())
        self.assertFalse(any(self.root.glob(".artifacts.staging-*")))

    def test_exact_inventory_rejects_extra_files(self):
        output = self.root / "artifacts"
        output.mkdir()
        (output / "weights.keras").write_bytes(b"weights")
        (output / "untracked.txt").write_text("extra")
        descriptor = self.descriptor([{**entry("weights.keras", b"weights"), "url": None}])
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            verify_artifacts(descriptor, output)

    def test_zip_traversal_cannot_write_outside_install(self):
        archive = self.root / "models.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../escape.keras", b"escape")
        with self.assertRaises(ValueError):
            extract_verified_zip(archive, self.root / "installed", [entry("weights.keras", b"weights")])
        self.assertFalse((self.root / "escape.keras").exists())

    def test_zip_duplicate_and_unlisted_files_rejected(self):
        for paths in (("weights.keras", "extra.keras"), ("weights.keras", "weights.keras")):
            archive = self.root / "models.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for path in paths:
                    bundle.writestr(path, b"weights")
            destination = self.root / ("installed-" + str(len(list(self.root.glob('installed-*')))))
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "duplicate or undeclared"):
                extract_verified_zip(archive, destination, [entry("weights.keras", b"weights")])

    def test_complete_zip_install_checks_each_file(self):
        archive = self.root / "models.zip"
        files = {"teacher/model.keras": b"teacher", "student/model.keras": b"student"}
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, contents in files.items():
                bundle.writestr(name, contents)
        descriptor = self.descriptor([{**entry("models", archive.read_bytes()), "kind": "zip", "url": "https://example.org/models.zip",
                                      "inventory": [entry(name, contents) for name, contents in files.items()]}])
        with patch("scripts.download_artifacts.download_file", side_effect=lambda artifact, destination: shutil.copyfile(archive, destination)):
            result = install_artifacts(descriptor, self.root / "artifacts")
        self.assertEqual(result["files"], 2)
        for name, contents in files.items():
            self.assertEqual((self.root / "artifacts" / "models" / name).read_bytes(), contents)

    def test_existing_output_not_overwritten(self):
        output = self.root / "artifacts"
        output.mkdir()
        path = output / "weights.keras"
        path.write_bytes(b"user-model")
        descriptor = self.descriptor([{**entry("weights.keras", b"expected"), "url": "https://example.org/weights.keras"}])
        with self.assertRaises(ValueError):
            install_artifacts(descriptor, output)
        self.assertEqual(path.read_bytes(), b"user-model")

    def test_concurrent_install_is_rejected(self):
        descriptor = self.descriptor([{**entry("weights.keras", b"expected"), "url": None}])
        lock = self.root / ".artifacts.install.lock"
        lock.write_text("existing installer")
        with self.assertRaisesRegex(ValueError, "Another artifact installation"):
            install_artifacts(descriptor, self.root / "artifacts")
        self.assertEqual(lock.read_text(), "existing installer")


class EnvironmentIsolationTests(unittest.TestCase):
    def test_global_environment_redirects_removed_and_cpu_forced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hostile = {"PYTHONPATH": "external", "PYTHONHOME": "external", "PIP_TARGET": "external", "PIP_EXTRA_INDEX_URL": "https://example.org", "CUDA_VISIBLE_DEVICES": "0", "PIP_CERT": "preserved"}
            with patch.dict(os.environ, hostile):
                result = child_environment(root)
            for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_EXTRA_INDEX_URL"):
                self.assertNotIn(key, result)
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(result["PIP_CERT"], "preserved")
            self.assertEqual(result["PIP_CONFIG_FILE"], os.devnull)
            self.assertTrue(Path(result["TEMP"]).resolve().is_relative_to(root.resolve()))

    def test_setup_cannot_write_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            with patch("scripts.setup_runtime.subprocess.run") as execute:
                with self.assertRaisesRegex(ValueError, "inside the project"):
                    setup(root, root.parent / "outside-venv")
                execute.assert_not_called()

    def test_environment_using_system_packages_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / "venv"
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
            with patch("scripts.setup_runtime.subprocess.run") as execute:
                self.assertFalse(healthy(venv, root, {}))
                execute.assert_not_called()

    def test_symlink_cannot_redirect_local_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "redirect").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Creating symbolic links is unavailable for this account")
            with self.assertRaises(ValueError):
                require_inside(root, root / "redirect" / "venv")


class CanonicalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, extra=None):
        files = {"Train.csv": b"Path,ClassId\nTrain/0/00000_00000_00000.png,0\n",
                 "Test.csv": b"Path,ClassId\nTest/00000.png,0\n",
                 "Meta.csv": b"Path,ClassId\nMeta/0.png,0\n",
                 "Train/0/00000_00000_00000.png": b"canonical-training-image",
                 "Test/00000.png": b"canonical-test-image", "Meta/0.png": b"canonical-meta-image",
                 "train/0/00000_00000_00000.png": b"case-variant-must-not-overwrite",
                 "test/00000.png": b"case-variant-test"}
        files.update(extra or {})
        archive = self.root / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, contents in files.items():
                bundle.writestr(name, contents)
        source = {"schema_version": 1, "archive": {**entry("dataset.zip", archive.read_bytes()), "url": None},
                  "expected_counts": {"Train": 1, "Test": 1, "Meta": 1}}
        descriptor = self.root / "dataset-source.json"
        descriptor.write_text(json.dumps(source), encoding="utf-8")
        return archive, descriptor, source

    def test_case_variant_images_do_not_overwrite_csv_references(self):
        archive, descriptor, _ = self.fixture()
        output = self.root / "installed"
        result = install_dataset(descriptor, output, archive=archive)
        self.assertTrue(result["verified"])
        self.assertEqual((output / "Train/0/00000_00000_00000.png").read_bytes(), b"canonical-training-image")
        self.assertEqual(result["files"], 6)
        self.assertEqual({path.name for path in output.iterdir() if path.is_dir()}, {"Train", "Test", "Meta"})

    def test_dataset_inventory_detects_changed_image(self):
        archive, descriptor, _ = self.fixture()
        output = self.root / "installed"
        install_dataset(descriptor, output, archive=archive)
        (output / "Test/00000.png").write_bytes(b"changed-test-image")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_dataset(descriptor, output)

    def test_dataset_traversal_rejected_even_in_ignored_member(self):
        archive, _, source = self.fixture({"../escape.png": b"malicious"})
        with self.assertRaises(ValueError):
            extract_canonical_dataset(archive, self.root / "installed", source)
        self.assertFalse((self.root / "escape.png").exists())

    def test_dataset_corruption_is_not_extracted(self):
        archive, descriptor, _ = self.fixture()
        archive.write_bytes(archive.read_bytes()[:-1] + b"X")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            install_dataset(descriptor, self.root / "installed", archive=archive)
        self.assertFalse((self.root / "installed").exists())


class KaggleBootstrapGuards(unittest.TestCase):
    def test_gpu_bootstrap_rejects_windows_before_any_download(self):
        with patch.object(kaggle_runtime.platform, "system", return_value="Windows"):
            with patch.object(kaggle_runtime, "install_uv") as install:
                with self.assertRaisesRegex(RuntimeError, "Kaggle kernels only"):
                    kaggle_runtime.setup(Path("unread-source"), Path("untouched-runtime"))
                install.assert_not_called()

    def test_linux_without_kaggle_mounts_is_rejected(self):
        with patch.object(kaggle_runtime.platform, "system", return_value="Linux"), \
             patch.object(kaggle_runtime.platform, "machine", return_value="x86_64"), \
             patch.object(kaggle_runtime.Path, "is_dir", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "real Kaggle kernel"):
                kaggle_runtime.require_kaggle()

    def test_kaggle_environment_preserves_assigned_gpu_and_removes_global_redirects(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with patch.object(kaggle_runtime, "scoped_path", side_effect=lambda path: path.resolve()), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": "global-python", "TF_USE_LEGACY_KERAS": "1", "PIP_TARGET": "global", "UV_PROJECT": "another-project"}):
                result = kaggle_runtime.child_environment(runtime)
            self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "1")
            for key in ("PYTHONPATH", "TF_USE_LEGACY_KERAS", "PIP_TARGET", "UV_PROJECT"):
                self.assertNotIn(key, result)
            self.assertTrue(Path(result["UV_PYTHON_INSTALL_DIR"]).is_relative_to(runtime))
            self.assertEqual(result["PIP_CONFIG_FILE"], os.devnull)

    def test_existing_unowned_runtime_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            sentinel = runtime / "other-project.txt"
            sentinel.write_text("retain")
            with patch.object(kaggle_runtime, "scoped_path", return_value=runtime):
                with self.assertRaisesRegex(ValueError, "not owned by this project"):
                    kaggle_runtime.prepare_runtime_directory(runtime)
            self.assertEqual(sentinel.read_text(), "retain")
            self.assertFalse((runtime / "runtime-owner.json").exists())

    def test_corrupted_bootstrap_archive_is_not_executed_or_installed(self):
        class Response(io.BytesIO):
            def geturl(self):
                return "https://github.com/astral-sh/uv/releases/download/0.13.0/archive"

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with patch.object(kaggle_runtime, "scoped_path", side_effect=lambda path: path.resolve()), \
                 patch.object(kaggle_runtime.urllib.request, "urlopen", return_value=Response(b"corrupted archive")), \
                 patch.object(kaggle_runtime.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "official release checksum"):
                    kaggle_runtime.install_uv(runtime)
                execute.assert_not_called()
            self.assertFalse((runtime / "uv-tools" / kaggle_runtime.UV_VERSION / "uv").exists())
            self.assertFalse(any(runtime.rglob("*.part")))


if __name__ == "__main__":
    unittest.main()
