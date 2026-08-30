import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PUBLIC_ROOT / "scripts" / "write_release_checksums.py"
SPEC = importlib.util.spec_from_file_location("release_checksums", SCRIPT)
release_checksums = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_checksums)


class ReleaseIntegrityTests(unittest.TestCase):
    def test_license_and_security_policy_are_distributed(self):
        license_text = (PUBLIC_ROOT / "LICENSE").read_text(encoding="utf-8")
        security_text = (PUBLIC_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        manifest = (PUBLIC_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("Apache License", license_text)
        self.assertIn("founder@arcafutura.com", security_text)
        self.assertIn("LICENSE", manifest)
        self.assertIn("SECURITY.md", manifest)
        self.assertIn("scripts/write_release_checksums.py", manifest)

    def test_manifest_hashes_exact_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            artifact = directory / "aegis_trial_sdk-0.1.1-py3-none-any.whl"
            artifact.write_bytes(b"public SDK release bytes\x00")
            output = directory / "SHA256SUMS"

            release_checksums.write_manifest([artifact], output)

            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(output.read_text(encoding="utf-8"), f"{expected}  {artifact.name}\n")

    def test_manifest_rejects_an_empty_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "at least one"):
                release_checksums.write_manifest([], Path(tmp) / "SHA256SUMS")


if __name__ == "__main__":
    unittest.main()
