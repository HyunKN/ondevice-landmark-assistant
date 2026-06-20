import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
REQUIRED_FILES = (
    CODE / "README.md",
    CODE / "SOURCES.md",
    CODE / "CONTRIBUTIONS.md",
    CODE / "training" / "README.md",
    CODE / "sprint1_prototype" / "README.md",
    CODE / "model_integration" / "README.md",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".data"}
FORBIDDEN_PARTS = {"wandb_export_tool", "wandb_exports"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".toml", ".yaml", ".yml", ".patch"}
CONFIG_DIR = CODE / "training" / "configs" / "main_matrix"


class PublicCodeSnapshotTest(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
        self.assertEqual([], missing)

    def test_no_private_or_large_artifacts(self):
        violations = []
        for path in CODE.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                violations.append(str(path.relative_to(ROOT)))
            if FORBIDDEN_PARTS.intersection(part.lower() for part in path.parts):
                violations.append(str(path.relative_to(ROOT)))
            if path.stat().st_size > 10 * 1024 * 1024:
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)

    def test_no_known_local_paths(self):
        violations = []
        for path in CODE.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "C:\\Users\\Ltp" in source or "D:\\app-test" in source:
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)

    def test_main_matrix_contains_eight_sanitized_configs(self):
        configs = sorted(path for path in CONFIG_DIR.glob("*.json") if path.name != "SOURCE_MAP.json")
        self.assertEqual(8, len(configs))
        for path in configs:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertNotIn("runtime", payload)

        source_map = json.loads((CONFIG_DIR / "SOURCE_MAP.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(8, len(source_map["configs"]))

    def test_integration_snapshot_contains_real_source_and_not_team_app_copy(self):
        expected = (
            CODE / "model_integration" / "scripts" / "check_model_contract.py",
            CODE / "model_integration" / "scripts" / "generate_semantic_text_artifacts.py",
            CODE / "model_integration" / "patches" / "2e4349b-android-asset-cache-fix.patch",
        )
        self.assertEqual([], [str(path.relative_to(ROOT)) for path in expected if not path.is_file()])
        self.assertFalse((CODE / "model_integration" / "lib").exists())
        self.assertFalse((CODE / "model_integration" / "android").exists())


if __name__ == "__main__":
    unittest.main()
