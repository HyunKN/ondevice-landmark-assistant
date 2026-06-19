import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "data" / "metrics.json"
README_PATH = ROOT / "README.md"
SVG_EXPECTATIONS = {
    "hero.svg": "On-device Landmark Assistant",
    "experiment-comparison.svg": "Validation Top-1",
    "deployment-flow.svg": "Confidence Policy",
    "npu-evidence.svg": "feasibility evidence only",
}
APP_CAPTURES = (
    "app-home.png",
    "app-image-result.png",
    "app-text-search.png",
)
PUBLIC_DIRS = ("assets", "data", "scripts")


class PortfolioAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def test_reviewed_metrics_are_complete(self):
        self.assertEqual(self.metrics["class_count"], 23)
        self.assertEqual(self.metrics["main_run_count"], 40)
        self.assertEqual(self.metrics["fold_count"], 5)
        self.assertEqual(len(self.metrics["configurations"]), 8)
        self.assertEqual(self.metrics["configurations"][0]["val_top1"], 99.05)

    def test_repository_contains_no_model_binaries(self):
        forbidden_suffixes = {".onnx", ".pt", ".pth", ".ckpt"}
        found = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
        self.assertEqual(found, [])

    def test_generated_svgs_are_safe_and_project_specific(self):
        for filename, expected_label in SVG_EXPECTATIONS.items():
            with self.subTest(filename=filename):
                path = ROOT / "assets" / filename
                self.assertTrue(path.is_file(), f"missing {path.relative_to(ROOT)}")
                source = path.read_text(encoding="utf-8")
                self.assertIn("<svg", source)
                self.assertIn("viewBox=", source)
                self.assertNotIn("<script", source.lower())
                self.assertNotRegex(source, r"[A-Za-z]:\\")
                self.assertIn(expected_label, source)

    def test_app_captures_exist(self):
        for filename in APP_CAPTURES:
            with self.subTest(filename=filename):
                path = ROOT / "assets" / filename
                self.assertTrue(path.is_file(), f"missing {path.relative_to(ROOT)}")
                self.assertGreater(path.stat().st_size, 10_000)

    def test_readme_has_required_sections_and_language(self):
        self.assertTrue(README_PATH.is_file(), "missing README.md")
        source = README_PATH.read_text(encoding="utf-8")
        required = (
            "## What this project does",
            "## App flow",
            "## System design",
            "## Experiment evidence",
            "## Deployment findings",
            "## My contribution",
            "## Limitations",
            "## Evidence",
            "closed-set",
            "final Android code implementation",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, source)

        forbidden = (
            "혁신",
            "여정",
            "압도적",
            "game-changing",
            "cutting-edge",
            "production-ready",
            "fully optimized",
        )
        lowered = source.lower()
        for phrase in forbidden:
            with self.subTest(forbidden=phrase):
                self.assertNotIn(phrase.lower(), lowered)

    def test_readme_images_exist_and_have_alt_text(self):
        self.assertTrue(README_PATH.is_file(), "missing README.md")
        source = README_PATH.read_text(encoding="utf-8")
        markdown_images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", source)
        html_images = re.findall(r'<img\s+[^>]*src="([^"]+)"[^>]*alt="([^"]+)"', source)
        self.assertGreaterEqual(len(markdown_images) + len(html_images), 7)
        for alt, target in markdown_images:
            with self.subTest(target=target):
                self.assertTrue(alt.strip())
                self.assertTrue((ROOT / target).is_file(), f"missing image {target}")
        for target, alt in html_images:
            with self.subTest(target=target):
                self.assertTrue(alt.strip())
                self.assertTrue((ROOT / target).is_file(), f"missing image {target}")

    def test_public_files_contain_no_local_paths_or_credentials(self):
        paths = [README_PATH] if README_PATH.exists() else []
        for dirname in PUBLIC_DIRS:
            directory = ROOT / dirname
            if directory.exists():
                paths.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".svg", ".json", ".py"})

        forbidden_patterns = (
            re.compile(r"[A-Za-z]:\\"),
            re.compile(r"PRIVATE KEY", re.IGNORECASE),
            re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
            re.compile(r"token\s*[:=]", re.IGNORECASE),
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                with self.subTest(path=path.name, pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(source))


if __name__ == "__main__":
    unittest.main()
