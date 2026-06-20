import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def load_json(name):
    path = CONTRACTS / name
    if not path.is_file():
        raise AssertionError(f"missing contract: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


class ContractMetadataTest(unittest.TestCase):
    def test_model_shape_contract_is_consistent(self):
        classes = load_json("classes.json")
        preprocessing = load_json("preprocessing.json")
        manifest = load_json("manifest.example.json")

        self.assertEqual(23, len(classes))
        self.assertEqual(23, manifest["class_count"])
        self.assertEqual(224, preprocessing["image_size"])
        self.assertEqual(224, manifest["image_size"])
        self.assertEqual(512, manifest["embedding_dim"])

    def test_policy_targets_the_exported_model(self):
        confidence = load_json("confidence_policy.json")
        manifest = load_json("manifest.example.json")

        self.assertEqual(manifest["model_id"], confidence["model_id"])
        self.assertEqual(manifest["precision"], confidence["precision"])
        self.assertFalse(manifest["binaries_included"])

    def test_text_search_weights_form_a_fusion_score(self):
        policy = load_json("text_search_policy.json")

        self.assertAlmostEqual(
            1.0,
            policy["semantic_weight"] + policy["keyword_weight"],
        )


if __name__ == "__main__":
    unittest.main()
