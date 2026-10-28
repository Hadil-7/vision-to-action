"""Unit tests plus an actual generate/train/save/load/predict integration test."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import main


class RuleAndImageTests(unittest.TestCase):
    def test_rule_for_all_occupancy_patterns(self):
        expected = [0, 0, 1, 1, 0, 0, 2, 3]
        for occupied, action in zip(itertools.product((False, True), repeat=3), expected):
            with self.subTest(occupied=occupied):
                self.assertEqual(main.navigation_action(occupied), action)

    def test_generator_is_deterministic_and_labels_are_valid(self):
        for action in range(4):
            first, label, occupied = main.scene(123, action)
            second, _, _ = main.scene(123, action)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (64, 64, 3))
            self.assertEqual(first.dtype, np.uint8)
            self.assertEqual(label, main.navigation_action(occupied))
            self.assertEqual(label, action)

    def test_preprocessing_preserves_rgb_order_and_normalisation(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        tensor = main.preprocess_rgb(image)
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.all(tensor[0] == 1))
        self.assertTrue(torch.all(tensor[1:] == 0))

    def test_image_round_trip_matches_training_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.png"
            image, _, _ = main.scene(42)
            main.write_rgb(path, image)
            np.testing.assert_array_equal(main.read_rgb(path), image)
            self.assertTrue(torch.equal(main.preprocess_rgb(image), main.preprocess_rgb(main.read_rgb(path))))

    def test_invalid_image_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            main.preprocess_rgb(np.zeros((64, 64), dtype=np.uint8))

    def test_pixel_only_colour_baseline_matches_synthetic_rule(self):
        for seed in range(20):
            for action in range(4):
                image, label, _ = main.scene(seed, action)
                self.assertEqual(main.colour_rule_prediction(image), label)


class DatasetTests(unittest.TestCase):
    def test_balanced_disjoint_splits_with_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = main.generate_dataset(Path(folder), {"train": 16, "validation": 8, "test": 8}, 12)
            manifest = main.load_manifest(manifest_path)
            previous = set()
            for split, records in manifest["splits"].items():
                with self.subTest(split=split):
                    seeds = {record["seed"] for record in records}
                    self.assertFalse(previous.intersection(seeds))
                    previous.update(seeds)
                    self.assertEqual(np.bincount([row["label"] for row in records]).tolist(), [len(records) // 4] * 4)
                    dataset = main.Scenes(manifest_path.parent, records)
                    self.assertEqual(tuple(dataset[0][0].shape), (3, 64, 64))

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = main.generate_dataset(Path(folder), {"train": 4, "validation": 4, "test": 4})
            manifest = json.loads(path.read_text())
            manifest["splits"]["test"][0]["seed"] = manifest["splits"]["train"][0]["seed"]
            main.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "overlap"):
                main.load_manifest(path)

    def test_bad_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = main.generate_dataset(Path(folder), {"train": 4, "validation": 4, "test": 4})
            manifest = json.loads(path.read_text())
            manifest["splits"]["test"][0]["label"] = -1
            main.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "rule"):
                main.load_manifest(path)

    def test_generator_rejects_invalid_counts_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaises(ValueError):
                main.generate_dataset(path, {"train": 5, "validation": 4, "test": 4})
            main.generate_dataset(path, {"train": 4, "validation": 4, "test": 4})
            with self.assertRaises(FileExistsError):
                main.generate_dataset(path, {"train": 4, "validation": 4, "test": 4})


class MetricsTests(unittest.TestCase):
    def test_hand_computed_metrics_and_matrix_orientation(self):
        result = main.classification_metrics([0, 0, 1, 2, 3], [0, 1, 1, 2, 3])
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(result["per_class"]["FORWARD"]["precision"], 1.0)
        self.assertEqual(result["per_class"]["FORWARD"]["recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 5 / 6)

    def test_missing_predictions_do_not_divide_by_zero(self):
        result = main.classification_metrics([0, 1, 2, 3], [0, 0, 0, 0])
        self.assertEqual(result["per_class"]["STOP"]["precision"], 0)
        with self.assertRaises(ValueError):
            main.classification_metrics([], [])


class EndToEndTests(unittest.TestCase):
    def test_generate_train_evaluate_checkpoint_and_prediction_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = main.generate_dataset(root / "data", {"train": 32, "validation": 8, "test": 8}, 17)
            args = argparse.Namespace(manifest=manifest_path, seed=17, threads=1, epochs=1,
                batch_size=8, learning_rate=0.001, output=root / "outputs", report=root / "report")
            report = main.train_model(args)
            self.assertEqual(report["test"]["samples"], 8)
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(sum(map(sum, report["test"]["confusion_matrix"])), 8)
            self.assertTrue((args.report / "results.md").is_file())
            model = main.load_model(args.output / "model.pt")
            manifest = main.load_manifest(manifest_path)
            dataset = main.Scenes(manifest_path.parent, manifest["splits"]["test"])
            reevaluated = main.evaluate_model(model, torch.utils.data.DataLoader(dataset, batch_size=8))
            self.assertEqual(reevaluated["confusion_matrix"], report["test"]["confusion_matrix"])
            image = manifest_path.parent / manifest["splits"]["test"][0]["path"]
            prediction = main.predict_image(model, image)
            self.assertIn(prediction["action"], main.LABELS)
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1, places=5)
            command = [sys.executable, str(Path(main.__file__)), "predict", str(image), "--model", str(args.output / "model.pt")]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["action"], prediction["action"])


if __name__ == "__main__":
    unittest.main()
