"""Learn a toy navigation rule from synthetic images with a PyTorch CNN."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
IMAGE_SIZE = 64
SPLIT_BLOCK = 1_000_000


def navigation_action(occupied):
    """Prefer forward, then left, then right; stop when all lanes are blocked."""
    if len(occupied) != 3:
        raise ValueError("Exactly three occupancy values are required.")
    return 0 if not occupied[1] else 1 if not occupied[0] else 2 if not occupied[2] else 3


def scene(seed, action=None):
    """Create a deterministic RGB scene; uniform actions avoid class imbalance."""
    rng = random.Random(seed)
    action = rng.randrange(len(LABELS)) if action is None else action
    if action not in range(len(LABELS)):
        raise ValueError("Unknown action index.")
    patterns = {
        0: ([False, False, False], [True, False, False], [False, False, True], [True, False, True]),
        1: ([False, True, False], [False, True, True]),
        2: ([True, True, False],), 3: ([True, True, True],),
    }
    occupied = list(rng.choice(patterns[action]))
    image = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[:] = tuple(rng.randint(30, 65) for _ in range(3))
    for lane, blocked in enumerate(occupied):
        x = lane * 21 + 2
        shade = rng.randint(77, 104)
        cv2.rectangle(image, (x, 6), (x + 17, 62), (shade,) * 3, -1)
        if blocked:
            y = rng.randint(12, 29)
            colour = (rng.randint(178, 240), rng.randint(45, 95), rng.randint(35, 80))
            cv2.rectangle(image, (x + rng.randint(0, 2), y), (x + 16, y + rng.randint(18, 28)), colour, -1)
    noise = np.random.default_rng(seed).integers(-5, 6, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8), action, occupied


def preprocess_rgb(image):
    """Shared training/inference transform: RGB, resize, CHW float32 in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div(255.0)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write image: {path}")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate_dataset(output, counts, seed=42):
    """Write balanced PNGs with provenance and reserve disjoint seed blocks."""
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("Counts must specify train, validation and test.")
    if any(count < 4 or count > SPLIT_BLOCK or count % 4 for count in counts.values()):
        raise ValueError("Each split size must be a multiple of 4 between 4 and 1,000,000.")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Dataset already exists: {manifest_path}. Use a new output folder.")
    manifest = {
        "format_version": 1, "generator": "synthetic-three-lane-v1", "seed": seed,
        "image_size": IMAGE_SIZE, "labels": list(LABELS),
        "description": "Balanced synthetic lane images. No real camera or robot data.", "splits": {},
    }
    for split_index, split in enumerate(("train", "validation", "test")):
        actions = [index % 4 for index in range(counts[split])]
        random.Random(seed + split_index).shuffle(actions)
        records = []
        for index, action in enumerate(actions):
            image_seed = seed * SPLIT_BLOCK * 3 + split_index * SPLIT_BLOCK + index
            image, label, occupied = scene(image_seed, action)
            relative = f"{split}/{index:06d}-{LABELS[label]}.png"
            write_rgb(output / relative, image)
            records.append({"path": relative, "seed": image_seed, "label": label, "occupied": occupied})
        manifest["splits"][split] = records
    write_json(manifest_path, manifest)
    return manifest_path


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("labels") != list(LABELS):
        raise ValueError("Unsupported dataset format or label ordering.")
    if set(manifest.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Dataset must have train, validation and test splits only.")
    all_seeds, all_paths = set(), set()
    for split, records in manifest["splits"].items():
        if not records:
            raise ValueError(f"Empty dataset split: {split}")
        for record in records:
            if record["seed"] in all_seeds or record["path"] in all_paths:
                raise ValueError("Dataset splits overlap in seeds or image paths.")
            all_seeds.add(record["seed"])
            all_paths.add(record["path"])
            if record["label"] != navigation_action(record["occupied"]):
                raise ValueError("A label disagrees with its occupancy rule.")
            image_path = (path.parent / record["path"]).resolve()
            if not image_path.is_relative_to(path.parent.resolve()) or not image_path.is_file():
                raise ValueError(f"Invalid dataset image path: {record['path']}")
    return manifest


class Scenes(Dataset):
    def __init__(self, root, records):
        self.root, self.records = root, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return preprocess_rgb(read_rgb(self.root / record["path"])), record["label"]


class Model(nn.Module):
    """Two convolutional blocks followed by a dense action classifier."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(24 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, len(LABELS)),
        )

    def forward(self, images):
        return self.layers(images)


def classification_metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError("Metrics require non-empty equal-length label arrays.")
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = int(matrix[index, index])
        support, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support}
    return {
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "samples": len(targets), "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; order matches labels", "labels": list(LABELS),
    }


def colour_rule_prediction(image):
    """A transparent pixel-only baseline tailored to this toy generator."""
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    channels = resized.astype(np.float32)
    red_mask = ((channels[:, :, 0] > 130) & (channels[:, :, 0] > channels[:, :, 1] * 1.5)
        & (channels[:, :, 0] > channels[:, :, 2] * 1.5))
    occupied = [bool(red_mask[6:63, lane * 21 + 2:lane * 21 + 20].mean() > 0.15) for lane in range(3)]
    return navigation_action(occupied)


def evaluate_model(model, loader):
    model.eval()
    targets, predictions, total_loss = [], [], 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
    result = classification_metrics(targets, predictions)
    result["loss"] = total_loss / len(targets)
    return result


def configure_runtime(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_model(args):
    configure_runtime(args.seed, args.threads)
    manifest = load_manifest(args.manifest)
    loaders = {split: DataLoader(Scenes(args.manifest.parent, records), batch_size=args.batch_size,
        shuffle=split == "train", num_workers=0, generator=torch.Generator().manual_seed(args.seed))
        for split, records in manifest["splits"].items()}
    model = Model()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history, started = [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            optimiser.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * labels.shape[0]
        validation = evaluate_model(model, loaders["validation"])
        row = {"epoch": epoch, "train_loss": total_loss / len(loaders["train"].dataset),
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_state, best_epoch = validation["loss"], copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "architecture": "two-convolution-cnn-v1", "labels": list(LABELS),
        "image_size": IMAGE_SIZE, "model_state_dict": model.state_dict()}, args.output / "model.pt")
    # The test set is first evaluated only after checkpoint selection on validation.
    test_metrics = evaluate_model(model, loaders["test"])
    test_rows = manifest["splits"]["test"]
    targets = [row["label"] for row in test_rows]
    baseline_predictions = [colour_rule_prediction(read_rgb(args.manifest.parent / row["path"])) for row in test_rows]
    report = {
        "task": "Synthetic three-lane visual action classification",
        "scope": "Same-generator synthetic holdout only; not real-world robot evaluation.",
        "dataset_seed": manifest["seed"],
        "dataset_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "training_seed": args.seed, "split_sizes": {s: len(r) for s, r in manifest["splits"].items()},
        "seed_ranges": {s: [min(r["seed"] for r in rows), max(r["seed"] for r in rows)]
            for s, rows in manifest["splits"].items()},
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "threads": args.threads},
        "environment": {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__, "device": "cpu"},
        "checkpoint_selection": "Lowest validation cross-entropy; no test-set tuning",
        "best_epoch": best_epoch, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history, "test": test_metrics,
        "baselines": {
            "constant_forward": classification_metrics(targets, [0] * len(targets)),
            "opencv_colour_rule": classification_metrics(targets, baseline_predictions),
            "note": "Fixed colour thresholds are tailored to the synthetic generator, not learned or tuned on test images.",
        },
    }
    write_json(args.output / "metrics.json", report)
    if args.report:
        write_json(args.report / "metrics.json", report)
        (args.report / "results.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output / "model.pt"), "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"]}), flush=True)
    return report


def markdown_report(report):
    metrics = report["test"]
    lines = ["# Measured synthetic benchmark\n",
        "Generated by an actual CPU training run. Test images use the same synthetic generator as training; these numbers do not measure real camera or robot performance.\n",
        f"- Split sizes: {report['split_sizes']}",
        f"- Dataset seed: {report['dataset_seed']}; training seed: {report['training_seed']}",
        f"- Best epoch by validation loss: {report['best_epoch']}",
        f"- Test accuracy: {metrics['accuracy']:.4f}", f"- Test macro F1: {metrics['macro_f1']:.4f}",
        f"- Constant-FORWARD baseline accuracy: {report['baselines']['constant_forward']['accuracy']:.4f}",
        f"- OpenCV colour-rule baseline accuracy: {report['baselines']['opencv_colour_rule']['accuracy']:.4f}",
        f"- Run time: {report['elapsed_seconds']:.3f} seconds (hardware dependent)\n",
        "| Action | Precision | Recall | F1 | Test images |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, row in metrics["per_class"].items():
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |")
    lines.extend(["\nConfusion matrix (rows = true action; columns = predicted action):\n", "```text",
        "                 FORWARD  TURN_LEFT  TURN_RIGHT  STOP"])
    for label, row in zip(LABELS, metrics["confusion_matrix"]):
        lines.append(f"{label:12}     " + "  ".join(f"{value:8d}" for value in row))
    lines.extend(["```\n", "Reproduce with the README commands. Full epoch history, package versions and disjoint seed ranges are in `metrics.json`.\n"])
    return "\n".join(lines)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != list(LABELS) or checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError("Checkpoint label ordering or image dimensions do not match.")
    model = Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_image(model, path):
    with torch.no_grad():
        probabilities = torch.softmax(model(preprocess_rgb(read_rgb(path)).unsqueeze(0)), dim=1)[0]
    return {"image": str(path), "action": LABELS[int(probabilities.argmax())],
        "probabilities": dict(zip(LABELS, probabilities.tolist())),
        "warning": "Synthetic teaching model. Probabilities are uncalibrated. Not for robot control or safety decisions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate balanced train/validation/test PNGs")
    generate.add_argument("--output", type=Path, default=Path("data"))
    generate.add_argument("--train", type=int, default=2400)
    generate.add_argument("--validation", type=int, default=400)
    generate.add_argument("--test", type=int, default=400)
    generate.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train, select on validation, evaluate once on test")
    train.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--output", type=Path, default=Path("outputs"))
    train.add_argument("--report", type=Path, help="Also save a shareable report, e.g. reports")
    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint on a chosen split")
    evaluate.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    evaluate.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    predict = sub.add_parser("predict", help="Predict with the same OpenCV preprocessing")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    examples = sub.add_parser("examples", help="Save one independent synthetic image per action")
    examples.add_argument("--output", type=Path, default=Path("examples"))
    examples.add_argument("--seed", type=int, default=9_000_000_000)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            path = generate_dataset(args.output, {"train": args.train, "validation": args.validation, "test": args.test}, args.seed)
            print(json.dumps({"manifest": str(path)}))
        elif args.command == "train":
            if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.threads < 1 or args.seed < 0:
                parser.error("Epochs, batch size, learning rate and threads must be positive; seed must be non-negative.")
            train_model(args)
        elif args.command == "evaluate":
            configure_runtime(42)
            manifest = load_manifest(args.manifest)
            loader = DataLoader(Scenes(args.manifest.parent, manifest["splits"][args.split]), batch_size=64)
            print(json.dumps(evaluate_model(load_model(args.model), loader), indent=2))
        elif args.command == "predict":
            configure_runtime(42)
            print(json.dumps(predict_image(load_model(args.model), args.image), indent=2))
        else:
            for index, label in enumerate(LABELS):
                image, _, _ = scene(args.seed + index, index)
                write_rgb(args.output / f"{label.lower()}.png", image)
            print(json.dumps({"examples": str(args.output), "seed": args.seed}))
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
