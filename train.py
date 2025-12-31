from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO

import wandb

# TODO: maybe try different models??
# tried yolo11s but was way too slow on my machine lol


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    model_name: str
    epochs: int
    batch: int
    imgsz: int
    workers: int
    val_split: float
    seed: int
    log_interval: int
    project: str
    run_name: str | None
    device: str | None
    save_dir: Path
    verbose: bool


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--project", default="yolo-fine-tune")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        val_split=args.val_split,
        seed=args.seed,
        log_interval=args.log_interval,
        project=args.project,
        run_name=args.run_name,
        device=args.device,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )


def setup_env(seed: int) -> None:
    # print(f"Setting up environment with seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # print("CUDA is available, seeding cuda too")


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config: TrainConfig = config
        self.manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        path: Path = self.config.dataset_dir
        if path.is_file():
            yaml_path = path
            self.root = path.parent
        else:
            yaml_path = path / "data.yaml"
            self.root = path

        if not yaml_path.exists():
            return {}
        with yaml_path.open("r") as f:
            return yaml.safe_load(f) or {}

    def _get_images(self, split: str) -> list[Path]:
        entry = self.manifest.get(split)
        if not entry:
            d = self.root / "images" / split
            if not d.exists():
                return []
            return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        p = Path(entry)
        if not p.is_absolute():
            p = (self.root / p).resolve()

        if p.is_file():
            return [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
        return sorted([i for i in p.iterdir() if i.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    def _prepare_data(self) -> Path:
        """Prepare YOLO data configuration with a fixed train/val split."""
        # print("Preparing data splits...")
        all_train_imgs: list[Path] = self._get_images("train")
        val_imgs: list[Path] = self._get_images("val")
        # print(f"Found {len(all_train_imgs)} training images, {len(val_imgs)} val images")

        if not val_imgs and all_train_imgs:
            random.shuffle(all_train_imgs)
            split_idx: int = max(1, int(len(all_train_imgs) * self.config.val_split))
            val_imgs = all_train_imgs[:split_idx]
            all_train_imgs = all_train_imgs[split_idx:]

        train_imgs = all_train_imgs

        splits_dir: Path = self.root / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        t_list: Path = splits_dir / "train.txt"
        v_list: Path = splits_dir / "val.txt"

        t_list.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n")
        v_list.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n")

        new_manifest: dict[str, Any] = dict(self.manifest)
        new_manifest["path"] = str(self.root.resolve())
        new_manifest["train"] = str(t_list.resolve())
        new_manifest["val"] = str(v_list.resolve())
        self._write_tiered_val_splits(new_manifest, splits_dir)
        if "names" not in new_manifest:
            new_manifest["names"] = ["target"]
        new_manifest["nc"] = len(new_manifest["names"])

        out: Path = self.root / "data_split.yaml"
        with out.open("w") as f:
            yaml.safe_dump(new_manifest, f, sort_keys=False)
        return out

    def _write_tiered_val_splits(self, manifest: dict[str, Any], splits_dir: Path) -> None:
        # this creates separate val sets for each difficulty tier
        # kinda useful to see how the model does on easy vs hard stuff
        difficulty_path = self.root / "difficulty.csv"
        if not difficulty_path.exists():
            return

        tiers: dict[str, list[Path]] = {"easy": [], "medium": [], "hard": []}
        # print(f"Loading difficulty CSV from {difficulty_path}")
        with difficulty_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != "val":
                    continue
                if row.get("negative", "").lower() == "true":
                    continue
                tier = row.get("tier")
                img_path = row.get("image")
                if not tier or not img_path or tier not in tiers:
                    continue
                p = Path(img_path)
                if not p.is_absolute():
                    p = (self.root / p).resolve()
                tiers[tier].append(p)

        for tier, paths in tiers.items():
            if not paths:
                continue
            tier_list = splits_dir / f"val_{tier}.txt"
            tier_list.write_text("\n".join(str(p) for p in paths) + "\n")
            manifest[f"val_{tier}"] = str(tier_list.resolve())

    def _callback(self, log_interval: int) -> Any:
        def on_fit_epoch_end(trainer: Any) -> None:
            epoch: int = getattr(trainer, "epoch", 0) + 1
            if log_interval > 0 and epoch % log_interval != 0:
                return

            metrics: dict = getattr(trainer, "metrics", {}) or {}
            log_data: dict[str, float] = {"epoch": float(epoch)}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"metrics/{k}"] = float(v)

            try:
                log_data["lr"] = float(trainer.optimizer.param_groups[0]["lr"])
            except Exception:
                pass

            wandb.log(log_data, step=epoch)

        return on_fit_epoch_end

    def run(self) -> None:
        data_yaml: Path = self._prepare_data()

        # print(f"Starting training with config: {self.config}")
        # print(f"Data YAML: {data_yaml}")

        # initialize wandb for tracking experiments
        run = wandb.init(
            project=self.config.project,
            name=self.config.run_name,
            config={
                "model": self.config.model_name,
                "epochs": self.config.epochs,
                "batch": self.config.batch,
                "imgsz": self.config.imgsz,
            },
        )

        # load the YOLO model
        model = YOLO(self.config.model_name)
        model.add_callback("on_fit_epoch_end", self._callback(self.config.log_interval))

        # actually train the model
        # NOTE: this takes forever lol, especially with larger models
        model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            workers=self.config.workers,
            project=str(self.config.save_dir),
            name=self.config.run_name,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=self.config.seed,
        )

        run.finish()
        # print("Training finished!")


def main() -> None:
    config: TrainConfig = parse_args()
    setup_env(config.seed)
    Trainer(config).run()


if __name__ == "__main__":
    main()
