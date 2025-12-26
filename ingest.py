import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class IngestConfig:
    ingest_dir: Path
    background_dir: Path
    output_dir: Path
    seed: int
    verbose: bool
    val_split: float


@dataclass
class BoundingBox:
    points: list[tuple[float, float]]

    def to_yolo_format(self, class_id: int) -> str:
        flat: list[float] = []
        for x, y in self.points:
            flat.extend([x, y])
        coords: str = " ".join(f"{v:.6f}" for v in flat)
        return f"{class_id} {coords}"


class DatasetGenerator:
    def __init__(self, config: IngestConfig) -> None:
        self.config: IngestConfig = config
        self.logger: logging.Logger = self._setup_logger()
        self.targets: list[Path] = []
        self.backgrounds: list[Path] = []

    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("DatasetGenerator")
        level: int = logging.DEBUG if self.config.verbose else logging.INFO
        logger.setLevel(level)

        handler: logging.StreamHandler = logging.StreamHandler()
        handler.setLevel(level)

        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _load_files(self) -> None:
        extensions: set[str] = {".jpg", ".jpeg", ".png"}

        if not self.config.ingest_dir.is_dir():
            raise NotADirectoryError(self.config.ingest_dir)
        if not self.config.background_dir.is_dir():
            raise NotADirectoryError(self.config.background_dir)

        self.targets = sorted(
            [p for p in self.config.ingest_dir.iterdir() if p.suffix.lower() in extensions]
        )
        self.backgrounds = sorted(
            [p for p in self.config.background_dir.iterdir() if p.suffix.lower() in extensions]
        )

        if not self.targets:
            raise FileNotFoundError(f"No targets in {self.config.ingest_dir}")
        if not self.backgrounds:
            raise FileNotFoundError(f"No backgrounds in {self.config.background_dir}")

    def _apply_transform(
        self,
        target: Image.Image,
        bg_w: int,
        bg_h: int,
    ) -> tuple[Image.Image, int, int, int, int, float, float, float]:
        min_scale: float = 0.05
        max_scale: float = 0.15

        rotation: float = random.uniform(0.0, 360.0)
        scale: float = random.uniform(min_scale, max_scale)

        sw: int = int(target.width * scale)
        sh: int = int(target.height * scale)

        margin: int = max(sw, sh) + 50
        cx: int = random.randint(margin, max(margin, bg_w - margin))
        cy: int = random.randint(margin, max(margin, bg_h - margin))

        scaled: Image.Image = target.resize((sw, sh), Image.BICUBIC)
        rotated: Image.Image = scaled.rotate(rotation, expand=True, resample=Image.BICUBIC)

        px: int = cx - (rotated.width // 2)
        py: int = cy - (rotated.height // 2)

        return rotated, px, py, cx, cy, float(sw), float(sh), rotation

    def _get_obb(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        rot: float,
        img_w: int,
        img_h: int,
    ) -> BoundingBox:
        hw: float = w / 2.0
        hh: float = h / 2.0
        rad: float = -math.radians(rot)
        cos_a: float = math.cos(rad)
        sin_a: float = math.sin(rad)

        local_corners: list[tuple[float, float]] = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh),
        ]

        normalized_corners: list[tuple[float, float]] = []
        for dx, dy in local_corners:
            rx: float = dx * cos_a - dy * sin_a
            ry: float = dx * sin_a + dy * cos_a
            wx: float = cx + rx
            wy: float = cy + ry

            nx: float = max(0.0, min(wx / img_w, 1.0))
            ny: float = max(0.0, min(wy / img_h, 1.0))
            normalized_corners.append((nx, ny))

        return BoundingBox(normalized_corners)

    def generate(self) -> None:
        random.seed(self.config.seed)
        self._load_files()

        it: Path = self.config.output_dir / "images" / "train"
        lt: Path = self.config.output_dir / "labels" / "train"
        iv: Path = self.config.output_dir / "images" / "val"
        lv: Path = self.config.output_dir / "labels" / "val"

        for p in [it, lt, iv, lv]:
            p.mkdir(parents=True, exist_ok=True)

        pairs: list[tuple[Path, Path]] = []
        for t in self.targets:
            for b in self.backgrounds:
                pairs.append((t, b))

        random.shuffle(pairs)
        val_size: int = max(1, int(len(pairs) * self.config.val_split))

        for i, (t_path, b_path) in enumerate(tqdm(pairs, desc="Generating")):
            is_val: bool = i < val_size
            img_out: Path = iv if is_val else it
            lbl_out: Path = lv if is_val else lt

            target_img: Image.Image = Image.open(t_path).convert("RGBA")
            bg_img: Image.Image = Image.open(b_path).convert("RGBA")

            res = self._apply_transform(target_img, bg_img.width, bg_img.height)
            transformed, px, py, cx, cy, sw, sh, rot = res

            composite: Image.Image = bg_img.copy()
            composite.paste(transformed, (px, py), transformed)

            bbox: BoundingBox = self._get_obb(
                float(cx), float(cy), sw, sh, rot, bg_img.width, bg_img.height
            )

            name: str = "synthetic_{i:06d}"
            composite.convert("RGB").save(img_out / f"{name}.jpg", "JPEG", quality=95)
            (lbl_out / f"{name}.txt").write_text(bbox.to_yolo_format(0) + "\n")

        self._write_yaml(len(pairs), val_size)

    def _write_yaml(self, total: int, val: int) -> None:
        path: Path = self.config.output_dir / "data.yaml"
        data: dict = {
            "path": str(self.config.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["target"],
            "metadata": {
                "seed": self.config.seed,
                "total": total,
                "val": val,
            },
        }
        with path.open("w") as f:
            yaml.dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ingest-dir", required=True, type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("-b", "--background-dir", required=True, type=Path)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)

    args = parser.parse_args()
    config = IngestConfig(
        ingest_dir=args.ingest_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        seed=args.seed,
        verbose=args.verbose,
        val_split=args.val_split,
    )

    generator = DatasetGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
