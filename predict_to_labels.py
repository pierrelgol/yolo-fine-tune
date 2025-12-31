import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# runs inference on images and saves predictions as YOLO labels
# automatically finds the latest trained model or falls back to pretrained


@dataclass(frozen=True)
class PredictConfig:
    model_path: str
    source: Path
    output_dir: Path
    imgsz: int
    conf: float


def find_latest_model() -> str:
    """find the latest trained model or fallback to pretrained."""
    # try to find the most recent trained model
    # if none exists, just use the pretrained one
    runs_dir = Path("runs/train")
    if runs_dir.exists():
        best_models = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_models:
            # print(f"Found trained model: {best_models[0]}")
            return str(best_models[0])

    # fallback to pretrained model if no trained models found
    # print("No trained models found, using pretrained yolo11n-obb.pt")
    return "yolo11n-obb.pt"


def parse_args() -> PredictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--source", type=Path, default=Path("dataset/images/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model_path = args.model if args.model else find_latest_model()

    return PredictConfig(
        model_path=model_path,
        source=args.source,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
    )


class Predictor:
    def __init__(self, config: PredictConfig) -> None:
        self.config: PredictConfig = config
        print(f"Using model: {config.model_path}")
        self.model: YOLO = YOLO(config.model_path)
        self.img_out, self.lbl_out = self._setup_dirs()

    def _setup_dirs(self) -> tuple[Path, Path]:
        img_dir: Path = self.config.output_dir / "images" / "train"
        lbl_dir: Path = self.config.output_dir / "labels" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, lbl_dir

    def _normalize(self, val: float, denom: float) -> float:
        # normalize pixel coords to [0, 1] range
        if denom == 0:
            return 0.0
        return max(0.0, min(1.0, val / denom))

    def _to_yolo_lines(self, result: Any) -> list[str]:
        # convert model predictions to YOLO OBB label format
        h, w = result.orig_shape
        lines: list[str] = []

        # check if result has OBB predictions (oriented bounding boxes)
        if hasattr(result, "obb") and result.obb is not None:
            for pts, cls in zip(result.obb.xyxyxyxy, result.obb.cls, strict=False):
                coords: list[float] = []
                for i, v in enumerate(pts.reshape(-1).tolist()):
                    coords.append(self._normalize(v, w if i % 2 == 0 else h))
                lines.append(f"{int(cls)} " + " ".join(f"{c:.6f}" for c in coords))

        # fallback to regular boxes if no OBB
        # convert axis-aligned boxes to OBB format by making a rectangle
        elif hasattr(result, "boxes") and result.boxes is not None:
            for xyxy, cls in zip(result.boxes.xyxy, result.boxes.cls, strict=False):
                x1, y1, x2, y2 = xyxy.tolist()
                # make a 4-point polygon from the box corners
                pts: list[float] = [x1, y1, x2, y1, x2, y2, x1, y2]
                coords: list[float] = []
                for i, v in enumerate(pts):
                    coords.append(self._normalize(v, w if i % 2 == 0 else h))
                lines.append(f"{int(cls)} " + " ".join(f"{c:.6f}" for c in coords))

        return lines

    def run(self) -> None:
        # run inference on all images
        # print("Starting inference...")
        results = self.model.predict(
            source=str(self.config.source),
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            save=True,
            project=str(self.config.output_dir),
            name="annotated",
        )

        # save predictions as YOLO labels
        for res in results:
            if not res.path:
                continue

            src_path: Path = Path(res.path)
            dst_img: Path = self.img_out / src_path.name
            dst_img.write_bytes(src_path.read_bytes())

            # convert predictions to yolo format and save
            lines: list[str] = self._to_yolo_lines(res)
            if lines:
                (self.lbl_out / f"{dst_img.stem}.txt").write_text("\n".join(lines) + "\n")
                # print(f"Saved {len(lines)} predictions for {dst_img.name}")


def main() -> None:
    config: PredictConfig = parse_args()
    Predictor(config).run()


if __name__ == "__main__":
    main()
