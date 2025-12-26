import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass(frozen=True)
class PredictConfig:
    model_path: str
    source: Path
    output_dir: Path
    imgsz: int
    conf: float


def parse_args() -> PredictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    return PredictConfig(
        model_path=args.model,
        source=args.source,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
    )


class Predictor:
    def __init__(self, config: PredictConfig) -> None:
        self.config: PredictConfig = config
        self.model: YOLO = YOLO(config.model_path)
        self.img_out, self.lbl_out = self._setup_dirs()

    def _setup_dirs(self) -> tuple[Path, Path]:
        img_dir: Path = self.config.output_dir / "images" / "train"
        lbl_dir: Path = self.config.output_dir / "labels" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        return img_dir, lbl_dir

    def _normalize(self, val: float, denom: float) -> float:
        if denom == 0:
            return 0.0
        return max(0.0, min(1.0, val / denom))

    def _to_yolo_lines(self, result: Any) -> list[str]:
        h, w = result.orig_shape
        lines: list[str] = []

        if hasattr(result, "obb") and result.obb is not None:
            for pts, cls in zip(result.obb.xyxyxyxy, result.obb.cls, strict=False):
                coords: list[float] = []
                for i, v in enumerate(pts.reshape(-1).tolist()):
                    coords.append(self._normalize(v, w if i % 2 == 0 else h))
                lines.append(f"{int(cls)} " + " ".join(f"{c:.6f}" for c in coords))

        elif hasattr(result, "boxes") and result.boxes is not None:
            for xyxy, cls in zip(result.boxes.xyxy, result.boxes.cls, strict=False):
                x1, y1, x2, y2 = xyxy.tolist()
                pts: list[float] = [x1, y1, x2, y1, x2, y2, x1, y2]
                coords: list[float] = []
                for i, v in enumerate(pts):
                    coords.append(self._normalize(v, w if i % 2 == 0 else h))
                lines.append(f"{int(cls)} " + " ".join(f"{c:.6f}" for c in coords))

        return lines

    def run(self) -> None:
        results = self.model.predict(
            source=str(self.config.source),
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            save=True,
            project=str(self.config.output_dir),
            name="annotated",
        )

        for res in results:
            if not res.path:
                continue

            src_path: Path = Path(res.path)
            dst_img: Path = self.img_out / src_path.name
            dst_img.write_bytes(src_path.read_bytes())

            lines: list[str] = self._to_yolo_lines(res)
            if lines:
                (self.lbl_out / f"{dst_img.stem}.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    config: PredictConfig = parse_args()
    Predictor(config).run()


if __name__ == "__main__":
    main()
