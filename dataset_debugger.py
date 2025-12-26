import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

OBB_PARTS_COUNT = 9


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    points: list[tuple[float, float]]

    @classmethod
    def from_line(cls, line: str) -> "YoloLabel":
        parts: list[str] = line.strip().split()
        if len(parts) != OBB_PARTS_COUNT:
            raise ValueError(f"Invalid OBB: {line}")

        cid: int = int(parts[0])
        raw: list[float] = list(map(float, parts[1:]))
        pts: list[tuple[float, float]] = [(raw[i], raw[i + 1]) for i in range(0, 8, 2)]
        return cls(cid, pts)

    def to_pixels(self, w: int, h: int) -> list[tuple[int, int]]:
        return [(int(px * w), int(py * h)) for px, py in self.points]


class Debugger:
    def __init__(self, dataset_dir: Path, verbose: bool) -> None:
        self.dir: Path = dataset_dir
        self.logger: logging.Logger = self._setup_logger(verbose)
        self.files: list[tuple[Path, Path]] = self._scan()
        self.index: int = 0

    def _setup_logger(self, verbose: bool) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("Debugger")
        level: int = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)
        h: logging.StreamHandler = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        return logger

    def _scan(self) -> list[tuple[Path, Path]]:
        idr: Path = self.dir / "images" / "train"
        ldr: Path = self.dir / "labels" / "train"
        exts: set[str] = {".jpg", ".png"}
        imgs: list[Path] = sorted([p for p in idr.iterdir() if p.suffix.lower() in exts])
        return [(p, ldr / f"{p.stem}.txt") for p in imgs]

    def _draw(self, img_path: Path, lbl_path: Path) -> np.ndarray:
        img: np.ndarray = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        lines: list[str] = []
        if lbl_path.exists():
            lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]

        labels: list[YoloLabel] = [YoloLabel.from_line(ln) for ln in lines]

        for lb in labels:
            pts: list[tuple[int, int]] = lb.to_pixels(w, h)
            poly: np.ndarray = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [poly], True, (0, 255, 0), 2)
            cv2.putText(
                img, f"{lb.class_id}", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        bar: np.ndarray = np.zeros((50, w, 3), dtype=np.uint8)
        txt: str = f"[{self.index + 1}/{len(self.files)}] {img_path.name} (Q: quit, L/R: nav)"
        cv2.putText(bar, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return np.vstack([bar, img])

    def run(self) -> None:
        if not self.files:
            return

        win: str = "Debugger"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            ip, lp = self.files[self.index]
            view: np.ndarray = self._draw(ip, lp)
            cv2.imshow(win, view)

            key: int = cv2.waitKey(0) & 0xFF
            if key in {ord("q"), 27}:
                break
            elif key in {81, 2, ord("a")}:
                self.index = max(0, self.index - 1)
            elif key in {83, 3, ord("d")}:
                self.index = min(len(self.files) - 1, self.index + 1)

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    Debugger(args.dataset_dir, args.verbose).run()


if __name__ == "__main__":
    main()
