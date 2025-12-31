import argparse
import csv
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# this script generates synthetic training data for YOLO OBB detection
# basically it just pastes target images onto coco128 backgrounds w/ random augmentations
# TODO: add more augmentation types?? maybe color jitter or something


@dataclass(frozen=True)
class IngestConfig:
    ingest_dir: Path
    background_dir: Path
    output_dir: Path
    seed: int
    verbose: bool
    val_split: float
    augmentation_multiplier: int
    mosaic_enabled: bool
    mosaic_size: int
    negative_ratio: float
    difficulty_easy: float
    difficulty_medium: float
    difficulty_hard: float
    geometry_difficulty_cap: float


@dataclass
class BoundingBox:
    points: list[tuple[float, float]]

    def to_yolo_format(self, class_id: int) -> str:
        flat: list[float] = []
        for x, y in self.points:
            flat.extend([x, y])
        coords: str = " ".join(f"{v:.6f}" for v in flat)
        return f"{class_id} {coords}"

    @classmethod
    def from_yolo_line(cls, line: str) -> "BoundingBox":
        """Parse YOLO OBB format line: 'class_id x1 y1 x2 y2 x3 y3 x4 y4'"""
        parts: list[str] = line.strip().split()
        if len(parts) != 9:  # 1 class_id + 8 coordinates
            raise ValueError(f"Invalid YOLO OBB format: expected 9 values, got {len(parts)}")

        # Skip class_id (parts[0]), parse 8 coordinates
        coords: list[float] = [float(parts[i]) for i in range(1, 9)]
        points: list[tuple[float, float]] = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        return cls(points)


class DatasetGenerator:
    def __init__(self, config: IngestConfig) -> None:
        self.config: IngestConfig = config
        self.logger: logging.Logger = self._setup_logger()
        self.targets: list[Path] = []
        self.backgrounds: list[Path] = []
        self.tiers = {
            "easy": (0.00, 0.33),
            "medium": (0.33, 0.66),
            "hard": (0.66, 1.00),
        }

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

        # load all target images (the objects we wanna detect)
        self.targets = sorted(
            [p for p in self.config.ingest_dir.iterdir() if p.suffix.lower() in extensions]
        )
        # load background images (using coco128 dataset)
        self.backgrounds = sorted(
            [p for p in self.config.background_dir.iterdir() if p.suffix.lower() in extensions]
        )
        # print(f"Loaded {len(self.targets)} targets and {len(self.backgrounds)} backgrounds")

        if not self.targets:
            raise FileNotFoundError(f"No targets in {self.config.ingest_dir}")
        if not self.backgrounds:
            raise FileNotFoundError(f"No backgrounds in {self.config.background_dir}")

    def _normalize_mix(self) -> dict[str, float]:
        mix = {
            "easy": self.config.difficulty_easy,
            "medium": self.config.difficulty_medium,
            "hard": self.config.difficulty_hard,
        }
        total = sum(mix.values())
        if total <= 0:
            return {"easy": 0.35, "medium": 0.40, "hard": 0.25}
        return {k: v / total for k, v in mix.items()}

    def _build_tier_list(self, total: int, mix: dict[str, float], shuffle: bool) -> list[str]:
        counts = {k: int(round(total * v)) for k, v in mix.items()}
        count_sum = sum(counts.values())
        if count_sum != total:
            remainder = total - count_sum
            order = ["medium", "easy", "hard"]
            for i in range(abs(remainder)):
                tier = order[i % len(order)]
                counts[tier] += 1 if remainder > 0 else -1

        tiers: list[str] = []
        for tier in ("easy", "medium", "hard"):
            tiers.extend([tier] * max(0, counts[tier]))

        if shuffle:
            random.shuffle(tiers)
        return tiers

    def _sample_difficulty(self, tier: str) -> float:
        low, high = self.tiers[tier]
        if tier == "hard":
            return random.uniform(low, 1.0)
        return random.uniform(low, high)

    def _apply_transform(
        self,
        target: Image.Image,
        bg_w: int,
        bg_h: int,
        perspective_matrix: np.ndarray | None = None,
        difficulty: float = 0.5,
    ) -> tuple[Image.Image, np.ndarray] | None:
        """Applies random scale and rotation with geometry difficulty scaling.

        Args:
            difficulty: 0.0 (easiest) to 1.0 (hardest) for geometry only
                       Easy: Small scale (0.05-0.08), low rotation
                       Hard: Larger scale (0.10-0.18), full rotation
        """
        # scale ranges: smaller objects = easier to fit
        # these values are kinda arbitrary ngl, just tried stuff until it worked
        # TODO: maybe make these configurable??
        easy_min_scale = 0.05
        easy_max_scale = 0.08
        hard_min_scale = 0.10
        hard_max_scale = 0.18

        min_scale = easy_min_scale + (hard_min_scale - easy_min_scale) * difficulty
        max_scale = easy_max_scale + (hard_max_scale - easy_max_scale) * difficulty

        target_array = np.array(target)
        target_w = target.width
        target_h = target.height
        h_p = (
            perspective_matrix.astype(np.float32)
            if perspective_matrix is not None
            else np.eye(3, dtype=np.float32)
        )

        eps: float = 1e-3
        # try up to 20 times to find a valid placement
        # sometimes the random params make stuff go out of bounds which is annoying
        for _ in range(20):
            # rotation range scales with difficulty: 0° to 360°
            rotation_range = 360.0 * difficulty
            rotation: float = random.uniform(-rotation_range / 2, rotation_range / 2)
            scale: float = random.uniform(min_scale, max_scale)
            # print(f"Trying rotation={rotation:.1f}, scale={scale:.3f}")

            sw: int = int(target_w * scale)
            sh: int = int(target_h * scale)
            if sw <= 0 or sh <= 0:
                continue

            theta: float = math.radians(rotation)
            cos_t: float = math.cos(theta)
            sin_t: float = math.sin(theta)
            rot_w: float = abs(sw * cos_t) + abs(sh * sin_t)
            rot_h: float = abs(sw * sin_t) + abs(sh * cos_t)
            if rot_w > bg_w or rot_h > bg_h:
                continue

            margin_x: int = int(math.ceil(rot_w / 2.0))
            margin_y: int = int(math.ceil(rot_h / 2.0))
            max_cx: int = bg_w - margin_x
            max_cy: int = bg_h - margin_y
            if max_cx < margin_x or max_cy < margin_y:
                continue

            cx: int = random.randint(margin_x, max_cx)
            cy: int = random.randint(margin_y, max_cy)

            center = (sw / 2.0, sh / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
            rot_mat[0, 2] += cx - center[0]
            rot_mat[1, 2] += cy - center[1]

            scale_mat = np.array(
                [
                    [sw / float(target_w), 0.0, 0.0],
                    [0.0, sh / float(target_h), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            aff = np.array(
                [
                    [rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2]],
                    [rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            transform = aff @ scale_mat @ h_p

            corners = np.array(
                [[[0.0, 0.0], [float(target_w), 0.0], [float(target_w), float(target_h)], [0.0, float(target_h)]]],
                dtype=np.float32,
            )
            transformed_corners = cv2.perspectiveTransform(corners, transform)[0]
            xs = transformed_corners[:, 0]
            ys = transformed_corners[:, 1]
            if xs.min() < -eps or ys.min() < -eps or xs.max() > (bg_w - 1 + eps) or ys.max() > (bg_h - 1 + eps):
                continue

            transformed = cv2.warpPerspective(
                target_array,
                transform,
                (bg_w, bg_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            return Image.fromarray(transformed), transform

        return None

    def _apply_perspective_warp(self, img: Image.Image, difficulty: float = 0.5) -> np.ndarray | None:
        """Generate a random perspective transform matrix with geometry difficulty scaling.

        Args:
            difficulty: 0.0 (easiest) to 1.0 (hardest) for geometry only
        """
        # probability increases with difficulty: ~20% at diff=0, ~80% at diff=1
        # not every image gets perspective warp, depends on difficulty
        apply_prob = 0.2 + 0.6 * difficulty
        if random.random() > apply_prob:
            return None

        h, w = img.height, img.width

        # strength scales with difficulty: 0.02 (mild) to 0.20 (strong)
        # these values seem to work well, anything higher looks weird af
        min_strength = 0.02
        max_strength = 0.20
        strength = min_strength + (max_strength - min_strength) * difficulty
        # print(f"Applying perspective warp with strength {strength:.3f}")

        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Define destination points with random distortion
        dst_points = np.float32([
            [random.uniform(0, w * strength), random.uniform(0, h * strength)],
            [w - random.uniform(0, w * strength), random.uniform(0, h * strength)],
            [w - random.uniform(0, w * strength), h - random.uniform(0, h * strength)],
            [random.uniform(0, w * strength), h - random.uniform(0, h * strength)],
        ])

        # Get perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return matrix

    def _apply_noise_augmentation(
        self, img: Image.Image, aug_type: int, difficulty: float = 0.5
    ) -> Image.Image:
        """Apply noise/blur with appearance difficulty scaling.

        Args:
            difficulty: 0.0 (easiest) to 1.0 (hardest) for appearance only
        """
        if aug_type == 0:
            # Original - no augmentation
            return img

        # skip augmentation if difficulty is too low for this aug type
        # aug_type 1 (noise) requires diff >= 0.25
        # aug_type 2 (blur) requires diff >= 0.50
        # aug_type 3 (brightness) requires diff >= 0.75
        # TODO: maybe add more augmentation types?? hue shift? saturation?
        aug_threshold = aug_type * 0.25
        if difficulty < aug_threshold:
            return img

        img_array = np.array(img)
        # print(f"Applying augmentation type {aug_type} with difficulty {difficulty:.2f}")

        if aug_type == 1:
            # gaussian noise sigma scales with difficulty
            # tried different sigma ranges, these seem ok
            sigma_min, sigma_max = 3, 20
            sigma = sigma_min + (sigma_max - sigma_min) * difficulty
            noise = np.random.normal(0, sigma, img_array.shape).astype(np.int16)
            noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)

        elif aug_type == 2:
            # Blur - radius scales with difficulty
            img_pil = Image.fromarray(img_array)
            blur_min, blur_max = 0.3, 3.0
            blur_radius = blur_min + (blur_max - blur_min) * difficulty
            return img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        elif aug_type == 3:
            # Brightness & contrast variation -> range scales with difficulty
            img_pil = Image.fromarray(img_array)

            # Variation increases with difficulti
            brightness_range = 0.2 + 0.4 * difficulty  # ±0.2 to ±0.6
            contrast_range = 0.1 + 0.3 * difficulty    # ±0.1 to ±0.4

            brightness_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
            contrast_factor = 1.0 + random.uniform(-contrast_range, contrast_range)

            brightness = ImageEnhance.Brightness(img_pil)
            img_pil = brightness.enhance(brightness_factor)
            contrast = ImageEnhance.Contrast(img_pil)
            return contrast.enhance(contrast_factor)

        return img

    def _extract_obb_from_mask(
        self,
        mask: np.ndarray,
        paste_x: int,
        paste_y: int,
        img_w: int,
        img_h: int,
    ) -> BoundingBox | None:
        """Extract OBB from alpha mask in BACKGROUND SPACE.

        This correctly handles:
        - Perspective warp transparent padding (ignored)
        - Partial occlusion (naturally clipped)

        The key insight: OBB must be computed from the pixels that actually landed
        on the background, not from the target-space mask. This replicates PIL's
        paste clipping behavior exactly.

        Args:
            mask: Alpha channel of the transformed target (2D array, target-space)
            paste_x: Top-left x coordinate where target was pasted
            paste_y: Top-left y coordinate where target was pasted
            img_w: Background image width
            img_h: Background image height

        Returns:
            BoundingBox with 4 normalized corners, or None if mask is empty
        """
        # NOTE: this took me way too long to figure out lmao
        # the trick is to work in background space, not target space
        # otherwise the bboxes are completely wrong when objects get clipped at the edges

        # 1. create background-sized alpha canvas (all zeros)
        bg_alpha = np.zeros((img_h, img_w), dtype=np.uint8)

        # 2. Compute paste region bounds (handles negative px/py)
        mask_h, mask_w = mask.shape

        # Source region (what part of mask to copy)
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(mask_w, img_w - paste_x)
        src_y2 = min(mask_h, img_h - paste_y)

        # Destination region (where to paste in background)
        # print(f"{src_x1}{src_x2}{src_y1}{src_y2}")
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Safety check: valid region?
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None  # Completely outside frame

        # 3. Paste visible portion into background canvas
        #    This is EXACTLY what PIL does we replicate clipping behavior
        bg_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

        # 4. Threshold in BACKGROUND SPACE
        _, binary = cv2.threshold(bg_alpha, 10, 255, cv2.THRESH_BINARY)

        # 5. Find contours in BACKGROUND SPACE
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # No visible pixels

        # 6. extract quadrilateral from the visible mask in BACKGROUND SPACE
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        perimeter = cv2.arcLength(hull, True)
        quad: np.ndarray | None = None
        # try different epsilon values to approximate the contour as a quad
        # this usually works but sometimes fails if the shape is too weird
        for eps_factor in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break
                # print(f"Found quad with epsilon={eps_factor}")

        if quad is None:
            # wtf why cant we find a quad
            # print("Failed to find a 4-point quad :(")
            self.logger.warning("Skipping sample: could not approximate a 4-point quad from mask")
            return None

        center = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        ordered = quad[np.argsort(angles)]

        # 7. Clamp to image bounds, then normalize to [0, 1]
        clamped_corners = [
            (
                max(0.0, min(float(img_w - 1), float(x))),
                max(0.0, min(float(img_h - 1), float(y))),
            )
            for x, y in ordered
        ]
        normalized_corners = [
            (x / img_w, y / img_h) for x, y in clamped_corners
        ]

        return BoundingBox(normalized_corners)

    def _extract_obb_from_transform(
        self,
        target_w: int,
        target_h: int,
        transform: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> BoundingBox:
        """Compute exact OBB corners by applying the same transforms as the image."""
        corners = np.array(
            [[[0.0, 0.0], [float(target_w), 0.0], [float(target_w), float(target_h)], [0.0, float(target_h)]]],
            dtype=np.float32,
        )
        transformed = cv2.perspectiveTransform(corners, transform)[0]
        placed = [(float(x), float(y)) for x, y in transformed]

        clamped_corners = [
            (
                max(0.0, min(float(img_w - 1), float(x))),
                max(0.0, min(float(img_h - 1), float(y))),
            )
            for x, y in placed
        ]
        normalized_corners = [
            (x / img_w, y / img_h) for x, y in clamped_corners
        ]

        return BoundingBox(normalized_corners)

    def _transform_bbox_to_quadrant(
        self,
        bbox: BoundingBox,
        src_w: int,
        src_h: int,
        offset_x: int,
        offset_y: int,
        quad_w: int,
        quad_h: int,
        mosaic_size: int,
    ) -> BoundingBox:
        """
        Transform normalized bbox from source image to mosaic quadrant space.

        Args:
            bbox: Source bounding box (normalized 0-1 coordinates)
            src_w, src_h: Source image dimensions in pixels
            offset_x, offset_y: Quadrant top-left position in mosaic (pixels)
            quad_w, quad_h: Quadrant dimensions in pixels
            mosaic_size: Mosaic image dimension (square)

        Returns:
            Transformed bounding box in mosaic space (normalized 0-1)
        """
        transformed_points: list[tuple[float, float]] = []

        for nx, ny in bbox.points:
            # Step 1: Denormalize to source pixel coordinates
            px: float = nx * src_w
            py: float = ny * src_h

            # Step 2: Scale to quadrant dimensions (handles resize)
            scale_x: float = quad_w / src_w
            scale_y: float = quad_h / src_h
            qx: float = px * scale_x
            qy: float = py * scale_y

            # Step 3: Offset to quadrant position in mosaic
            mx: float = qx + offset_x
            my: float = qy + offset_y

            # Step 4: Normalize to mosaic space
            nx_new: float = mx / mosaic_size
            ny_new: float = my / mosaic_size

            # Step 5: Clamp to [0, 1] (safety measure)
            nx_new = max(0.0, min(1.0, nx_new))
            ny_new = max(0.0, min(1.0, ny_new))

            transformed_points.append((nx_new, ny_new))

        return BoundingBox(transformed_points)

    def _create_mosaic(
        self,
        image_paths: list[Path],
        label_paths: list[Path],
        mosaic_size: int,
    ) -> tuple[Image.Image, list[BoundingBox]]:
        """
        Create 2x2 mosaic from 4 source images.

        Args:
            image_paths: List of 4 image file paths
            label_paths: List of 4 corresponding label file paths
            mosaic_size: Target dimension for square mosaic output

        Returns:
            Tuple of (mosaic_image, list_of_bboxes)
        """
        if len(image_paths) != 4 or len(label_paths) != 4:
            raise ValueError(f"Expected 4 images and labels, got {len(image_paths)} and {len(label_paths)}")

        # random split point (yolo-style: 40-60% of mosaic_size)
        # this makes the quadrants different sizes which is kinda cool
        split_x: int = random.randint(int(mosaic_size * 0.4), int(mosaic_size * 0.6))
        split_y: int = random.randint(int(mosaic_size * 0.4), int(mosaic_size * 0.6))
        # print(f"Mosaic split at x={split_x}, y={split_y}")

        # Quadrant bounds (x1, y1, x2, y2)
        quadrants: list[tuple[int, int, int, int]] = [
            (0, 0, split_x, split_y),  # top-left
            (split_x, 0, mosaic_size, split_y),  # top-right
            (0, split_y, split_x, mosaic_size),  # bottom-left
            (split_x, split_y, mosaic_size, mosaic_size),  # bottom-right
        ]

        # Create blank canvas
        mosaic: Image.Image = Image.new("RGB", (mosaic_size, mosaic_size), (0, 0, 0))
        all_bboxes: list[BoundingBox] = []

        for i in range(4):
            # Load source image
            src_img: Image.Image = Image.open(image_paths[i]).convert("RGB")
            src_w, src_h = src_img.size

            # Get quadrant bounds
            x1, y1, x2, y2 = quadrants[i]
            quad_w: int = x2 - x1
            quad_h: int = y2 - y1

            # Resize source to fill quadrant
            resized: Image.Image = src_img.resize((quad_w, quad_h), Image.BICUBIC)

            # Paste into mosaic
            mosaic.paste(resized, (x1, y1))

            # Transform bounding boxes
            if label_paths[i].exists():
                label_text: str = label_paths[i].read_text()
                for line in label_text.splitlines():
                    line = line.strip()
                    if line:
                        bbox: BoundingBox = BoundingBox.from_yolo_line(line)
                        transformed_bbox: BoundingBox = self._transform_bbox_to_quadrant(
                            bbox, src_w, src_h, x1, y1, quad_w, quad_h, mosaic_size
                        )
                        all_bboxes.append(transformed_bbox)

        return mosaic, all_bboxes

    def generate(self) -> None:
        # set random seeds for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)  # Also seed numpy for reproducibility
        self._load_files()
        mix = self._normalize_mix()

        # setup output directories
        it: Path = self.config.output_dir / "images" / "train"
        lt: Path = self.config.output_dir / "labels" / "train"
        iv: Path = self.config.output_dir / "images" / "val"
        lv: Path = self.config.output_dir / "labels" / "val"

        for p in [it, lt, iv, lv]:
            p.mkdir(parents=True, exist_ok=True)

        # Generate base pairs (all combinations of targets x backgrounds)
        pairs: list[tuple[Path, Path]] = []
        for t in self.targets:
            for b in self.backgrounds:
                pairs.append((t, b))

        random.shuffle(pairs)
        # print(f"Generated {len(pairs)} target-background pairs")

        # split pairs into train/val BEFORE generation (prevents leakage)
        # IMPORTANT: we split the pairs first then generate augmentations
        # otherwise same target+background could show up in both train and val which would be bad
        val_pair_count = max(1, int(len(pairs) * self.config.val_split))
        val_pairs = pairs[:val_pair_count]
        train_pairs = pairs[val_pair_count:]
        # print(f"Split: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs")

        # Track base images separately by split
        train_base_paths: list[Path] = []
        train_label_paths: list[Path] = []
        val_base_paths: list[Path] = []
        val_label_paths: list[Path] = []

        train_total = len(train_pairs) * self.config.augmentation_multiplier
        val_total = len(val_pairs) * self.config.augmentation_multiplier
        train_tiers = self._build_tier_list(train_total, mix, shuffle=True)
        val_tiers = self._build_tier_list(val_total, mix, shuffle=False)

        difficulty_records: list[dict[str, str]] = []

        # PHASE 1A: generate TRAIN base images
        # for each target-background pair we create multiple augmented versions
        train_counter = 0
        desc = f"Generating train base images ({len(train_pairs)} pairs × {self.config.augmentation_multiplier})"
        train_sample_idx = 0
        for t_path, b_path in tqdm(train_pairs, desc=desc):
            target_img: Image.Image = Image.open(t_path).convert("RGBA")
            bg_img: Image.Image = Image.open(b_path).convert("RGBA")

            # create multiple augmented versions of each pair
            for aug_idx in range(self.config.augmentation_multiplier):
                tier = train_tiers[train_sample_idx]
                difficulty = self._sample_difficulty(tier)
                # cap geometry difficulty so objects dont get too rotated/scaled
                # appearance difficulty is uncapped tho
                geometry_difficulty = min(difficulty, self.config.geometry_difficulty_cap)
                # print(f"Sample {train_counter}: tier={tier}, diff={difficulty:.2f}, geom_diff={geometry_difficulty:.2f}")

                # Apply perspective warp with geometry difficulty
                perspective_matrix = self._apply_perspective_warp(target_img, geometry_difficulty)

                # Apply transform with geometry difficulty
                res = self._apply_transform(
                    target_img, bg_img.width, bg_img.height, perspective_matrix, geometry_difficulty
                )
                if res is None:
                    self.logger.warning(f"Skipping train sample {train_counter}: transformed target exceeds background")
                    train_counter += 1
                    train_sample_idx += 1
                    continue
                transformed, transform = res

                # Paste and extract bbox from transform math
                composite: Image.Image = bg_img.copy()
                composite.paste(transformed, (0, 0), transformed)

                bbox = self._extract_obb_from_transform(
                    target_img.width,
                    target_img.height,
                    transform,
                    bg_img.width,
                    bg_img.height,
                )

                # Apply noise augmentation AFTER bbox extraction (appearance difficulty)
                augmented = self._apply_noise_augmentation(composite, aug_idx, difficulty)

                # Save to TRAIN directories (encode difficulty in filename)
                name: str = f"synthetic_{train_counter:06d}_d{int(difficulty*100):02d}"
                img_path: Path = it / f"{name}.jpg"
                lbl_path: Path = lt / f"{name}.txt"

                augmented.convert("RGB").save(img_path, "JPEG", quality=95)
                lbl_path.write_text(bbox.to_yolo_format(0) + "\n")

                train_base_paths.append(img_path)
                train_label_paths.append(lbl_path)
                difficulty_records.append({
                    "split": "train",
                    "image": str(img_path),
                    "label": str(lbl_path),
                    "tier": tier,
                    "difficulty": f"{difficulty:.4f}",
                    "negative": "false",
                })
                train_counter += 1
                train_sample_idx += 1

        # PHASE 1B: Generate VAL base images
        val_counter = 0
        desc = f"Generating val base images ({len(val_pairs)} pairs × {self.config.augmentation_multiplier})"
        val_sample_idx = 0
        for t_path, b_path in tqdm(val_pairs, desc=desc):
            target_img: Image.Image = Image.open(t_path).convert("RGBA")
            bg_img: Image.Image = Image.open(b_path).convert("RGBA")

            for aug_idx in range(self.config.augmentation_multiplier):
                tier = val_tiers[val_sample_idx]
                difficulty = self._sample_difficulty(tier)
                geometry_difficulty = min(difficulty, self.config.geometry_difficulty_cap)

                # Apply perspective warp with geometry difficulty
                perspective_matrix = self._apply_perspective_warp(target_img, geometry_difficulty)

                # Apply transform with geometry difficulty
                res = self._apply_transform(
                    target_img, bg_img.width, bg_img.height, perspective_matrix, geometry_difficulty
                )
                if res is None:
                    self.logger.warning(f"Skipping val sample {val_counter}: transformed target exceeds background")
                    val_counter += 1
                    val_sample_idx += 1
                    continue
                transformed, transform = res

                # Paste and extract bbox from transform math
                composite: Image.Image = bg_img.copy()
                composite.paste(transformed, (0, 0), transformed)

                bbox = self._extract_obb_from_transform(
                    target_img.width,
                    target_img.height,
                    transform,
                    bg_img.width,
                    bg_img.height,
                )

                # Apply noise augmentation AFTER bbox extraction (appearance difficulty)
                augmented = self._apply_noise_augmentation(composite, aug_idx, difficulty)

                # Save to VAL directories (encode difficulty in filename)
                name: str = f"synthetic_{val_counter:06d}_d{int(difficulty*100):02d}"
                img_path: Path = iv / f"{name}.jpg"
                lbl_path: Path = lv / f"{name}.txt"

                augmented.convert("RGB").save(img_path, "JPEG", quality=95)
                lbl_path.write_text(bbox.to_yolo_format(0) + "\n")

                val_base_paths.append(img_path)
                val_label_paths.append(lbl_path)
                difficulty_records.append({
                    "split": "val",
                    "image": str(img_path),
                    "label": str(lbl_path),
                    "tier": tier,
                    "difficulty": f"{difficulty:.4f}",
                    "negative": "false",
                })
                val_counter += 1
                val_sample_idx += 1

        # PHASE 1C: Generate negative (background-only) samples
        neg_train_count = int(round(train_counter * self.config.negative_ratio))
        neg_val_count = int(round(val_counter * self.config.negative_ratio))

        neg_train_tiers = self._build_tier_list(neg_train_count, mix, shuffle=True)
        for i in range(neg_train_count):
            tier = neg_train_tiers[i]
            difficulty = self._sample_difficulty(tier)
            bg_path = random.choice(self.backgrounds)
            bg_img = Image.open(bg_path).convert("RGB")
            aug_type = random.randint(0, 3)
            bg_img = self._apply_noise_augmentation(bg_img, aug_type, difficulty)
            name: str = f"negative_{i:06d}_d{int(difficulty*100):02d}"
            img_path: Path = it / f"{name}.jpg"
            lbl_path: Path = lt / f"{name}.txt"
            bg_img.save(img_path, "JPEG", quality=95)
            lbl_path.write_text("")
            train_base_paths.append(img_path)
            train_label_paths.append(lbl_path)
            difficulty_records.append({
                "split": "train",
                "image": str(img_path),
                "label": str(lbl_path),
                "tier": tier,
                "difficulty": f"{difficulty:.4f}",
                "negative": "true",
            })

        neg_val_tiers = self._build_tier_list(neg_val_count, mix, shuffle=False)
        for i in range(neg_val_count):
            tier = neg_val_tiers[i]
            difficulty = self._sample_difficulty(tier)
            bg_path = random.choice(self.backgrounds)
            bg_img = Image.open(bg_path).convert("RGB")
            aug_type = random.randint(0, 3)
            bg_img = self._apply_noise_augmentation(bg_img, aug_type, difficulty)
            name: str = f"negative_{i:06d}_d{int(difficulty*100):02d}"
            img_path: Path = iv / f"{name}.jpg"
            lbl_path: Path = lv / f"{name}.txt"
            bg_img.save(img_path, "JPEG", quality=95)
            lbl_path.write_text("")
            val_base_paths.append(img_path)
            val_label_paths.append(lbl_path)
            difficulty_records.append({
                "split": "val",
                "image": str(img_path),
                "label": str(lbl_path),
                "tier": tier,
                "difficulty": f"{difficulty:.4f}",
                "negative": "true",
            })

        difficulty_path = self.config.output_dir / "difficulty.csv"
        with difficulty_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "image", "label", "tier", "difficulty", "negative"],
            )
            writer.writeheader()
            writer.writerows(difficulty_records)

        # PHASE 2: generate mosaic images if enabled (separate per split)
        # mosaics help the model learn objects at different scales n positions
        if self.config.mosaic_enabled:
            # PHASE 2A: generate TRAIN mosaics (sample from train_base_paths ONLY)
            # important: only use training images for training mosaics!! no leakage
            train_mosaic_count = len(train_base_paths)
            desc = f"Generating train mosaics ({train_mosaic_count})"
            for mosaic_idx in tqdm(range(train_mosaic_count), desc=desc):
                # select 4 random TRAIN images to combine
                indices: list[int] = random.choices(range(len(train_base_paths)), k=4)
                selected_imgs: list[Path] = [train_base_paths[i] for i in indices]
                selected_lbls: list[Path] = [train_label_paths[i] for i in indices]
                # print(f"Creating mosaic from images: {[img.name for img in selected_imgs]}")

                mosaic_img, mosaic_bboxes = self._create_mosaic(
                    selected_imgs, selected_lbls, self.config.mosaic_size
                )

                # Save to TRAIN directories
                name: str = f"mosaic_{mosaic_idx:06d}"
                mosaic_img.save(it / f"{name}.jpg", "JPEG", quality=95)
                label_lines: list[str] = [bbox.to_yolo_format(0) for bbox in mosaic_bboxes]
                (lt / f"{name}.txt").write_text("\n".join(label_lines) + "\n")

            # PHASE 2B: Generate VAL mosaics (sample from val_base_paths ONLY)
            val_mosaic_count = len(val_base_paths)
            desc = f"Generating val mosaics ({val_mosaic_count})"
            for mosaic_idx in tqdm(range(val_mosaic_count), desc=desc):
                # Select from VAL images only
                indices: list[int] = random.choices(range(len(val_base_paths)), k=4)
                selected_imgs: list[Path] = [val_base_paths[i] for i in indices]
                selected_lbls: list[Path] = [val_label_paths[i] for i in indices]

                mosaic_img, mosaic_bboxes = self._create_mosaic(
                    selected_imgs, selected_lbls, self.config.mosaic_size
                )

                # Save to VAL directories
                name: str = f"mosaic_{mosaic_idx:06d}"
                mosaic_img.save(iv / f"{name}.jpg", "JPEG", quality=95)
                label_lines: list[str] = [bbox.to_yolo_format(0) for bbox in mosaic_bboxes]
                (lv / f"{name}.txt").write_text("\n".join(label_lines) + "\n")

        # calculate total counts
        # base images + mosaic images (if enabled)
        base_total = train_counter + val_counter + neg_train_count + neg_val_count
        total_images = base_total * (2 if self.config.mosaic_enabled else 1)
        val_total = val_counter * (2 if self.config.mosaic_enabled else 1)
        # print(f"Total dataset size: {total_images} images ({val_total} val)")

        self._write_yaml(total_images, val_total)
        # print("Done! Dataset generated successfully")

    def _write_yaml(self, total: int, val: int) -> None:
        path: Path = self.config.output_dir / "data.yaml"

        metadata: dict = {
            "seed": self.config.seed,
            "total": total,
            "val": val,
            "difficulty_mix": {
                "easy": self.config.difficulty_easy,
                "medium": self.config.difficulty_medium,
                "hard": self.config.difficulty_hard,
            },
            "difficulty_tiers": {
                "easy": [0.00, 0.33],
                "medium": [0.33, 0.66],
                "hard": [0.66, 1.00],
            },
            "geometry_difficulty_cap": self.config.geometry_difficulty_cap,
            "difficulty_csv": "difficulty.csv",
        }

        # Add mosaic metadata if enabled
        if self.config.mosaic_enabled:
            base_count = total // 2
            metadata["base_images"] = base_count
            metadata["mosaic_images"] = base_count
            metadata["mosaic_enabled"] = True
            metadata["mosaic_size"] = self.config.mosaic_size

        data: dict = {
            "path": str(self.config.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["target"],
            "metadata": metadata,
        }
        with path.open("w") as f:
            yaml.dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ingest-dir", type=Path, default=Path("ingest"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("-b", "--background-dir", type=Path, default=Path("augments"))
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--aug-multiplier", type=int, default=6, help="Number of augmented versions per base image")
    parser.add_argument("--mosaic", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable mosaic augmentation")
    parser.add_argument("--mosaic-size", type=int, default=640, help="Target size for mosaic images")
    parser.add_argument("--negative-ratio", type=float, default=0.2, help="Background-only samples as a ratio of positives per split")
    parser.add_argument("--difficulty-easy", type=float, default=0.35, help="Easy sample ratio")
    parser.add_argument("--difficulty-medium", type=float, default=0.40, help="Medium sample ratio")
    parser.add_argument("--difficulty-hard", type=float, default=0.25, help="Hard sample ratio")
    parser.add_argument("--geometry-cap", type=float, default=0.6, help="Max geometry difficulty (0-1)")

    args = parser.parse_args()
    config = IngestConfig(
        ingest_dir=args.ingest_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        seed=args.seed,
        verbose=args.verbose,
        val_split=args.val_split,
        augmentation_multiplier=args.aug_multiplier,
        mosaic_enabled=args.mosaic,
        mosaic_size=args.mosaic_size,
        negative_ratio=args.negative_ratio,
        difficulty_easy=args.difficulty_easy,
        difficulty_medium=args.difficulty_medium,
        difficulty_hard=args.difficulty_hard,
        geometry_difficulty_cap=args.geometry_cap,
    )

    generator = DatasetGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
