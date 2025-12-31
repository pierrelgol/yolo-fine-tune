#!/usr/bin/env python3
"""Quick verification script to check if bounding boxes are correctly aligned."""
# basically just visualizes a few samples to make sure the bboxes look right
# saves them to bbox_verification/ directory

import cv2
import numpy as np
from pathlib import Path


def visualize_sample(image_path: Path, label_path: Path, output_path: Path):
    """visualize an image with its bounding box overlay."""
    # read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return

    h, w = img.shape[:2]
    # print(f"Image size: {w}x{h}")

    # read label file
    if not label_path.exists():
        print(f"Label not found: {label_path}")
        return

    label_text = label_path.read_text().strip()
    if not label_text:
        print(f"Empty label: {label_path}")
        return

    # parse OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    parts = label_text.split()
    if len(parts) != 9:
        print(f"Invalid label format: {label_path}")
        return

    # get normalized coordinates
    coords = [float(parts[i]) for i in range(1, 9)]

    # denormalize to pixel coordinates
    points = []
    for i in range(0, 8, 2):
        x = int(coords[i] * w)
        y = int(coords[i + 1] * h)
        points.append((x, y))

    # draw bounding box (4 corners)
    points_np = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points_np], isClosed=True, color=(0, 255, 0), thickness=3)

    # draw corner points with labels
    # this helps see if the corners are in the right order
    for i, pt in enumerate(points):
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
        cv2.putText(img, str(i), (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # save visualization
    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization to {output_path}")


def main():
    # just grab a few samples and visualize them
    dataset_dir = Path("dataset")
    output_dir = Path("bbox_verification")
    output_dir.mkdir(exist_ok=True)

    # get first 5 training images to check
    # adjust this if you wanna check more or different ones
    train_images = sorted((dataset_dir / "images" / "train").glob("*.jpg"))[:5]
    # print(f"Found {len(train_images)} images to verify")

    for img_path in train_images:
        label_path = dataset_dir / "labels" / "train" / f"{img_path.stem}.txt"
        output_path = output_dir / f"verified_{img_path.name}"

        visualize_sample(img_path, label_path, output_path)

    print(f"\nVerification complete! Check {output_dir} for visualizations.")


if __name__ == "__main__":
    main()
