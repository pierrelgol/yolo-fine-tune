#!/usr/bin/env python3
"""Monitor YOLO training progress and provide analysis."""
# watches the training run and shows metrics in real-time
# super helpful for checking if training is actually working lol
# run with --once to just see current status, or let it loop to watch live

import argparse
import csv
import re
import time
from pathlib import Path


def parse_epoch_metrics(line):
    """parse metrics from epoch line."""
    # example line: "1/50      4.43G     0.9249      1.281      1.971         16        640"
    # this regex is kinda brittle but it works for now
    pattern = r'(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'total_epochs': int(match.group(2)),
            'box_loss': float(match.group(3)),
            'cls_loss': float(match.group(4)),
            'dfl_loss': float(match.group(5)),
        }
    return None


def parse_val_metrics(line):
    """parse validation metrics."""
    # example: "all        204        510      0.988       0.99      0.995      0.885"
    pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'precision': float(match.group(1)),
            'recall': float(match.group(2)),
            'mAP50': float(match.group(3)),
            'mAP50-95': float(match.group(4)),
        }
    return None


def find_latest_results(runs_dir: Path) -> Path | None:
    # find the most recent results.csv file
    # useful when you have multiple training runs
    candidates = list(runs_dir.glob("**/results.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("epoch")]


def summarize_results(rows: list[dict[str, str]]) -> None:
    # print a nice summary of training progress
    if not rows:
        print("No epoch data found yet. Training may just be starting...")
        return

    latest = rows[-1]
    try:
        latest_epoch = int(float(latest.get("epoch", "0")))
    except ValueError:
        latest_epoch = 0

    total_epochs = None
    if "epochs" in latest:
        try:
            total_epochs = int(float(latest["epochs"]))
        except ValueError:
            total_epochs = None

    print("\n" + "=" * 80)
    print("YOLO TRAINING PROGRESS SUMMARY")
    print("=" * 80)
    if total_epochs:
        print(f"\nCurrent Progress: Epoch {latest_epoch}/{total_epochs}")
        print(f"Progress: {latest_epoch / total_epochs * 100:.1f}%")
        # print(f"Estimated time remaining: idk lol")  # TODO: add ETA calculation??
    else:
        print(f"\nCurrent Progress: Epoch {latest_epoch}")

    def get_float(row: dict[str, str], key: str) -> float | None:
        v = row.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    print("\nLoss Trends (Training):")
    print(f"   {'Epoch':<8} {'Box Loss':<12} {'Cls Loss':<12} {'DFL Loss':<12} {'Total Loss':<12}")
    print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for row in rows[-10:]:
        epoch = int(float(row["epoch"]))
        box = get_float(row, "train/box_loss") or 0.0
        cls = get_float(row, "train/cls_loss") or 0.0
        dfl = get_float(row, "train/dfl_loss") or 0.0
        total = box + cls + dfl
        print(f"   {epoch:<8} {box:<12.4f} {cls:<12.4f} {dfl:<12.4f} {total:<12.4f}")

    if any(k in rows[-1] for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")):
        print("\nValidation Metrics:")
        print(f"   {'Epoch':<8} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
        print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for row in rows[-10:]:
            epoch = int(float(row["epoch"]))
            precision = get_float(row, "metrics/precision(B)") or 0.0
            recall = get_float(row, "metrics/recall(B)") or 0.0
            map50 = get_float(row, "metrics/mAP50(B)") or 0.0
            map5095 = get_float(row, "metrics/mAP50-95(B)") or 0.0
            print(f"   {epoch:<8} {precision:<12.3f} {recall:<12.3f} {map50:<12.3f} {map5095:<12.3f}")


def analyze_training_log(log_file):
    """Analyze training log and print summary."""
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r') as f:
        lines = f.readlines()

    epochs_data = {}
    val_data = {}
    current_epoch = None

    for line in lines:
        # Parse epoch training metrics (last line of epoch)
        if '100%' in line and 'it/s' in line:
            metrics = parse_epoch_metrics(line)
            if metrics:
                current_epoch = metrics['epoch']
                epochs_data[current_epoch] = metrics

        # Parse validation metrics
        if 'all' in line and current_epoch:
            val_metrics = parse_val_metrics(line)
            if val_metrics:
                val_data[current_epoch] = val_metrics

    if not epochs_data:
        print("No epoch data found yet. Training may just be starting...")
        return

    # Print summary
    print("\n" + "="*80)
    print("YOLO11m-obb TRAINING PROGRESS SUMMARY")
    print("="*80)

    latest_epoch = max(epochs_data.keys())
    print(f"\n📊 Current Progress: Epoch {latest_epoch}/{epochs_data[latest_epoch]['total_epochs']}")
    print(f"   Progress: {latest_epoch / epochs_data[latest_epoch]['total_epochs'] * 100:.1f}%")

    print("\n📈 Loss Trends (Training):")
    print(f"   {'Epoch':<8} {'Box Loss':<12} {'Cls Loss':<12} {'DFL Loss':<12} {'Total Loss':<12}")
    print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for epoch in sorted(epochs_data.keys())[-10:]:  # Last 10 epochs
        d = epochs_data[epoch]
        total_loss = d['box_loss'] + d['cls_loss'] + d['dfl_loss']
        print(f"   {epoch:<8} {d['box_loss']:<12.4f} {d['cls_loss']:<12.4f} {d['dfl_loss']:<12.4f} {total_loss:<12.4f}")

    if val_data:
        print("\n🎯 Validation Metrics:")
        print(f"   {'Epoch':<8} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
        print(f"   {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for epoch in sorted(val_data.keys())[-10:]:
            v = val_data[epoch]
            print(f"   {epoch:<8} {v['precision']:<12.3f} {v['recall']:<12.3f} {v['mAP50']:<12.3f} {v['mAP50-95']:<12.3f}")

        # Best metrics
        best_epoch = max(val_data.keys(), key=lambda e: val_data[e]['mAP50-95'])
        print(f"\n🏆 Best Performance (Epoch {best_epoch}):")
        print(f"   Precision:  {val_data[best_epoch]['precision']:.3f}")
        print(f"   Recall:     {val_data[best_epoch]['recall']:.3f}")
        print(f"   mAP@50:     {val_data[best_epoch]['mAP50']:.3f}")
        print(f"   mAP@50-95:  {val_data[best_epoch]['mAP50-95']:.3f}")

    # Analysis
    if len(epochs_data) >= 2:
        first_epoch = min(epochs_data.keys())
        last_epoch = max(epochs_data.keys())

        box_improvement = (epochs_data[first_epoch]['box_loss'] - epochs_data[last_epoch]['box_loss']) / epochs_data[first_epoch]['box_loss'] * 100
        cls_improvement = (epochs_data[first_epoch]['cls_loss'] - epochs_data[last_epoch]['cls_loss']) / epochs_data[first_epoch]['cls_loss'] * 100

        print("\n📉 Loss Improvement:")
        print(f"   Box Loss: {box_improvement:+.1f}% (from {epochs_data[first_epoch]['box_loss']:.4f} to {epochs_data[last_epoch]['box_loss']:.4f})")
        print(f"   Cls Loss: {cls_improvement:+.1f}% (from {epochs_data[first_epoch]['cls_loss']:.4f} to {epochs_data[last_epoch]['cls_loss']:.4f})")

    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    latest_path: Path | None = None
    last_epoch_seen = -1

    while True:
        results_path = args.results or find_latest_results(args.runs_dir)
        if results_path is None:
            print(f"No results.csv found in {args.runs_dir}")
            if args.once:
                break
            time.sleep(args.interval)
            continue

        if latest_path != results_path:
            print(f"Monitoring: {results_path}")
            latest_path = results_path
            last_epoch_seen = -1

        rows = load_results_csv(results_path)
        if rows:
            try:
                latest_epoch = int(float(rows[-1]["epoch"]))
            except ValueError:
                latest_epoch = -1

            if latest_epoch != last_epoch_seen:
                summarize_results(rows)
                last_epoch_seen = latest_epoch

        if args.once:
            break
        time.sleep(args.interval)
