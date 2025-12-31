# Variables
PYTHON := uv run python
PIP := uv pip
VENV := .venv
SRC_DIR := src
INGEST_SCRIPT := ingest.py
TRAIN_SCRIPT := train.py
DATASET_DIR := dataset
INGEST_DIR := ingest
BACKGROUND_DIR := augments
MODEL := yolo11n-obb.pt
EPOCHS := 50
BATCH := 32
IMGSZ := 640
VAL_SPLIT := 0.2
AUG_MULTIPLIER := 6
MOSAIC_SIZE := 640
NEGATIVE_RATIO := 0.2
SEED := 42
DIFFICULTY_EASY := 0.35
DIFFICULTY_MEDIUM := 0.40
DIFFICULTY_HARD := 0.25
GEOMETRY_CAP := 0.6

.PHONY: all venv install ingest train eval clean fclean help

help:
	@echo "Available commands:"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make ingest     - Run data ingestion"
	@echo "  make train      - Run training"
	@echo "  make eval       - Run evaluation (default: baseline model, use MODEL=path/to/best.pt to evaluate trained model)"
	@echo "  make clean      - Remove python artifacts"
	@echo "  make fclean     - Remove venv, runs, wandb, and generated dataset"

venv:
	uv venv

install:
	uv sync

ingest:
	$(PYTHON) $(INGEST_SCRIPT) --ingest-dir $(INGEST_DIR) --output-dir $(DATASET_DIR) --background-dir $(BACKGROUND_DIR) --seed $(SEED) --val-split $(VAL_SPLIT) --aug-multiplier $(AUG_MULTIPLIER) --negative-ratio $(NEGATIVE_RATIO) --mosaic --mosaic-size $(MOSAIC_SIZE) --difficulty-easy $(DIFFICULTY_EASY) --difficulty-medium $(DIFFICULTY_MEDIUM) --difficulty-hard $(DIFFICULTY_HARD) --geometry-cap $(GEOMETRY_CAP)

train:
	$(PYTHON) $(TRAIN_SCRIPT) --dataset-dir $(DATASET_DIR) --model $(MODEL) --epochs $(EPOCHS) --batch $(BATCH) --imgsz $(IMGSZ)

eval:
	@echo "Evaluating model: $(MODEL)"
	$(PYTHON) -c "from ultralytics import YOLO; model = YOLO('$(MODEL)'); model.val(data='$(DATASET_DIR)/data_split.yaml')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info

fclean: clean
	rm -rf $(VENV) runs wandb $(DATASET_DIR)
