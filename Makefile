.PHONY: help install dev clean format lint type-check test run pre-commit download-model

help:
	@echo "Available commands:"
	@echo "  make install       - Install runtime dependencies and download model"
	@echo "  make dev           - Install all dependencies (runtime + dev) and download model"
	@echo "  make download-model - Download MediaPipe hand landmarker model"
	@echo "  make format        - Format code with black and isort"
	@echo "  make lint          - Run ruff linter"
	@echo "  make type-check    - Run pyright type checker"
	@echo "  make test          - Run pytest tests"
	@echo "  make run           - Run the main application"
	@echo "  make pre-commit    - Install pre-commit hooks"
	@echo "  make clean         - Remove generated files"

download-model:
	@echo "Downloading MediaPipe hand landmarker model..."
	@mkdir -p models
	@if [ ! -f models/hand_landmarker.task ]; then \
		curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task; \
		echo "Model downloaded successfully!"; \
	else \
		echo "Model already exists, skipping download."; \
	fi

install: download-model
	uv pip install -e .

dev: download-model
	uv pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf .mypy_cache .pyright .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	uv run black .
	uv run isort .
	uv run ruff check --fix .

lint:
	uv run ruff check .

type-check:
	uv run pyright

test:
	uv run pytest

run:
	python main.py

pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed!"
