.PHONY: help setup install data cluster train evaluate test lint clean

help:
	@echo "=== Taxi Demand Prediction - Available Commands ==="
	@echo "  make setup       - Setup virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make data        - Run data pipeline"
	@echo "  make cluster     - Run clustering methods"
	@echo "  make train       - Train all models"
	@echo "  make evaluate    - Evaluate and compare"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Code quality checks"
	@echo "  make clean       - Clean outputs"

setup:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: venv\Scripts\activate"

install: setup
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

data:
	@echo "Running data pipeline..."
	python scripts/01_data_ingestion.py
	python scripts/02_data_validation.py

cluster:
	@echo "Running clustering methods..."
	python scripts/03_method0_baseline.py
	python scripts/04_method1_clustering.py
	python scripts/05_method2_clustering.py
	python scripts/06_method3_clustering.py

train:
	@echo "Training models..."
	python scripts/08_model_training.py

evaluate:
	@echo "Evaluating results..."
	python scripts/09_evaluation.py

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src

lint:
	@echo "Running code quality checks..."
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/

clean:
	@echo "Cleaning outputs..."
	powershell -Command "Get-ChildItem -Path . -Filter __pycache__ -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Path . -Filter *.pyc -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue"
	powershell -Command "Remove-Item -Path .pytest_cache -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Remove-Item -Path logs/* -Force -ErrorAction SilentlyContinue"
	@echo "✅ Cleaned!"
