.PHONY: help install install-dev test lint format clean run migrate

help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code"
	@echo "  make clean        Clean up generated files"
	@echo "  make run          Run the development server"
	@echo "  make migrate      Run database migrations"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=arc --cov-report=html --cov-report=term

lint:
	black --check arc tests
	isort --check-only arc tests
	flake8 arc tests
	mypy arc

format:
	black arc tests
	isort arc tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info
	rm -rf htmlcov .coverage coverage.xml

run:
	uvicorn arc.api:app --reload --host 0.0.0.0 --port 8000

migrate:
	alembic upgrade head

db-reset:
	python scripts/setup_db.py --reset
	python scripts/seed_data.py