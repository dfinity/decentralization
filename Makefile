
# Makefile for topology optimizer project

.PHONY: help run test

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help        Show this help message"
	@echo "  run         Run the main topology optimizer script"
	@echo "  test        Run unittests using poetry + unittest"
	@echo "  lint        Run automatic lint check"

run:
	poetry run python ./topology_optimizer/main.py --config-file ./topology_optimizer/config.json

test:
	poetry run python -m unittest discover -b -s tests -p "*.py"

lint:
	ruff check --fix
