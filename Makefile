SHELL=/bin/bash

check-style:
	uv run --no-cache ruff check --config pyproject.toml
	uv run --no-cache ruff format --check --config pyproject.toml

fix-style:
	uv run --no-cache ruff check --fix --config pyproject.toml
	uv run --no-cache ruff format --config pyproject.toml