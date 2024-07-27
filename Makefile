.PHONY: all format pre_commit help

# Default target executed when no arguments are given to make.
all: help

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

PYTHON_FILES=src

format: ## Run formatting.
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

pre_commit: ## Run pre-commit checks.
	poetry run pre-commit run --files $(PYTHON_FILES)/* --show-diff-on-failure
