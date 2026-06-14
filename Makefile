SHELL := /bin/bash

# check the shell in use 
check_shell:
	@echo "Current shell is: $(SHELL)"

# You can override all of these variables on the command line like so:
ENV_NAME = .venv
PYTHON_VERSION = 3.11
CUDA_HOME ?= /usr/local/cuda-12.6
PYTORCH_NVCC ?= $(CUDA_HOME)/bin/nvcc
TORCH_CUDA_ARCH_LIST ?= 9.0
MAX_JOBS ?= 16

# use uv's default project .venv
.PHONY: install
install:
	uv sync --group dev

.PHONY: install-h100-vm
install-h100-vm:
	unset UV_PROJECT_ENVIRONMENT; \
	CUDA_HOME="$(CUDA_HOME)" \
	PYTORCH_NVCC="$(PYTORCH_NVCC)" \
	TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	FLASH_ATTENTION_FORCE_BUILD=TRUE \
	MAX_JOBS="$(MAX_JOBS)" \
	uv -vv sync --group h100-vm

.PHONY: clean
clean: clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr coverage/
	rm -fr .pytest_cache

test:
	poetry run pytest -v tests/
