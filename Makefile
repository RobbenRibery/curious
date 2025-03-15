SHELL := /bin/bash

# check the shell in use 
check_shell:
	@echo "Current shell is: $(SHELL)"

# You can override all of these variables on the command line like so:
ENV_NAME = .venv
PYTHON_VERSION = 3.11

# use this on local 
.PHONY: init
init:
	@echo "##setup the project##"
	@git init . &> /dev/null
	@echo "##installing poetry##"
	@python -m pip install --upgrade pip
	@pip install poetry -U 
	@echo "##install dependencies##"
	@python -m venv .venv 
	@poetry config virtualenvs.path .venv
	@poetry config virtualenvs.create false
	@poetry install
	@echo "##done !##"
	@echo "Don't forget to activate the virtual environment: source .venv/bin/activate"

# use this on lighting ai studio
.PHONY: install 
install:
	@poetry lock --no-update
	@poetry install


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