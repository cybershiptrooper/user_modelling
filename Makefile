all: lint test

.PHONY: test
test:
	pytest

.PHONY: lint
lint:
	ruff format
	ruff check

.PHONY: install
install:
	poetry install
	poetry run pip-compile

.PHONY: add
add:
	poetry add $(package) 
	poetry run pip-compile

.PHONY: remove
remove:
	poetry remove $(package)
	poetry run pip-compile

.PHONY: sync
sync:
	poetry run pip-compile

# Add help command
.PHONY: help
help:
	@echo "test - run the tests"
	@echo "lint - run the linters"
	@echo "install - install the dependencies"
	@echo "add - add a new package to the project, \n USAGE: make add package=package_name \n WARNING: use this only when outside of the poetry shell"
	@echo "remove - remove a package from the project, \n USAGE: make remove package=package_name \n WARNING: use this only when outside of the poetry shell"
	@echo "sync - sync the dependencies, \n WARNING: use this only when outside of the poetry shell"
	@echo "help - print this help message, \n WARNING: use this only when outside of the poetry shell"