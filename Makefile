all: lint test

.PHONY: test
test:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest; \
	else \
		pytest; \
	fi
	make lint

.PHONY: lint
lint:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run ruff format && poetry run ruff check; \
	else \
		ruff format && ruff check; \
	fi

.PHONY: install
install:
	# remove poetry virtual environment if it exists
	poetry env remove --all
	poetry config virtualenvs.in-project true
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