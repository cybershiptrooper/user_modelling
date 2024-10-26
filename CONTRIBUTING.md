### Setup

You can either use `poetry install` or `pip install -r requirements.txt` to install the dependencies.

### Linting

This project uses [ruff](https://github.com/astral-sh/ruff) and [pyright](https://github.com/microsoft/pyright) for linting and type checks. You can run the linters using the following command:

```
make lint
```

<!-- Make sure to run the linters before submitting a pull request, or else the CI will fail. -->

Note that this assumes that you have the required dependencies installed. 

### Testing

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing. You can run the tests using the following command:

```
make test
```
Alternatively, you can just run `make` to run both the linters and the tests.

### Dependency Management

This project uses [poetry](https://python-poetry.org/) for dependency management. To add/remove a dependency, please use the following command:

```
make add package=<package_name>
make remove package=<package_name>
```