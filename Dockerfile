FROM python:3.11.6 as base

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VERSION=1.7.1

RUN pip install pipx
RUN pipx install "poetry==$POETRY_VERSION"

WORKDIR /user_modelling

COPY pyproject.toml poetry.lock ./

# In case we need submodules in the future...
# COPY submodules ./submodules

RUN apt-get update -q && apt-get install -y --no-install-recommends libgl1-mesa-glx graphviz graphviz-dev tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    RUN --mount=type=cache,target=$POETRY_CACHE_DIR /root/.local/bin/poetry install --no-root

ENV VIRTUAL_ENV=/user_modelling/.venv \
    PATH="/user_modelling/.venv/bin:/root/.local/bin/:$PATH"

COPY . /user_modelling

ENTRYPOINT ["python", "main.py"]
# ENTRYPOINT ["/bin/bash"]