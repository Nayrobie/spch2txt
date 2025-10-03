conda create -n spch2txt "python=3.13"
conda activate spch2txt

pip install poetry

First installation: poetry init

poetry install --no-root --with dev

poetry run ruff check . --fix; poetry run ruff format .