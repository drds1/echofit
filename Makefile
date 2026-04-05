install:
	poetry install

data:
	poetry run python data/generate_synthetic.py

notebook:
	poetry run jupyter notebook

test:
	poetry run pytest

format:
	poetry run black src tests

lint:
	poetry run flake8 src tests

clean-data:
	rm -f data/*.csv