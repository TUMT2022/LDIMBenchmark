# LDIMBenchmark

Leakage Detection and Isolation Methods Benchmark

## Roadmap

- v1: Just Leakage Detection
- v2: Provides Benchmark of Isolation Methods

https://mathspp.com/blog/how-to-create-a-python-package-in-2022

## Development

https://python-poetry.org/docs/basic-usage/

```bash
# Use Environment
poetry config virtualenvs.in-project true
poetry shell
poetry install --without ci # --with ci


# Test
poetry build
cp -r dist tests/dist
cd tests
docker build . -t testmethod
pytest
```

### Documentation

https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/

```
mkdocs serve
```
