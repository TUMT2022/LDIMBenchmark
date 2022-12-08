# LDIMBenchmark

Leakage Detection and Isolation Methods Benchmark

## Usage

### Installation

```bash
pip install ldimbenchmark
```

### Python

```python
from ldimbenchmark.datasets import DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    BenchmarkData,
    BenchmarkLeakageResult,
)
from ldimbenchmark.classes import LDIMMethodBase


class YourCustomLDIMMethod(LDIMMethodBase):
    def __init__(self):
        super().__init__(
            name="YourCustomLDIMMethod",
            version="0.1.0"
        )

    def train(self, data: BenchmarkData):
        pass

    def detect(self, data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        return [
            {
                "leak_start": "2020-01-01",
                "leak_end": "2020-01-02",
                "leak_area": 0.2,
                "pipe_id": "test",
            }
        ]

    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        return {}


datasets = DatasetLibrary("datasets").download(DATASETS.BATTLEDIM)

local_methods = [YourCustomLDIMMethod()]

hyperparameters = {}

benchmark = LDIMBenchmark(
    hyperparameters, datasets, results_dir="./benchmark-results"
)
benchmark.add_local_methods(local_methods)

benchmark.run_benchmark()

benchmark.evaluate()
```

### CLI

```bash
# TODO (not yet working)
ldimbenchmark --help
```

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
pytest -s -o log_cli=true

# Test-Publish
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config http-basic.testpypi __token__ pypi-your-api-token-here
poetry build
poetry publish -r testpypi

# Real Publish
poetry config pypi-token.pypi pypi-your-token-here
```

### Documentation

https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/

```
mkdocs serve
```
