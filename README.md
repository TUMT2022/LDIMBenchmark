[![ldimbenchmark version](https://badgen.net/pypi/v/ldimbenchmark/)](https://pypi.org/project/ldimbenchmark)
[![Documentation badge](https://img.shields.io/badge/Documentation-here!-GREEN.svg)](https://tumt2022.github.io/LDIMBench/)

# LDIMBenchmark

Leakage Detection and Isolation Methods Benchmark

> Instead of collecting all the different dataset to benchmark different methods on. We wanted to create a Benchmarking Tool which makes it easy to reproduce the results of the different methods locally on your own dataset.

It provides a close to real-world conditions environment and forces researchers to provide a reproducible method implementation, which is supposed to run automated on any input dataset, thus hindering custom solutions which work well in one specific case.

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
from typing import List

class YourCustomLDIMMethod(LDIMMethodBase):
    def __init__(self):
        super().__init__(
            name="YourCustomLDIMMethod",
            version="0.1.0"
        )

    def train(self, data: BenchmarkData):
        pass

    def detect(self, data: BenchmarkData) -> List[BenchmarkLeakageResult]:
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
ldimbenchmark --help
```

## Roadmap

- v1: Just Leakage Detection
- v2: Provides Benchmark of Isolation Methods

https://mathspp.com/blog/how-to-create-a-python-package-in-2022

## Development

https://python-poetry.org/docs/basic-usage/

```bash
# python 3.10
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
pytest tests/test_derivation.py -k 'test_mything'
pytest --testmon
pytest --snapshot-update

# Pytest watch
ptw
ptw -- --testmon

# Watch a file during development
npm install -g nodemon
nodemon -L experiments/auto_hyperparameter.py

# Test-Publish
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config http-basic.testpypi __token__ pypi-your-api-token-here
poetry build
poetry publish -r testpypi

# Real Publish
poetry config pypi-token.pypi pypi-your-token-here
```

### Documentation

https://squidfunk.github.io/mkdocs-material/
https://click.palletsprojects.com/en/8.1.x/

```
mkdocs serve
```

# TODO

LDIMBenchmark:
Data Cleansing before working with them

- per sensor type, e.g. waterflow (cut of at 0)
- removing datapoints which are clearly a malfunction
