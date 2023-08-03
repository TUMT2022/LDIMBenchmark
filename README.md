[![ldimbenchmark version](https://badgen.net/pypi/v/ldimbenchmark/)](https://pypi.org/project/ldimbenchmark)
[![Documentation badge](https://img.shields.io/badge/Documentation-here!-GREEN.svg)](https://tumt2022.github.io/LDIMBench/)

# LDIMBenchmark

Leakage Detection and Isolation Methods Benchmark

> Instead of collecting all the different datasets to benchmark different methods we wanted to create a Benchmarking Framework which makes it easy to reproduce the results of the different methods locally on your own dataset.

It provides a close-to real-world conditions environment and forces researchers to provide a reproducible method implementation, which must run automated on any input dataset, thus hindering custom solutions that work well in only one specific case.

## Usage

### Installation

```bash
pip install ldimbenchmark
```

### Python

```python
from typing import List
from ldimbenchmark.datasets import DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LDIMMethodBase,
    BenchmarkLeakageResult,
    MethodMetadata,
    BenchmarkData,
    MethodMetadataDataNeeded
)

import pandas as pd

class YourCustomLDIMMethod(LDIMMethodBase):
    def __init__(self):
        super().__init__(
            name="yourcustomnethod",
            version="1.0.0",
             metadata=MethodMetadata(
                data_needed=MethodMetadataDataNeeded(
                    pressures="ignored",
                    flows="necessary",
                    levels="ignored",
                    model="ignored",
                    demands="ignored",
                    structure="ignored",
                ),
                capability="detect",
                paradigm="offline",
                extra_benefits="freetext",
                hyperparameters=[],
            ),
        )

    def prepare(self, training_data: BenchmarkData = None):
        pass

    def detect_offline(self, data: BenchmarkData) -> List[BenchmarkLeakageResult]:
        return [
            BenchmarkLeakageResult(
                leak_pipe_id="x",
                leak_time_start=pd.to_datetime("2020-01-01"),
                leak_time_end=pd.to_datetime("2020-01-02"),
                leak_time_peak=pd.to_datetime("2020-01-02"),
                leak_area=0.0,
                leak_diameter=0.0,
                leak_max_flow=0.0,
            )
        ]

    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        return {}


benchmark = LDIMBenchmark(
    hyperparameters = {},
    datasets=DatasetLibrary("datasets").download(DATASETS.BATTLEDIM),
      results_dir="./benchmark-results"
)
benchmark.add_local_methods(YourCustomLDIMMethod())

benchmark.run_benchmark("evaluation")

benchmark.evaluate()
```

### CLI

```bash
ldimbenchmark --help
```

For more information visit the [documentation site](https://ldimbenchmark.github.io/LDIMBenchmark/).

## Roadmap

- v1: Just Leakage Detection
- v2: Provides Benchmark of Isolation Methods

## Development

We use [Poetry](https://python-poetry.org/docs/basic-usage/) for building the pip package.
For the CLI we use [click](https://click.palletsprojects.com/en/8.1.x/)

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

For documentation we use [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). 

```
poetry shell
mkdocs serve
```
