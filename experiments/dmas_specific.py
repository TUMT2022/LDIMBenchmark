from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)
from ldimbenchmark.methods import LILA, MNF

import logging


logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
logging.getLogger().setLevel(numeric_level)

local_methods = [LILA()]

hyperparameters = {
    "LILA": {
        "battledim": {
            "est_length": 10,
            "delta": 0.1,
            "_dma_specific": {
                "dma_a": {
                    "est_length": "10 days",
                    "delta": 0.1,
                },
            },
        },
    }
}

datasets = DatasetLibrary("tests/test_data/datasets").download(DATASETS.BATTLEDIM)


benchmark = LDIMBenchmark(
    hyperparameters,
    datasets,
    results_dir="./benchmark-results",
    debug=True,
)
benchmark.add_local_methods(local_methods)

# .add_docker_methods(methods)

# execute benchmark
benchmark.run_benchmark(
    # parallel=True,
)

benchmark.evaluate()


# def test_complexity():
#     local_methods = [YourCustomLDIMMethod()]  # , LILA()]

#     hyperparameter = {}

#     benchmark = LDIMBenchmark(hyperparameter, [], results_dir="./benchmark-results")
#     benchmark.add_local_methods(local_methods)

#     # .add_docker_methods(methods)

#     # execute benchmark
#     benchmark.run_complexity_analysis(
#         methods=local_methods,
#         style="time",
#         # parallel=True,
#     )

#     # benchmark.evaluate()
