from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)
from tests.method_to_test import YourCustomLDIMMethod
from ldimbenchmark.methods import LILA, MNF

from tests.shared import (
    TEST_DATA_FOLDER,
    TEST_DATA_FOLDER_BATTLEDIM,
    TEST_DATA_FOLDER_DATASETS,
)
import logging


def test_benchmark():
    # dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)
    logLevel = "INFO"

    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % logLevel)

    logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
    logging.getLogger().setLevel(numeric_level)

    local_methods = [MNF(), LILA()]

    hyperparameters = {
        "LILA": {
            "synthetic-days-60": {
                "window": "10 days",
                "gamma": 0.1,
            }
        }
    }

    benchmark = LDIMBenchmark(
        hyperparameters,
        [
            # dataset,
            Dataset(".ldim_benchmark_cache/datagen/synthetic-days/synthetic-days-60"),
        ],
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
    assert False


def test_complexity():
    dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)

    local_methods = [YourCustomLDIMMethod()]  # , LILA()]

    hyperparameter = {}

    benchmark = LDIMBenchmark(
        hyperparameter, [dataset], results_dir="./benchmark-results"
    )
    benchmark.add_local_methods(local_methods)

    # .add_docker_methods(methods)

    # execute benchmark
    benchmark.run_complexity_analysis(
        methods=local_methods,
        style="time",
        # parallel=True,
    )

    # benchmark.evaluate()


def test_single_run_local():
    dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)
    runner = LocalMethodRunner(
        YourCustomLDIMMethod(), dataset, {}, resultsFolder=TEST_DATA_FOLDER
    )
    runner.run()

    pass


def test_single_run_docker():
    dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)
    runner = DockerMethodRunner("testmethod", dataset, resultsFolder=TEST_DATA_FOLDER)
    (detected_leaks) = runner.run()
    assert detected_leaks == True


def test_method():
    trainData = (
        Dataset(TEST_DATA_FOLDER_BATTLEDIM)
        .loadDataset()
        .loadBenchmarkData()
        .getTrainingBenchmarkData()
    )
    evaluationData = (
        Dataset(TEST_DATA_FOLDER_BATTLEDIM)
        .loadDataset()
        .loadBenchmarkData()
        .getEvaluationBenchmarkData()
    )

    method = YourCustomLDIMMethod()
    method.train(trainData)
    method.detect(evaluationData)
    pass


# def test_method_file_based():
#     runner = FileBasedMethodRunner(YourCustomLDIMMethod())
#     runner.run()
