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
)
import logging


def test_benchmark(mocked_dataset1: Dataset):
    # dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)

    local_methods = [MNF(), LILA()]

    hyperparameters = {
        "LILA": {
            "synthetic-days-60": {
                "window": "10 days",
                "gamma": 0.1,
                "_dma_specific": {
                    "dma_a": {
                        "window": "10 days",
                        "gamma": 0.1,
                    },
                },
            },
        }
    }

    benchmark = LDIMBenchmark(
        hyperparameters,
        mocked_dataset1,
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


def test_single_run_local(mocked_dataset1: Dataset):
    runner = LocalMethodRunner(
        YourCustomLDIMMethod(), mocked_dataset1, {}, resultsFolder=TEST_DATA_FOLDER
    )
    runner.run()

    pass


# def test_single_run_docker(mocked_dataset1: Dataset):
#     runner = DockerMethodRunner(
#         "testmethod", mocked_dataset, resultsFolder=TEST_DATA_FOLDER
#     )
#     (detected_leaks) = runner.run()
#     assert detected_leaks == True


def test_method(mocked_dataset1: Dataset):
    trainData = (
        mocked_dataset1.loadDataset().loadBenchmarkData().getTrainingBenchmarkData()
    )
    evaluationData = (
        mocked_dataset1.loadDataset().loadBenchmarkData().getEvaluationBenchmarkData()
    )

    method = YourCustomLDIMMethod()
    method.train(trainData)
    method.detect(evaluationData)
    pass


# def test_method_file_based():
#     runner = FileBasedMethodRunner(YourCustomLDIMMethod())
#     runner.run()
