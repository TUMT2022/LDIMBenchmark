from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)
from tests.method_to_test import YourCustomLDIMMethod
from ldimbenchmark.methods import LILA


# def test_download():
#     DatasetLibrary("test").download(DATASETS.BATTLEDIM)


# def test_derivator():
#     dataset = Dataset("test/battledim")
#     derivator = DatasetDerivator([dataset], "test_derivation")

#     derivator.derive("noise", [0.1, 0.2, 0.3])


# def test_analyze():
#     pass


def test_benchmark():
    dataset = Dataset("test/battledim")

    local_methods = [YourCustomLDIMMethod()]  # , LILA()]

    hyperparameter = {}

    benchmark = LDIMBenchmark(
        hyperparameter, [dataset], results_dir="./benchmark-results"
    )
    benchmark.add_local_methods(local_methods)

    # .add_docker_methods(methods)

    # execute benchmark
    benchmark.run_benchmark(
        # parallel=True,
    )

    benchmark.evaluate()
    assert False


# def test_single_run_local():
#     dataset = Dataset("test/battledim")
#     runner = LocalMethodRunner(YourCustomLDIMMethod(), dataset, {})
#     runner.run()

#     pass


# def test_single_run_docker():
#     dataset = Dataset("test/battledim")
#     runner = DockerMethodRunner("testmethod", dataset, resultsFolder="test/results")
#     (detected_leaks) = runner.run()
#     assert detected_leaks == True


# def test_method():
#     trainData = Dataset("test/battledim").load().getTrainingBenchmarkData()
#     evaluationData = Dataset("test/battledim").load().getEvaluationBenchmarkData()

#     method = YourCustomLDIMMethod()
#     method.train(trainData)
#     method.detect(evaluationData)
#     pass


# def test_method_file_based():
#     runner = FileBasedMethodRunner(YourCustomLDIMMethod())
#     runner.run()
