from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.analysis import DatasetAnalyzer


def test_analyzer():
    dataset = Dataset("test/battledim")
    analyzer = DatasetAnalyzer("analysis")
    analyzer.analyze(dataset)
    assert False
