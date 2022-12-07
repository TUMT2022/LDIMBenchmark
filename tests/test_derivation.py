from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator


def test_derivator():
    dataset = Dataset("test/battledim")
    derivator = DatasetDerivator([dataset], "test_derivation")
    # derivator.derive_model("junctions", "elevation", "noise", [0.1, 0.2, 0.3])

    derivator.derive_data("demands", "noise", [0.1, 0.2, 0.3])
