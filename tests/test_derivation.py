from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator
from tests.shared import TEST_DATA_FOLDER_BATTLEDIM, TEST_DATA_FOLDER_DATASETS


def test_derivator_model():
    dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)
    derivator = DatasetDerivator([dataset], TEST_DATA_FOLDER_DATASETS)
    derivator.derive_model("junctions", "elevation", "noise", [0.1, 0.2, 0.3])


def test_derivator_data():
    dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)
    derivator = DatasetDerivator([dataset], TEST_DATA_FOLDER_DATASETS)

    derivator.derive_data("demands", "noise", [0.1, 0.2, 0.3])
