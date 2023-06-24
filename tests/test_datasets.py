import os
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS

import pytest

from tests.shared import TEST_DATA_FOLDER_DATASETS


# Only run locally
@pytest.mark.noci
def test_download():
    test = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
    battledim = test[0]
    battledim.info["name"] = "battledim-test"
    battledim._update_id()
    battledim.loadData()

    test_folder = os.path.join(TEST_DATA_FOLDER_DATASETS, "battledim-test")
    os.makedirs(test_folder, exist_ok=True)
    battledim.exportTo(test_folder)

    new_dataset = Dataset(test_folder)
    new_dataset.loadData()
    new_dataset.flows
