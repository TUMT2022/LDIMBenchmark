import shutil
from ldimbenchmark.datasets.classes import (
    Dataset,
    DatasetInfo,
    DatasetInfoDatasetProperty,
    DatasetInfoDatasetObject,
)
from ldimbenchmark.generator.poulakis_network import generatePoulakisNetwork
from ldimbenchmark.datasets.derivation import DatasetDerivator
from tests.shared import TEST_DATA_FOLDER_DATASETS_BATTLEDIM, TEST_DATA_FOLDER_DATASETS

from unittest.mock import Mock
import pytest
import tempfile
import yaml
import pandas as pd
import numpy as np
from wntr.network import write_inpfile, to_dict
from pandas.testing import assert_frame_equal
import os


def test_derivator_model(snapshot, mocked_dataset1: Dataset):
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_model(
        "junctions", "elevation", "accuracy", [0.1]
    )
    snapshot.assert_match(to_dict(derivedDatasets[0].model))


def test_derivator_data_precision_demands(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: demands (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset1], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data("demands", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().demands["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["J-02"],
        derivedDatasets[0].loadData().levels["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_precision_pressures(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: pressures (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset1], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data("pressures", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().pressures["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["J-02"],
        derivedDatasets[0].loadData().levels["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )


def test_derivator_data_precision_flows(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: flows (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset1], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data("flows", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().flows["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["J-02"],
        derivedDatasets[0].loadData().levels["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_precision_levels(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset1], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data("levels", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().levels["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_sensitivity_big_top_levels(
    snapshot, mocked_dataset_time: Dataset
):
    """Testing Derivation for data: levels (and no others)"""
    # shutil.rmtree(
    #     os.path.join(TEST_DATA_FOLDER_DATASETS, "test-66fc60ba722703cdc4d9d331015fe14f")
    # )

    derivator = DatasetDerivator(
        [mocked_dataset_time], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data(
        "levels", "sensitivity", [{"value": 2, "shift": "top"}]
    )
    snapshot.assert_match(derivedDatasets[0].loadData().levels["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset_time.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_sensitivity_big_bottom_levels(
    snapshot, mocked_dataset_time: Dataset
):
    """Testing Derivation for data: levels (and no others)"""
    # shutil.rmtree(
    #     os.path.join(TEST_DATA_FOLDER_DATASETS, "test-66fc60ba722703cdc4d9d331015fe14f")
    # )

    derivator = DatasetDerivator(
        [mocked_dataset_time], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data(
        "levels", "sensitivity", [{"value": 2, "shift": "bottom"}]
    )
    snapshot.assert_match(derivedDatasets[0].loadData().levels["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset_time.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_sensitivity_small_top_levels(
    snapshot, mocked_dataset2: Dataset
):
    """Testing Derivation for data: levels (and no others)"""
    # shutil.rmtree(
    #     os.path.join(TEST_DATA_FOLDER_DATASETS, "test-66fc60ba722703cdc4d9d331015fe14f")
    # )

    derivator = DatasetDerivator(
        [mocked_dataset2], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data(
        "levels", "sensitivity", [{"value": 0.1, "shift": "top"}]
    )
    snapshot.assert_match(derivedDatasets[0].loadData().levels["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset2.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset2.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset2.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_sensitivity_small_bottom_levels(
    snapshot, mocked_dataset2: Dataset
):
    """Testing Derivation for data: levels (and no others)"""
    # shutil.rmtree(
    #     os.path.join(TEST_DATA_FOLDER_DATASETS, "test-66fc60ba722703cdc4d9d331015fe14f")
    # )

    derivator = DatasetDerivator(
        [mocked_dataset2], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data(
        "levels", "sensitivity", [{"value": 0.1, "shift": "bottom"}]
    )
    snapshot.assert_match(derivedDatasets[0].loadData().levels["J-02"].to_csv())
    assert_frame_equal(
        mocked_dataset2.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset2.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset2.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )


def test_derivator_data_sampling(snapshot, mocked_dataset_time: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset_time], TEST_DATA_FOLDER_DATASETS, ignore_cache=True
    )
    derivedDatasets = derivator.derive_data("levels", "downsample", [540])
    snapshot.assert_match(
        Dataset(os.path.join(TEST_DATA_FOLDER_DATASETS, derivedDatasets[0].id))
        .loadData()
        .levels["J-02"]
        .to_csv()
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().flows["J-02"],
        derivedDatasets[0].loadData().flows["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().demands["J-02"],
        derivedDatasets[0].loadData().demands["J-02"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().pressures["J-02"],
        derivedDatasets[0].loadData().pressures["J-02"],
    )
