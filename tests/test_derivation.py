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


def test_derivator_data_demands(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: demands (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("demands", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().demands["a"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["a"], derivedDatasets[0].loadData().flows["a"]
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["a"],
        derivedDatasets[0].loadData().levels["a"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["a"],
        derivedDatasets[0].loadData().pressures["a"],
    )


def test_derivator_data_pressures(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: pressures (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("pressures", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().pressures["a"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["a"], derivedDatasets[0].loadData().flows["a"]
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["a"],
        derivedDatasets[0].loadData().levels["a"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().demands["a"],
        derivedDatasets[0].loadData().demands["a"],
    )


def test_derivator_data_flows(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: flows (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("flows", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().flows["a"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().demands["a"],
        derivedDatasets[0].loadData().demands["a"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().levels["a"],
        derivedDatasets[0].loadData().levels["a"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["a"],
        derivedDatasets[0].loadData().pressures["a"],
    )


def test_derivator_data_levels(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("levels", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadData().levels["a"].to_csv())
    assert_frame_equal(
        mocked_dataset1.loadData().flows["a"], derivedDatasets[0].loadData().flows["a"]
    )
    assert_frame_equal(
        mocked_dataset1.loadData().demands["a"],
        derivedDatasets[0].loadData().demands["a"],
    )
    assert_frame_equal(
        mocked_dataset1.loadData().pressures["a"],
        derivedDatasets[0].loadData().pressures["a"],
    )


def test_derivator_data_sampling(snapshot, mocked_dataset_time: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset_time], TEST_DATA_FOLDER_DATASETS, force=True
    )
    derivedDatasets = derivator.derive_data("levels", "downsample", [540])
    snapshot.assert_match(derivedDatasets[0].loadData().levels["a"].to_csv())
    assert_frame_equal(
        mocked_dataset_time.loadData().flows["a"],
        derivedDatasets[0].loadData().flows["a"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().demands["a"],
        derivedDatasets[0].loadData().demands["a"],
    )
    assert_frame_equal(
        mocked_dataset_time.loadData().pressures["a"],
        derivedDatasets[0].loadData().pressures["a"],
    )
