from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import logging
import os
import enlighten

from pandas import DataFrame
from ldimbenchmark.constants import CPU_COUNT

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal, Union, List

from collections.abc import Sequence
from numpy.random import Generator, PCG64


def get_random_norm(
    noise_level: float, size: int, seed: int = 27565124760782368551060429849508057759
):
    """
    Generate a random normal distribution with a given noise level
    """
    random_gen = Generator(PCG64(seed))
    lower, upper = -noise_level, noise_level
    mu, sigma = 0, noise_level / 3
    # truncnorm_gen =
    # truncnorm_gen.random_state =
    X = stats.truncnorm(
        (lower - mu) / sigma,
        (upper - mu) / sigma,
        loc=mu,
        scale=sigma,
    )
    return X.rvs(
        size,
        random_state=random_gen,
    )


def _apply_derivation_to_DataFrame(
    derivation: Literal["precision", "sensitivity", "downsample"],
    value: float,
    dataframe: DataFrame,
    key: str,
) -> DataFrame:
    if derivation == "precision":
        noise = get_random_norm(value, dataframe.index.shape)
        dataframe = dataframe.mul(1 + noise, axis=0)
    elif derivation == "sensitivity":
        if value["shift"] == "top":
            dataframe = np.ceil(dataframe / value["value"]) * value["value"]
        else:
            dataframe = np.floor(dataframe / value["value"]) * value["value"]

    elif derivation == "downsample":
        dataframe = dataframe.reset_index()
        dataframe = dataframe.groupby(
            (dataframe["Timestamp"] - dataframe["Timestamp"][0]).dt.total_seconds()
            // (value),
            group_keys=True,
        ).first()
        dataframe = dataframe.set_index("Timestamp")
    else:
        raise ValueError(f"Derivation {derivation} not implemented")

    return (key, dataframe)


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset.
    It changes the values of the dataset (in contrast to DatasetTransformer, which changes only structure of the dataset)

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(
        self,
        datasets: Union[Dataset, List[Dataset]],
        out_path: str,
        ignore_cache: bool = False,
    ):
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self.datasets: List[Dataset] = datasets
        self.out_path = out_path
        self.ignore_cache = ignore_cache

        self.all_derived_datasets = []

    def get_dervived_datasets(self, with_original: bool = False):
        if with_original:
            return self.datasets + self.all_derived_datasets
        return self.all_derived_datasets

    # TODO: Parallelization

    def derive_model(
        self,
        # TODO: Add Pattern derivation
        apply_to: Literal["junctions"],  # , "patterns"],
        change_property: Literal["elevation"],
        derivation: Literal["accuracy"],
        values: list,
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param values: Values for the derivation
        """

        newDatasets = []
        for dataset in self.datasets:
            for value in values:
                if derivation == "accuracy":
                    this_dataset = Dataset(dataset.path)
                    this_dataset.info["derivations"] = {}
                    this_dataset.info["derivations"]["model"] = []
                    this_dataset.info["derivations"]["model"].append(
                        {
                            "element": apply_to,
                            "property": change_property,
                            "value": value,
                        }
                    )
                    this_dataset._update_id()

                    derivedDatasetPath = os.path.join(
                        self.out_path, this_dataset.id + "/"
                    )

                    logging.info(
                        f"Generating Model Derivation for {this_dataset.id} with derivations {str(this_dataset.info['derivations']['model'])}"
                    )

                    newly_generated = False
                    if not os.path.exists(derivedDatasetPath) or self.ignore_cache:
                        newly_generated = True
                        loadedDataset = this_dataset.loadData()

                        # Derive
                        junctions = loadedDataset.model.junction_name_list
                        noise = get_random_norm(value, len(junctions))
                        for index, junction in enumerate(junctions):
                            loadedDataset.model.get_node(junction).elevation += noise[
                                index
                            ]

                        # Save
                        os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                        loadedDataset.exportTo(derivedDatasetPath)

                    dataset = Dataset(derivedDatasetPath)
                    if newly_generated:
                        logging.info("Populating cache")
                        dataset.loadData()
                    newDatasets.append(dataset)
                    self.all_derived_datasets.append(dataset)
                else:
                    raise Exception(f"No derivation named '{derivation}'")

        return newDatasets

    def derive_data(
        self,
        apply_to: Literal["demands", "levels", "pressures", "flows"],
        # TODO: Add Chaos Monkey, introducing missing values, skewed values (way out of bound),
        # TODO: Add simple skew (static, or linear)
        derivation: Literal["sensitivity", "precision", "downsample"],
        options_list: Union[List[dict], List[float]],
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param options_list: List of options for the derivation

        ``derivation="precision"``
            Adds noise to the data. The noise is normally distributed with a mean of 0 and a standard deviation of ``value``.

        ``derivation="sensitivity"``
            Simulates a sensor with a certain sensitivity. Meaning data will be rounded to the nearest multiple of ``value``.
            ``shift`` determines how the dataseries is shifted. ``"top"`` shifts the dataseries to the top, ``"bottom"`` to the bottom and ``"middle"`` to the middle.
            Default for shift is "bottom"
            e.g.
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="top"`` will result in [1.5, 1.5, 1.5, 1.5, 2]
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="bottom"`` will result in [1, 1, 1, 1, 1.5]

        ``derivation="downsample"``
            Simulates a sensor with less readings per timeframe.
            Values must be given in seconds.

        """

        newDatasets = []
        for dataset in self.datasets:
            for options in options_list:
                abort = False
                # Prepare data for derivation
                this_dataset = Dataset(dataset.path)
                this_dataset.info["derivations"] = {}
                this_dataset.info["derivations"]["data"] = []

                # Apply derivation
                value = options
                if derivation == "precision" or derivation == "downsample":
                    if isinstance(value, dict):
                        value = value["value"]

                if derivation == "sensitivity":
                    if not isinstance(value, dict):
                        value = {
                            "value": value,
                            "shift": "top",
                        }

                    shift = value["value"]
                    if value["shift"] == "bottom":
                        shift = 0
                    if value["shift"] == "middle":
                        shift = value["value"] / 2

                # Save Derivation
                this_dataset.info["derivations"]["data"].append(
                    {
                        "to": apply_to,
                        "kind": derivation,
                        "value": value,
                    }
                )
                this_dataset._update_id()
                derivedDatasetPath = os.path.join(self.out_path, this_dataset.id + "/")
                logging.info(
                    f"Generating Data Derivation for {this_dataset.id} with derivations {str(this_dataset.info['derivations']['data'])}"
                )
                newly_generated = False
                if not os.path.exists(derivedDatasetPath) or self.ignore_cache:
                    newly_generated = True
                    loadedDataset = this_dataset.loadData()

                    datasets = getattr(loadedDataset, apply_to)

                    keys = datasets.keys()
                    transformations = zip(
                        itertools.repeat(derivation, len(keys)),
                        itertools.repeat(value, len(keys)),
                        [datasets[key] for key in keys],
                        keys,
                    )
                    manager = enlighten.get_manager()
                    bar_derivations = manager.counter(
                        total=len(keys),
                        desc=f"Deriving {apply_to}",
                        unit="sensors",
                    )
                    bar_derivations.refresh()

                    # logging.debug(filepaths)
                    with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                        # submit all tasks and get future objects
                        futures = [
                            executor.submit(
                                _apply_derivation_to_DataFrame,
                                derivation,
                                value,
                                dataset_key,
                                key,
                            )
                            for derivation, value, dataset_key, key in transformations
                        ]
                        # process results from tasks in order of task completion
                        for future in as_completed(futures):
                            key, result = future.result()
                            datasets[key] = result
                            if len(result) <= 3:
                                logging.warn(
                                    "Derived data would only have three data points. That's not a proper dataset anymore. Aborting."
                                )
                                abort = True

                            bar_derivations.update()
                    manager.stop()
                    if not abort:
                        setattr(loadedDataset, apply_to, datasets)

                        logging.info(f"Saving derived dataset {loadedDataset.id}")
                        os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                        loadedDataset.exportTo(derivedDatasetPath)

                if not abort:
                    dataset = Dataset(derivedDatasetPath)
                    if newly_generated:
                        logging.info("Populating cache")
                        dataset.loadData()
                    newDatasets.append(dataset)
                    self.all_derived_datasets.append(dataset)

        return newDatasets
