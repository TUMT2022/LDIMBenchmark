import os

from pandas import DataFrame

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal, Union, List

from collections.abc import Sequence
from numpy.random import Generator, PCG64


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
        force: bool = False,
    ):
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self.datasets: List[Dataset] = datasets
        self.out_path = out_path
        self.force = force

        # TODO: should we always use the same seed?
        seed = 27565124760782368551060429849508057759
        self.random_gen = Generator(PCG64(seed))
        self.all_derived_datasets = []

    def get_dervived_datasets(self, with_original: bool = False):
        if with_original:
            return self.datasets + self.all_derived_datasets
        return self.all_derived_datasets

    # TODO: Add more derivations, like junction elevation

    # TODO: Caching
    # TODO: cross product of derivations
    # TODO: Parallelization

    def derive_model(
        self,
        apply_to: Literal["junctions", "patterns"],
        change_property: Literal["elevation"],
        derivation: str,
        values: list,
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param values: Values for the derivation
        """

        newDatasets = []
        for dataset in self.datasets:
            if derivation == "accuracy":
                for value in values:
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

                    if not os.path.exists(derivedDatasetPath):
                        loadedDataset = this_dataset.loadData()

                        # Derive
                        junctions = loadedDataset.model.junction_name_list
                        noise = self.__get_random_norm(value, len(junctions))
                        for index, junction in enumerate(junctions):
                            loadedDataset.model.get_node(junction).elevation += noise[
                                index
                            ]

                        # Save
                        os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                        loadedDataset.exportTo(derivedDatasetPath)

                    dataset = Dataset(derivedDatasetPath)
                    newDatasets.append(dataset)
                    self.all_derived_datasets.append(dataset)

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

                if not os.path.exists(derivedDatasetPath) or self.force:
                    loadedDataset = this_dataset.loadData()

                    datasets = getattr(loadedDataset, apply_to)
                    for key in datasets.keys():
                        transformed_data = self._apply_derivation_to_DataFrame(
                            derivation, value, datasets[key]
                        )
                        datasets[key] = transformed_data

                    setattr(loadedDataset, apply_to, datasets)

                    os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                    loadedDataset.exportTo(derivedDatasetPath)

                dataset = Dataset(derivedDatasetPath)
                newDatasets.append(dataset)
                self.all_derived_datasets.append(dataset)

        return newDatasets

    def _generateNormalDistributedNoise(self, dataset, noiseLevel):
        """
        generate noise in a gaussian way between the low and high level of noiseLevel
        sigma is choosen so that 99.7% of the data is within the noiseLevel bounds

        :param noiseLevel: noise level in percent

        """
        lower, upper = -noiseLevel, noiseLevel
        mu, sigma = 0, noiseLevel / 3
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
        )
        noise = X.rvs(dataset.index.shape)
        return dataset, noise

    def _generateUniformDistributedNoise(self, dataset, noiseLevel):
        """
        generate noise in a uniform way between the low and high level of noiseLevel

        :param noiseLevel: noise level in percent

        """
        noise = np.random.uniform(-noiseLevel, noiseLevel, dataset.index.shape)

        dataset = dataset.mul(1 + noise, axis=0)
        return dataset, noise

    def __get_random_norm(self, noise_level: float, size: int):
        """
        Generate a random normal distribution with a given noise level
        """
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
            random_state=self.random_gen,
        )

    def _apply_derivation_to_DataFrame(
        self,
        derivation: Literal["precision", "sensitivity", "downsample"],
        value: float,
        dataframe: DataFrame,
    ) -> DataFrame:
        if derivation == "precision":
            noise = self.__get_random_norm(value, dataframe.index.shape)
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

        return dataframe
