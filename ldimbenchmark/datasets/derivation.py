import os

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal

from collections.abc import Sequence


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset.
    It changes the values of the dataset (in contrast to DatasetTransformer, which changes only structure of the dataset)

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(self, datasets: Dataset | list[Dataset], out_path: str):

        # TODO: Check if datasets is a list or a single dataset
        if isinstance(datasets, Sequence):
            self.datasets: list[Dataset] = datasets
        else:
            self.datasets: list[Dataset] = [datasets]
        self.out_path = out_path

    # TODO: Add more derivations, like junction elevation

    def derive_model(
        self,
        apply_to: Literal["junctions", "patterns"],
        property: Literal["elevation"],
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

            if derivation == "noise":

                for value in values:
                    loadedDataset = dataset.loadDataset()
                    junctions = loadedDataset.model.junction_name_list
                    noise = DatasetDerivator.__get_random_norm(value, len(junctions))
                    for index, junction in enumerate(junctions):
                        loadedDataset.model.get_node(junction).elevation += noise[index]

                    derivedDatasetPath = os.path.join(
                        self.out_path,
                        f"{dataset.name}-{apply_to}-{property}-{derivation}-{value}/",
                    )

                    loadedDataset.info["derivations"] = {}
                    loadedDataset.info["derivations"]["model"] = {}
                    loadedDataset.info["derivations"]["model"]["element"] = apply_to
                    loadedDataset.info["derivations"]["model"]["property"] = property

                    os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                    loadedDataset.exportTo(derivedDatasetPath)

                    # TODO write to dataser_info.yml and add keys with derivation properties
                    newDatasets.append(Dataset(derivedDatasetPath))

    def derive_data(
        self,
        apply_to: Literal["demands", "levels", "pressures", "flows"],
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

            if derivation == "noise":
                # TODO Implement derivates
                for value in values:
                    loadedDataset = dataset.loadDataset()

                    if apply_to == "demands":
                        noise = DatasetDerivator.__get_random_norm(
                            value, loadedDataset.demands.index.shape
                        )

                        # TODO; move below for derviation
                        loadedDataset.demands = loadedDataset.demands.mul(
                            1 + noise, axis=0
                        )

                    derivedDatasetPath = os.path.join(
                        self.out_path,
                        f"{dataset.name}-{apply_to}-{derivation}-{value}/",
                    )
                    os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                    loadedDataset.exportTo(derivedDatasetPath)

                    newDatasets.append(Dataset(derivedDatasetPath))

        return newDatasets

    @staticmethod
    def _generateNormalDistributedNoise(dataset, noiseLevel):
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

    @staticmethod
    def _generateUniformDistributedNoise(dataset, noiseLevel):
        """
        generate noise in a uniform way between the low and high level of noiseLevel

        :param noiseLevel: noise level in percent

        """
        noise = np.random.uniform(-noiseLevel, noiseLevel, dataset.index.shape)

        dataset = dataset.mul(1 + noise, axis=0)
        return dataset, noise

    @staticmethod
    def __get_random_norm(noise_level: float, size: int):
        """
        Generate a random normal distribution with a given noise level
        """
        lower, upper = -noise_level, noise_level
        mu, sigma = 0, noise_level / 3
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
        )
        return X.rvs(size)
