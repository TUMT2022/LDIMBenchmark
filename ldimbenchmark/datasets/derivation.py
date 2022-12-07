import os

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(self, datasets: Dataset | list[Dataset], out_path: str):

        # TODO: Check if datasets is a list or a single dataset

        self.datasets = datasets
        self.out_path = out_path

    # TODO: Add more derivations, like elevation

    def derive(self, derivation: str, values: list):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param values: Values for the derivation
        """

        newDatasets = []
        for dataset in self.datasets:

            if derivation == "noise":
                loadedDataset = dataset.load()
                # TODO Implement derivates
                for value in values:
                    derivedDatasetPath = os.path.join(
                        self.out_path, f"{dataset.name}-{derivation}-{value}"
                    )
                    os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)

                    newDatasets.append(Dataset(derivedDatasetPath))

        return newDatasets

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
        dataset = dataset.mul(1 + noise, axis=0)
        return dataset, noise

    def _generateUniformDistributedNoise(dataset, noiseLevel):
        """
        generate noise in a uniform way between the low and high level of noiseLevel

        :param noiseLevel: noise level in percent

        """
        noise = np.random.uniform(-noiseLevel, noiseLevel, dataset.index.shape)

        dataset = dataset.mul(1 + noise, axis=0)
        return dataset, noise
