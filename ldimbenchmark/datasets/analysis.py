from ldimbenchmark.datasets.classes import Dataset
import os
import pandas as pd
import wntr
import matplotlib.pyplot as plt


class DatasetAnalyzer:
    """
    Analyze a dataset
    """

    def __init__(self, analyisis_out_dir: str):
        self.analyisis_out_dir = analyisis_out_dir

    def compare(self, datasets: list[Dataset]):
        """
        Compare the datasets
        """

        # original_dataset = pd.read_csv(dataset_source_dir, index_col="Timestamp")

        # plot = original_dataset["J-02"].plot()
        # normalDataset["J-02"].plot(ax=plot.axes)
        # uniformDataset["J-02"].plot(ax=plot.axes)

        # first_figure = plt.figure()
        # first_figure_axis = first_figure.add_subplot()
        # first_figure_axis.plot(noise)

        # first_figure = plt.figure()
        # first_figure_axis = first_figure.add_subplot()
        # count, bins, ignored = first_figure_axis.hist(noise, 30, density=True)
        # sigma = 0.01 / 3
        # mu = 0
        # first_figure_axis.plot(
        #     bins,
        #     1
        #     / (sigma * np.sqrt(2 * np.pi))
        #     * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
        #     linewidth=2,
        #     color="r",
        # )

    def analyze(self, datasets: Dataset | list[Dataset]):
        """
        Analyze the dataset
        """
        if datasets is not list:
            dataset_list: list[Dataset] = [datasets]
        else:
            dataset_list = datasets

        os.makedirs(self.analyisis_out_dir, exist_ok=True)

        network_models = {}
        network_model_details = {}

        for dataset in dataset_list:
            loadedDataset = dataset.loadBenchmarkData()

            network_models[dataset.name] = loadedDataset
            network_model_details[dataset.name] = pd.json_normalize(
                loadedDataset.model.describe(3)
            )

            fig, ax = plt.subplots(1, 1, figsize=(60, 40))
            ax = wntr.graphics.plot_network(
                loadedDataset.model,
                ax=ax,
                node_size=10,
                title=f"{dataset} Network",
                node_labels=True,
                link_labels=True,
            )
            fig.savefig(
                os.path.join(self.analyisis_out_dir, f"{dataset.name}_network.png")
            )

        overview = pd.concat(network_model_details)
        overview.to_csv(
            os.path.join(self.analyisis_out_dir, "network_model_details.csv")
        )
        overview.style.to_latex(
            os.path.join(self.analyisis_out_dir, "network_model_details.tex")
        )

        # TODO: add total flow analysis
