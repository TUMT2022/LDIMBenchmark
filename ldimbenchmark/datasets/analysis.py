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
        network_model_details_medium = {}
        network_model_details_fine = {}

        for dataset in dataset_list:
            loadedDataset = dataset.loadDataset()

            network_models[dataset.name] = loadedDataset
            network_model_details[dataset] = pd.json_normalize(
                loadedDataset.model.describe()
            )
            network_model_details_medium[dataset] = pd.json_normalize(
                loadedDataset.model.describe(1)
            )
            network_model_details_fine[dataset] = pd.json_normalize(
                loadedDataset.model.describe(2)
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
        overview_medium = pd.concat(network_model_details_medium)
        overview_fine = pd.concat(network_model_details_fine)

        overview = overview.reset_index(level=1, drop=True)
        overview_medium = overview_medium.reset_index(level=1, drop=True)
        overview_fine = overview_fine.reset_index(level=1, drop=True)

        overview.to_csv(
            os.path.join(self.analyisis_out_dir, "network_model_details.csv")
        )
        overview_medium.to_csv(
            os.path.join(self.analyisis_out_dir, "network_model_details_medium.csv")
        )
        overview_fine.to_csv(
            os.path.join(self.analyisis_out_dir, "network_model_details_fine.csv")
        )

        overview_table = pd.concat(
            [
                overview[["Controls"]],
                overview_medium[
                    [
                        "Nodes.Junctions",
                        "Nodes.Tanks",
                        "Nodes.Reservoirs",
                        "Links.Pipes",
                        "Links.Pumps",
                        "Links.Valves",
                    ]
                ],
            ],
            axis=1,
        )
        # overview_table.index = overview_table.index.rename("LDM")
        # overview_table.index.rename("LDM", inplace=True)

        overview_table = overview_table.rename(
            columns={
                "Controls": "Controls",
                "Nodes.Junctions": "Junctions",
                "Nodes.Tanks": "Tanks",
                "Nodes.Reservoirs": "Reservoirs",
                "Links.Pipes": "Pipes",
                "Links.Pumps": "Pumps",
                "Links.Valves": "Valves",
            }
        )

        # .hide(axis="index") \
        # .set_table_styles([
        #     {'selector': 'toprule', 'props': ':hline;'},
        #     {'selector': 'midrule', 'props': ':hline;'},
        #     {'selector': 'bottomrule', 'props': ':hline;'},
        # ], overwrite=False) \
        overview_table.style.format(escape="latex").set_table_styles(
            [
                # {'selector': 'toprule', 'props': ':hline;'},
                {"selector": "midrule", "props": ":hline;"},
                # {'selector': 'bottomrule', 'props': ':hline;'},
            ],
            overwrite=False,
        ).to_latex(
            # .relabel_index(["", "B", "C"], axis="columns") \
            os.path.join(self.analyisis_out_dir, "network_model_details.tex"),
            position_float="centering",
            clines="all;data",
            column_format="l|rrrrrrr",
            position="H",
            label="table:networks_overview",
            caption="Overview of the water networks.",
        )

        # TODO: add total flow analysis
        # TODO: Add dataset granularity of the sensors (5min, 30min)
