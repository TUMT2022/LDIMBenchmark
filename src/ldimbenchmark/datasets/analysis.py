from math import nan
from ldimbenchmark.datasets.classes import Dataset
import os
import pandas as pd
import wntr
import matplotlib.pyplot as plt
from typing import Literal, Union, List


class DatasetAnalyzer:
    """
    Analyze a dataset
    """

    def __init__(self, analyisis_out_dir: str):
        self.analysis_out_dir = analyisis_out_dir
        os.makedirs(self.analysis_out_dir, exist_ok=True)

    def compare(
        self,
        datasets: List[Dataset],
        data_type: Literal["demands", "pressures", "flows", "levels"],
    ):
        """
        Compare the datasets, e.g. especially helpful when comparing the original dataset with derived ones.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]
        dataset_list: List[Dataset] = datasets

        datasets_info = {}
        for dataset in dataset_list:
            datasets_info[dataset.id] = pd.json_normalize(dataset.info, max_level=0)

        datasets_info = pd.concat(datasets_info)
        datasets_info = datasets_info.reset_index(level=1, drop=True)

        original_dataset_frame = datasets_info[datasets_info["derivations"].isnull()]
        if original_dataset_frame.shape[0] == 0:
            raise Exception("No original dataset found")
        if original_dataset_frame.shape[0] > 1:
            raise Exception("More than one original dataset found")
        original_dataset_id = original_dataset_frame.index[0]

        loaded_datasets = {}
        for dataset in dataset_list:
            loadedDataset = dataset.loadData()
            loaded_datasets[dataset.id] = loadedDataset

        original_dataset = loaded_datasets[original_dataset_id]
        del loaded_datasets[original_dataset_id]

        # # Plot each time series
        # # TODO: Only plot timeseries with difference...
        # for dataset_id, dataset in loaded_datasets.items():
        #     if dataset.info["derivations"] is not None:
        #         if dataset.info["derivations"]["data"] is not None:
        #             data_name = dataset.info["derivations"]["data"][0]["kind"]

        data = getattr(original_dataset, data_type)
        for key in data.keys():
            # data.columns = [f"[Original] {col}" for col in data.columns]
            DatasetAnalyzer._plot_time_series(
                data[key],
                data_type,
                self.analysis_out_dir,
                [
                    getattr(ldata, data_type)[key]
                    for i, ldata in loaded_datasets.items()
                ],
            )

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

    def _plot_time_series(
        df: pd.DataFrame,
        title: str,
        out_dir: str,
        compare_df: List[pd.DataFrame] = None,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.set_title(title)

        if compare_df is not None:
            for compare in compare_df:
                compare.plot(
                    ax=ax,
                    alpha=0.5,
                    marker="o",
                )
            df.plot(
                ax=ax,
                marker="x",
            )
            ax.legend([f"[Original] {label}" for label in df.columns])
        else:
            df.plot(ax=ax)

        fig.savefig(os.path.join(out_dir, f"{title}_{df.columns[0]}.png"))
        plt.close(fig)

    def analyze(self, datasets: Union[Dataset, List[Dataset]]):
        """
        Analyze the dataset
        """
        if type(datasets) is not list:
            dataset_list: List[Dataset] = [datasets]
        else:
            dataset_list = datasets

        datasets_table = {}

        network_model_details = {}
        network_model_details_medium = {}
        network_model_details_fine = {}

        for dataset in dataset_list:
            datasets_table[dataset.id] = pd.json_normalize(dataset.info)

            network_model_details[dataset.id] = pd.json_normalize(
                dataset.model.describe()
            )
            network_model_details_medium[dataset.id] = pd.json_normalize(
                dataset.model.describe(1)
            )
            network_model_details_fine[dataset.id] = pd.json_normalize(
                dataset.model.describe(2)
            )

            dataset_analysis_out_dir = os.path.join(self.analysis_out_dir, dataset.id)
            os.makedirs(dataset_analysis_out_dir, exist_ok=True)

            dataset.loadData()
            # Plot each time series
            for data_name in ["demands", "pressures", "flows", "levels"]:
                data_group = getattr(dataset, data_name)
                for sensor_name, sensor_data in data_group.items():
                    if sensor_data.shape[1] > 0:
                        DatasetAnalyzer._plot_time_series(
                            sensor_data,
                            f"{dataset.id}: {data_name}",
                            dataset_analysis_out_dir,
                        )

            # Plot Network
            fig, ax = plt.subplots(1, 1, figsize=(60, 40))
            ax = wntr.graphics.plot_network(
                dataset.model,
                ax=ax,
                node_size=10,
                title=f"{dataset} Network",
                node_labels=True,
                link_labels=True,
            )
            fig.savefig(
                os.path.join(dataset_analysis_out_dir, f"network_{dataset.id}.png")
            )

        datasets_table = pd.concat(datasets_table)
        overview = pd.concat(network_model_details)
        overview_medium = pd.concat(network_model_details_medium)
        overview_fine = pd.concat(network_model_details_fine)

        datasets_table = datasets_table.reset_index(level=1, drop=True)
        overview = overview.reset_index(level=1, drop=True)
        overview_medium = overview_medium.reset_index(level=1, drop=True)
        overview_fine = overview_fine.reset_index(level=1, drop=True)

        overview_medium.to_csv(
            os.path.join(dataset_analysis_out_dir, "network_model_details_medium.csv")
        )
        overview_fine.to_csv(
            os.path.join(dataset_analysis_out_dir, "network_model_details_fine.csv")
        )

        overview_table = pd.concat(
            [
                datasets_table[["name"]],
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

        overview_table.to_csv(
            os.path.join(self.analysis_out_dir, "network_model_details.csv")
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
            os.path.join(self.analysis_out_dir, "network_model_details.tex"),
            position_float="centering",
            clines="all;data",
            column_format="l|rrrrrrr",
            position="H",
            label="table:networks_overview",
            caption="Overview of the water networks.",
        )

        # TODO: add total flow analysis
        # TODO: Add dataset granularity of the sensors (5min, 30min)
