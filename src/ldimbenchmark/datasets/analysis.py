from asyncio import as_completed
from concurrent.futures import ProcessPoolExecutor
from math import nan
import math
from ldimbenchmark.benchmark.benchmark import plot_leak
from ldimbenchmark.constants import CPU_COUNT
from ldimbenchmark.datasets.classes import Dataset
import os
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
from typing import Literal, Union, List

from ldimbenchmark.utilities import (
    delta_format,
    get_unit_for_property,
    read_multiple_dataset_infos,
)


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
        quick: bool = False,
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
        datasets_info[datasets_info["derivations"].isnull()] = "{}"
        datasets_info = datasets_info.rename(
            columns={"derivations": "dataset_derivations"}
        )

        datasets_info = read_multiple_dataset_infos(datasets_info)

        original_dataset_frame = datasets_info[datasets_info["is_original"] == True]
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

        derivated_dataset_frame = datasets_info[datasets_info["is_original"] == False]
        derived_property = derivated_dataset_frame["dataset_derivations.data.to"].iloc[
            0
        ]

        derived_type = derivated_dataset_frame["dataset_derivations.data.kind"].iloc[0]
        # # Plot each time series
        # # TODO: Only plot timeseries with difference...
        # for dataset_id, dataset in loaded_datasets.items():
        #     if dataset.info["derivations"] is not None:
        #         if dataset.info["derivations"]["data"] is not None:
        #             data_name = dataset.info["derivations"]["data"][0]["kind"]

        data = getattr(original_dataset, data_type)

        unit = data_type[:-1] + " " + get_unit_for_property(data_type)

        for sensor_id in data.keys():
            # data.columns = [f"[Original] {col}" for col in data.columns]
            DatasetAnalyzer._plot_time_series(
                data[sensor_id],
                os.path.join(
                    self.analysis_out_dir,
                    f"comparison_{original_dataset.name}_{derived_property}_{derived_type}_{sensor_id}{'_quick' if quick else ''}.png",
                ),
                # title=f"Compare '{data_type}' of {original_dataset.name}",
                compare_df=[
                    getattr(ldata, data_type)[sensor_id]
                    for i, ldata in loaded_datasets.items()
                ],
                quick_detail=quick,
                unit=unit,
            )
            break

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
        out_path: str,
        title: str = None,
        compare_df: List[pd.DataFrame] = None,
        quick_detail: bool = False,
        unit: str = None,
    ):
        """
        Plots the time series data of each sensor and possible data for comparison

        out_path: Path to save the plot (with filename)
        compare_df: List of dataframes to compare with
        quick_detail: If true, only plot the first 40 timestamps
        """

        max_timestamp = df.index[-1]
        # default figsize
        figsize = (8, 6)

        if quick_detail:
            max_timestamp = df.index[40]
            figsize = (15, 5)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(title)

        min_timestamp = df.index[0]

        if compare_df is not None:
            for compare in compare_df:
                compare[
                    (compare.index > min_timestamp) & (compare.index < max_timestamp)
                ].add_prefix("[Comparison] ").plot(
                    ax=ax,
                    alpha=0.5,
                    marker="o",
                )
        df[(df.index > min_timestamp) & (df.index < max_timestamp)].add_prefix(
            "[Original] "
        ).plot(
            ax=ax,
            marker="x",
        )

        ax.set_ylabel(unit)
        ax.legend(loc="upper left")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # Plot Overview of timeseries sensors
    def _plot_sensor_dates(sensor_data, labels, filename):
        colors = ["C{}".format(i) for i in range(len(labels))]

        fig, axs = plt.subplots(1, 1, figsize=(90, len(labels) / 4))
        # # create a horizontal plot
        axs.eventplot(sensor_data, label=labels, colors=colors)
        axs.set_yticks(range(len(sensor_data)))
        axs.set_yticklabels(labels)

        # x_ticks=np.array(sensor_data.index)
        # x_ticks_1=pd.date_range(start=x_ticks.min(), end=x_ticks.max())
        # axs.set_xticklabels(x_ticks_1,rotation = 45)
        plt.show()
        plt.savefig("out/" + filename + ".png")
        plt.close(fig)

    def analyze(self, datasets: Union[Dataset, List[Dataset]]):
        """
        Analyze the dataset
        """
        if type(datasets) is not list:
            dataset_list: List[Dataset] = [datasets]
        else:
            dataset_list = datasets

        datasets_table = []
        datasets_table_common = {}

        network_model_details = {}
        network_model_details_medium = {}
        network_model_details_fine = {}

        for dataset in dataset_list:
            info_table = pd.json_normalize(dataset.info)
            info_table.index = [dataset.id]
            datasets_table.append(info_table)

            length = 0
            for pipe_id in dataset.model.pipe_name_list:
                pipe = dataset.model.get_link(pipe_id)
                length += pipe.length

            model_description = dataset.model.describe(1)
            model_description["overall_length"] = length
            network_model_details[dataset.id] = pd.json_normalize(
                dataset.model.describe()
            )
            network_model_details_medium[dataset.id] = pd.json_normalize(
                model_description
            )
            network_model_details_fine[dataset.id] = pd.json_normalize(
                dataset.model.describe(2)
            )

            dataset_analysis_out_dir = os.path.join(self.analysis_out_dir, dataset.id)
            os.makedirs(dataset_analysis_out_dir, exist_ok=True)

            dataset.loadData()
            intervals = []
            # Plot each time series
            for data_name in ["demands", "pressures", "flows", "levels"]:
                data_group = getattr(dataset, data_name)
                for sensor_name, sensor_data in data_group.items():
                    if sensor_data.shape[1] > 0:
                        # DatasetAnalyzer._plot_time_series(
                        #     sensor_data,
                        #     os.path.join(dataset_analysis_out_dir, f"{title}_{df.columns[0]}.png")
                        #     f"{dataset.id}_{data_name}",
                        # )
                        minTime = sensor_data.index.min()
                        maxTime = sensor_data.index.max()
                        timeLength = maxTime - minTime
                        datapoint_count = len(sensor_data.index)
                        interval_median = timeLength / datapoint_count
                        intervals.append(interval_median)

            common_table = {}
            common_table["interval_avg"] = np.average(intervals)
            common_table["interval_med"] = np.median(intervals)
            common_table["interval_min"] = min(intervals)
            common_table["interval_max"] = max(intervals)
            leaks_analysis = dataset.leaks
            leaks_analysis["duration"] = (
                dataset.leaks["leak_time_end"] - dataset.leaks["leak_time_start"]
            )
            mean_duration = leaks_analysis["duration"].mean().round("1s")
            leaks_analysis["smaller"] = leaks_analysis["duration"] < mean_duration
            leaks_analysis["longer"] = leaks_analysis["duration"] >= mean_duration
            common_table["leaks_number"] = len(dataset.leaks)
            common_table["leaks_duration_mean"] = mean_duration
            common_table["leaks_duration_shortest"] = (
                leaks_analysis["duration"].min().round("1s")
            )
            common_table["leaks_duration_longest"] = (
                leaks_analysis["duration"].max().round("1s")
            )
            common_table["leaks_shorter_then_mean"] = leaks_analysis["smaller"].sum()
            common_table["leaks_longer_then_mean"] = leaks_analysis["longer"].sum()
            common_table["leaks_no_duration"] = leaks_analysis["duration"].isna().sum()

            common_table["sensor_count_demands"] = len(dataset.demands)
            common_table["sensor_count_pressures"] = len(dataset.pressures)
            common_table["sensor_count_flows"] = len(dataset.flows)
            common_table["sensor_count_levels"] = len(dataset.levels)

            common_table["sensor_count_demands_ratio"] = (
                (common_table["sensor_count_demands"] / dataset.model.num_nodes)
                if common_table["sensor_count_demands"] > 0
                else 0
            )
            common_table["sensor_count_pressures_ratio"] = (
                (common_table["sensor_count_pressures"] / dataset.model.num_pipes)
                if common_table["sensor_count_pressures"] > 0
                else 0
            )
            common_table["sensor_count_flows_ratio"] = (
                (common_table["sensor_count_flows"] / dataset.model.num_pipes)
                if common_table["sensor_count_flows"] > 0
                else 0
            )
            common_table["sensor_count_levels_ratio"] = (
                (common_table["sensor_count_levels"] / dataset.model.num_tanks)
                if common_table["sensor_count_levels"] > 0
                else 0
            )

            datasets_table_common[dataset.id] = common_table

            # Plot Network
            size = math.log(len(dataset.model.link_name_list)) * 2
            fig, ax = plt.subplots(1, 1, figsize=(size * 1.33, size))
            ax = wntr.graphics.plot_network(
                dataset.model,
                ax=ax,
                node_size=10,
                # title=f"{dataset.name} Network",
                # node_labels=True,
                # link_labels=True,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    dataset_analysis_out_dir,
                    f"network_{dataset.id}.png",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            # PLot leaks
            # parallel = False
            # if parallel:
            #     with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            #         # submit all tasks and get future objects
            #         futures = []
            #         for leak in dataset.leaks.iterrows():
            #             future = executor.submit(
            #                 plot_leak,
            #                 dataset,
            #                 leak_pair=(leak[1], None),
            #                 out_dir=dataset_analysis_out_dir,
            #                 compare_leaks=False,
            #             )
            #             futures.append(future)

            #         # process results from tasks in order of task completion
            #         for future in as_completed(futures):
            #             future.result()

            # else:
            #     for leak in dataset.leaks.iterrows():
            #         plot_leak(
            #             dataset,
            #             leak_pair=(leak[1], None),
            #             out_dir=dataset_analysis_out_dir,
            #             compare_leaks=False,
            #         )

            # TODO: Plot Leak Overview

            # mask = (
            #     sensors.index >= leaks.start_times.min() - timedelta(minutes=20)
            # ) & (sensors.index <= leaks.end_times.max() + timedelta(minutes=20))
            # maskedsensors = sensors[mask]

            # ax = maskedsensors.plot(figsize=(18, 6))

            # ax.legend(
            #     loc="upper center",
            #     bbox_to_anchor=(0.5, 1.3),
            #     ncol=3,
            #     fancybox=True,
            #     shadow=True,
            # )

            # for i, leak in leaks.iterrows():
            #     plt.axvspan(leak.start_times, leak.end_times, color="red", alpha=0.5)

        datasets_table_common = pd.DataFrame.from_dict(
            datasets_table_common, orient="index"
        )

        datasets_table = pd.concat(datasets_table)
        datasets_table = pd.concat([datasets_table, datasets_table_common], axis=1)

        overview = pd.concat(network_model_details)
        overview_medium = pd.concat(network_model_details_medium)
        overview_fine = pd.concat(network_model_details_fine)

        overview = overview.reset_index(level=1, drop=True)
        overview_medium = overview_medium.reset_index(level=1, drop=True)
        overview_fine = overview_fine.reset_index(level=1, drop=True)

        overview_medium.to_csv(
            os.path.join(dataset_analysis_out_dir, "network_model_details_medium.csv")
        )
        overview_fine.to_csv(
            os.path.join(dataset_analysis_out_dir, "network_model_details_fine.csv")
        )

        datasets_table["time_duration"] = (
            datasets_table["dataset.evaluation.end"]
            - datasets_table["dataset.training.start"]
        )
        datasets_table["time_duration_evaluation"] = (
            datasets_table["dataset.evaluation.end"]
            - datasets_table["dataset.evaluation.start"]
        )
        datasets_table["time_duration_training"] = (
            datasets_table["dataset.training.end"]
            - datasets_table["dataset.training.start"]
        )
        # TODO include leak free time interval

        datasets_table = pd.concat(
            [datasets_table, overview, overview_medium, overview_fine], axis=1
        )
        # Remove duplicate columns
        datasets_table = datasets_table.loc[
            :, ~datasets_table.columns.duplicated()
        ].copy()

        datasets_table.to_csv(
            os.path.join(self.analysis_out_dir, "overview_all_data.csv")
        )

        # TODO: Merge tables
        overview_table = datasets_table[
            [
                "name",
                "Controls",
                "Nodes.Junctions",
                "Nodes.Tanks",
                "Nodes.Reservoirs",
                "Links.Pipes",
                "Links.Pumps",
                "Links.Valves",
                "overall_length",
                "time_duration",
                "interval_min",
                "interval_max",
                "sensor_count_demands",
                "sensor_count_pressures",
                "sensor_count_flows",
                "sensor_count_levels",
                "leaks_number",
                "leaks_duration_mean",
                "leaks_shorter_then_mean",
                "leaks_longer_then_mean",
            ]
        ]

        formatters = {
            "overall_length": "\\SI{{{:,.0f}}}{{\\meter}}",
            "time_duration": lambda v: "\\begin{tabular}{@{}c@{}}"
            + str(v)[: str(v).find("days") + 4]
            + "\\\\"
            + str(v)[str(v).find("days") + 4 :]
            + "\\end{tabular}",
            "leaks_duration_mean": lambda v: "\\begin{tabular}{@{}c@{}}"
            + str(v)[: str(v).find("days") + 4]
            + "\\\\"
            + str(v)[str(v).find("days") + 4 :]
            + "\\end{tabular}",
            "interval_min": lambda v: delta_format(v),
            "interval_max": lambda v: delta_format(v),
        }
        for key, value in formatters.items():
            # if function apply function
            if callable(value):
                overview_table[key] = overview_table[key].apply(value)
            else:
                overview_table[key] = overview_table[key].apply(value.format)

        overview_table = (
            overview_table
            # # Fix index column
            # # https://stackoverflow.com/questions/46797598/how-to-remove-extra-row-after-set-index-without-losing-index-name
            .reset_index(drop=True)
            .set_index("name")
            .rename_axis(None, axis=0)
            .rename_axis("name", axis=1)
            .rename(
                columns={
                    "Controls": "Controls",
                    "Nodes.Junctions": "Junctions",
                    "Nodes.Tanks": "Tanks",
                    "Nodes.Reservoirs": "Reservoirs",
                    "Links.Pipes": "Pipes",
                    "Links.Pumps": "Pumps",
                    "Links.Valves": "Valves",
                    "overall_length": "Network Length",
                    "time_duration": "timespan",
                    "interval_min": "min internal",
                    "interval_max": "max interval",
                    "leaks_number": "Leaks",
                    "leaks_duration_mean": "Leak Duration",
                    "leaks_shorter_then_mean": "shorter Leaks",
                    "leaks_longer_then_mean": "longer Leaks",
                    "sensor_count_demands": "Demand Sensors",
                    "sensor_count_pressures": "Pressure Sensors",
                    "sensor_count_flows": "Flow Sensors",
                    "sensor_count_levels": "Level Sensors",
                }
            )
        ).T

        overview_table.index = overview_table.index.rename("property")
        overview_table["class"] = np.concatenate(
            [
                np.repeat("model", 8),
                np.repeat("data", 7),
                np.repeat("leaks", len(overview_table) - 15),
            ]
        )
        overview_table = overview_table.reset_index().set_index(["class", "property"])

        overview_table.style.format_index(
            # formatter=lambda x: "\\rotatebox{45}{" + x + "}", axis="columns"
            formatter=lambda x: x,
            axis="columns",
        ).format_index(
            formatter=lambda x: "\\rotatebox[origin=c]{-90}{" + x + "}"
            if x == "model" or x == "data" or x == "leaks"
            else x,
            # formatter=lambda x: x,
            axis="index",
        ).set_table_styles(
            [
                # {'selector': 'toprule', 'props': ':hline;'},
                {"selector": "midrule", "props": ":hline;"},
                # {'selector': 'bottomrule', 'props': ':hline;'},
            ],
            overwrite=False,
        ).to_latex(
            os.path.join(self.analysis_out_dir, "network_model_overview.tex"),
            position_float="centering",
            column_format="ll|rrrr",
            position="H",
            clines="skip-last;data",
            multirow_align="c",
            label="table:networks_overview",
            caption="Overview of the water networks.",
        )

        # Data

        # TODO: add total flow analysis
