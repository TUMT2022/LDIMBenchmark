# %%
# %load_ext autoreload
# %autoreload 2
import datetime
import json
from typing import List, Tuple
import pandas as pd
from sqlalchemy import create_engine
import os

import itertools

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import ast
import seaborn as sns

from ldimbenchmark.utilities import (
    delta_format,
    get_unit_for_property,
    read_multiple_dataset_infos,
)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_derivation_plot(
    flat_results: pd.DataFrame,
    derivations: List[Tuple[str, str]],
    applied_to: List[Tuple[str, str]],
    out_folder: str,
    performance_indicator: str = "F1",
):
    colors = ["C0", "C1", "C2"]

    # TODO add tex output
    #     table_data = flat_results[
    #     (
    #         (
    #             ((flat_results[col_derivation] == derivation_type))
    #             & (flat_results[col_modified] == modified_property)
    #         )
    #         | (flat_results["is_original"])
    #     )
    # ]
    # overview_data = (
    #     flat_results.set_index(["dataset", "method", "dataset_derivations.value"])[
    #         ["F1"]
    #     ]
    #     .reorder_levels(["dataset", "method", "dataset_derivations.value"])
    #     .stack()
    #     .unstack(level=2)
    # )
    # overview_data.to_csv(f"out/{derivation_type}.csv")

    for col_modified, modified_property in applied_to:
        for col_derivation, derivation_type in derivations:
            for dataset in flat_results["dataset"].unique():
                fig, ax = plt.subplots(figsize=(15, 8))
                ax2 = ax.twinx()
                ax3 = ax.twinx()
                ax3.spines.right.set_position(("axes", 1.07))

                maxrows = (
                    flat_results[
                        (
                            (
                                (flat_results[col_derivation] == derivation_type)
                                & (flat_results[col_modified] == modified_property)
                            )
                            | (flat_results["is_original"])
                        )
                        & (flat_results["dataset"] == dataset)
                    ]
                    .groupby("method")
                    .count()["dataset_derivations.value"]
                    .max()
                )

                for num, method in enumerate(flat_results["method"].unique()):
                    method_data = flat_results[
                        (
                            (
                                (flat_results[col_derivation] == derivation_type)
                                & (flat_results[col_modified] == modified_property)
                            )
                            | (flat_results["is_original"])
                        )
                        & (flat_results["dataset"] == dataset)
                        & (flat_results["method"] == method)
                    ].sort_values(
                        # Make sure original dataset comes first
                        by=["dataset_derivation_type", "dataset_derivations.value"],
                        ascending=derivation_type != "count",
                        na_position="first",
                    )
                    method_data = method_data.reindex(
                        method_data.index.append(
                            pd.Index(range(maxrows - len(method_data)), name="_folder")
                        )
                    )
                    if method_data["dataset_derivations.value"].notna().all():
                        labels = method_data["dataset_derivations.value"]
                    method_data["true_positives"] = method_data[
                        "true_positives"
                    ].fillna(0)
                    method_data["false_positives"] = method_data[
                        "false_positives"
                    ].fillna(0)
                    spacing = np.array(
                        range(0, len(method_data["dataset_derivations.value"]))
                    )

                    bar_width = 0.3
                    offset = num * bar_width - bar_width  # - (bar_width / 2)
                    ax.bar(
                        spacing + offset,
                        method_data["true_positives"],
                        label=f"{method}: True Positives",
                        width=bar_width,
                        alpha=0.5,
                        color=colors[num],
                    )

                    # TODO: Explore negative grahpical display of false positives
                    ax.bar(
                        spacing + offset,
                        method_data["false_positives"],
                        bottom=method_data["true_positives"],
                        label=f"{method}: False Positives",
                        width=bar_width,
                        alpha=0.5,
                        color=lighten_color(colors[num], 0.2),
                    )
                    ax2.plot(
                        spacing,
                        method_data[performance_indicator],
                        label=method,
                        marker="o",
                        color=colors[num],
                    )

                    ax3.plot(
                        spacing,
                        method_data["time_to_detection_avg"],
                        label=f"{method} TTD",
                        linestyle="dotted",
                        # marker="-.",
                        color=colors[num],
                    )

                ax2.set_ylim([0, 1])
                ax2.set_ylabel(f"Performance Indicator: {performance_indicator}")

                ax3.yaxis.set_major_formatter(
                    lambda x, pos: f"{delta_format(datetime.timedelta(seconds=x))}"
                )
                ax3.set_ylabel(f"Time to Detection [h]")

                ax.set_xticks(ticks=spacing)
                if derivation_type == "downsample":
                    labels = [
                        f"{delta_format(datetime.timedelta(seconds=t))} H"
                        for t in method_data["dataset_derivations.value"]
                    ]

                if derivation_type == "precision":
                    labels = [
                        "{:.0%}".format(t)
                        for t in method_data["dataset_derivations.value"]
                    ]
                # TODO: add unit conversion
                # if accuracy and diameter convert to mm

                labels[0] = "Original"
                ax.set_xticklabels(labels=labels)
                # TODO: add units
                ax.set_xlabel(f"{derivation_type} derivations")
                ax.set_ylabel(f"Detected leak count")
                # ax2.set_ylim(bottom=0, top=100)
                ax2.legend(loc="upper left")
                ax.legend(loc="upper right")
                plt.title(
                    f"Dataset: {dataset}, Property: {modified_property}", fontsize=10
                )
                plt.suptitle("Sensitivity Analysis")
                fig.savefig(
                    os.path.join(
                        out_folder,
                        f"sensitivity_{dataset}_{modified_property}_{derivation_type}",
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)


def evaluate_derivations(database_path: str, out_folder: str):
    results_db_path = os.path.join(database_path)
    engine = create_engine(f"sqlite:///{results_db_path}")
    results = pd.read_sql("results", engine, index_col="_folder")

    flattened_results = read_multiple_dataset_infos(results)

    if (flattened_results["dataset_derivation_type"] == "data").any():
        data_derivation_type = flattened_results[
            flattened_results["dataset_derivations.data.kind"].notna()
        ]["dataset_derivations.data.kind"].unique()
        data_to = flattened_results[
            flattened_results["dataset_derivations.data.to"].notna()
        ]["dataset_derivations.data.to"].unique()
    else:
        data_derivation_type = []
        data_to = []

    if (flattened_results["dataset_derivation_type"] == "model").any():
        model_derivation_property = flattened_results[
            flattened_results["dataset_derivations.model.property"].notna()
        ]["dataset_derivations.model.property"].unique()
        model_to = flattened_results[
            flattened_results["dataset_derivations.model.element"].notna()
        ]["dataset_derivations.model.element"].unique()
    else:
        model_derivation_property = []
        model_to = []

    # Fill Nan values for F1 score
    flattened_results["F1"] = flattened_results["F1"].fillna(0)
    # Set derivation factor for original datasets to 0
    flattened_results["dataset_derivations.value"] = flattened_results[
        "dataset_derivations.value"
    ].fillna(0)

    # Data based derivations
    derivations = [
        ("dataset_derivations.data.kind", derivation_type)
        for derivation_type in data_derivation_type
    ]

    applied_to = [("dataset_derivations.data.to", to) for to in data_to]
    plot_derivation_plot(
        flattened_results,
        derivations=derivations,
        applied_to=applied_to,
        out_folder=out_folder,
    )

    # Model based derivations
    derivations = [
        ("dataset_derivations.model.property", derivation_type)
        for derivation_type in model_derivation_property
    ]
    applied_to = [("dataset_derivations.model.element", to) for to in model_to]
    plot_derivation_plot(
        flattened_results,
        derivations=derivations,
        applied_to=applied_to,
        out_folder=out_folder,
    )

    # %%
