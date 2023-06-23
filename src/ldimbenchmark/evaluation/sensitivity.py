# %%
# %load_ext autoreload
# %autoreload 2
import json
from typing import List
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
    derivations: list[tuple[str, str]],
    applied_to: list[tuple[str, str]],
    out_folder: str,
    performance_indicator: str = "F1",
):
    colors = ["C0", "C1", "C2"]

    # TODO add tex output
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
            table_data = flat_results[
                (
                    (
                        ((flat_results[col_derivation] == derivation_type))
                        & (flat_results[col_modified] == modified_property)
                    )
                    | (flat_results["is_original"])
                )
            ]

            for dataset in flat_results["dataset"].unique():
                fig, ax = plt.subplots(figsize=(15, 8))
                ax2 = ax.twinx()

                for num, method in enumerate(flat_results["method"].unique()):
                    method_data = flat_results[
                        (
                            (flat_results[col_derivation] == derivation_type)
                            | (flat_results["is_original"])
                        )
                        & (flat_results["dataset"] == dataset)
                        & (flat_results["method"] == method)
                    ].sort_values(by="dataset_derivations.value")
                    method_data
                    spacing = np.array(
                        range(0, len(method_data["dataset_derivations.value"]))
                    )
                    offset = num * 0.1 - 0.1
                    ax2.bar(
                        spacing + offset,
                        method_data["true_positives"],
                        label=f"{method}: TP",
                        width=0.1,
                        alpha=0.5,
                        color=colors[num],
                    )

                    ax2.bar(
                        spacing + offset,
                        method_data["false_positives"],
                        bottom=method_data["true_positives"],
                        label=f"{method}: FP",
                        width=0.1,
                        alpha=0.5,
                        color=lighten_color(colors[num], 0.2),
                    )
                    ax.plot(
                        spacing,
                        method_data[performance_indicator],
                        label=method,
                        marker="o",
                        color=colors[num],
                    )

                ax.set_ylim([0, 1])
                ax.set_xticks(ticks=spacing)
                ax.set_xticklabels(labels=method_data["dataset_derivations.value"])
                ax.set_ylabel(f"Performance Indicator: {performance_indicator}")
                ax2.set_ylabel(f"Leak Count")
                ax.set_xlabel(f"{derivation_type} derivations")
                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")
                plt.title(
                    f"Dataset: {dataset}, Property: {modified_property}", fontsize=10
                )
                plt.suptitle("Sensitivity Analysis")
                fig.savefig(
                    os.path.join(
                        out_folder,
                        f"sensitivity_{dataset}_{modified_property}_{derivation_type}",
                    )
                )
                # plt.close(fig)


def evaluate_derivations(database_path: str, out_folder: str):
    results_db_path = os.path.join(database_path)
    engine = create_engine(f"sqlite:///{results_db_path}")
    results = pd.read_sql("results", engine, index_col="_folder")

    # results.hyperparameters = results.hyperparameters.astype("str")
    # df_hyperparameters = pd.json_normalize(
    #     results.hyperparameters.apply(ast.literal_eval)
    # ).add_prefix("hyperparameters.")
    # df_hyperparameters.index = results.index
    # df_hyperparameters
    # results = results.drop(columns=["hyperparameters"])
    # results = pd.concat([results, df_hyperparameters], axis=1)

    results.dataset_derivations = results.dataset_derivations.astype("str")
    results.dataset_derivations
    results["is_original"] = results.dataset_derivations == "{}"

    df_dataset_derivations = pd.json_normalize(
        results.dataset_derivations.apply(ast.literal_eval),
        errors="ignore",
    ).add_prefix("dataset_derivations.")

    if "dataset_derivations.data" in df_dataset_derivations:
        derivations_data = pd.json_normalize(
            df_dataset_derivations["dataset_derivations.data"].explode(
                "dataset_derivations.data"
            )
        ).add_prefix("dataset_derivations.data.")
        derivations_data.index = results.index
        flattened_results = pd.concat([results, derivations_data], axis=1)
        flattened_results["dataset_derivations.value"] = flattened_results[
            "dataset_derivations.data.value"
        ].fillna(flattened_results["dataset_derivations.data.value.value"])
        flattened_results = flattened_results.drop(
            columns=["dataset_derivations.data.value.value"]
        )
        data_derivation_type = flattened_results[
            flattened_results["dataset_derivations.data.kind"].notna()
        ]["dataset_derivations.data.kind"].unique()
        data_to = flattened_results[
            flattened_results["dataset_derivations.data.to"].notna()
        ]["dataset_derivations.data.to"].unique()
    else:
        data_derivation_type = []
        data_to = []

    if "dataset_derivations.model" in df_dataset_derivations:
        derivations_model = pd.json_normalize(
            df_dataset_derivations["dataset_derivations.model"].explode(
                "dataset_derivations.model"
            )
        ).add_prefix("dataset_derivations.model.")
        derivations_model.index = results.index
        flattened_results = pd.concat([results, derivations_model], axis=1)
        flattened_results["dataset_derivations.value"] = flattened_results[
            "dataset_derivations.value"
        ].fillna(flattened_results["dataset_derivations.model.value"])

        flattened_results = flattened_results.drop(
            columns=["dataset_derivations.model.value"]
        )
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

    derivations = [
        ("dataset_derivations.data.kind", derivation_type)
        for derivation_type in data_derivation_type
    ] + [
        ("dataset_derivations.model.property", derivation_type)
        for derivation_type in model_derivation_property
    ]

    applied_to = [("dataset_derivations.data.to", to) for to in data_to] + [
        ("dataset_derivations.model.element", to) for to in model_to
    ]
    plot_derivation_plot(
        flattened_results,
        derivations=derivations,
        applied_to=applied_to,
        out_folder=out_folder,
    )

    # %%
