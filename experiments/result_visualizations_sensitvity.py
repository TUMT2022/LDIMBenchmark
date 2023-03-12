# %%
# %load_ext autoreload
# %autoreload 2
import json
from typing import List
import pandas as pd
from sqlalchemy import create_engine
import os

import itertools

from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset
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


results_db_path = os.path.join(
    "sensitivity-analysis", "evaluation_results", "results.db"
)
engine = create_engine(f"sqlite:///{results_db_path}")
results = pd.read_sql("results", engine, index_col="_folder")  # .tail(-1)

# results.hyperparameters = results.hyperparameters.astype("str")
# df_hyperparameters = pd.json_normalize(
#     results.hyperparameters.apply(ast.literal_eval)
# ).add_prefix("hyperparameters.")
# df_hyperparameters.index = results.index
# df_hyperparameters
# # results = results.drop(columns=["hyperparameters"])
# results = pd.concat([results, df_hyperparameters], axis=1)

results.dataset_derivations = results.dataset_derivations.astype("str")
results.dataset_derivations
results["is_original"] = results.dataset_derivations == "{}"
# df_dataset_derivations = pd.json_normalize(results.dataset_derivations)
# df_dataset_derivations

df_dataset_derivations = pd.json_normalize(
    results.dataset_derivations.apply(ast.literal_eval),
    # results.dataset_derivations,
    # record_path=["data"],
    # meta=['school', ['info', 'contacts', 'tel']],
    # results.dataset_derivations.apply(ast.literal_eval),
    # "data",
    # record_prefix="dataset_derivations",
    errors="ignore",
).add_prefix("dataset_derivations.")

df_dataset_derivations
derivations_data = pd.json_normalize(
    df_dataset_derivations["dataset_derivations.data"].explode(
        "dataset_derivations.data"
    )
).add_prefix("dataset_derivations.data.")
derivations_data.index = results.index


derivations_model = pd.json_normalize(
    df_dataset_derivations["dataset_derivations.model"].explode(
        "dataset_derivations.model"
    )
).add_prefix("dataset_derivations.model.")
derivations_model.index = results.index

derivations_model
flattened_results = pd.concat([results, derivations_data, derivations_model], axis=1)
flattened_results["dataset_derivations.value"] = flattened_results[
    "dataset_derivations.data.value"
].fillna(flattened_results["dataset_derivations.data.value.value"])
flattened_results["dataset_derivations.value"] = flattened_results[
    "dataset_derivations.value"
].fillna(flattened_results["dataset_derivations.model.value"])
flattened_results = flattened_results.drop(
    columns=["dataset_derivations.data.value.value", "dataset_derivations.model.value"]
)

# Fill Nan values for F1 score
flattened_results["F1"] = flattened_results["F1"].fillna(0)
# Set derivation factor for original datasets to 0
flattened_results["dataset_derivations.value"] = flattened_results[
    "dataset_derivations.value"
].fillna(0)

# %%

performance_indicator = "F1"
base_out_folder = "out"
applied_to = "pressures"

derivations = [
    ("dataset_derivations.data.kind", "precision"),
    ("dataset_derivations.data.kind", "sensitivity"),
    ("dataset_derivations.data.kind", "downsample"),
    ("dataset_derivations.model.property", "elevation"),
]
colors = ["C0", "C1", "C2"]

for col, value in derivations:
    table_data = flattened_results[
        ((flattened_results[col] == value) | (flattened_results["is_original"]))
    ]
    table_data.set_index(["dataset", "method", "dataset_derivations.value"])[
        ["F1"]
    ].reorder_levels(
        ["dataset", "method", "dataset_derivations.value"]
    ).stack().unstack(
        level=2
    ).to_csv(
        f"out/{value}.csv"
    )
    for dataset in flattened_results["dataset"].unique():
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        for num, method in enumerate(flattened_results["method"].unique()):
            method_data = flattened_results[
                ((flattened_results[col] == value) | (flattened_results["is_original"]))
                & (flattened_results["dataset"] == dataset)
                & (flattened_results["method"] == method)
            ].sort_values(by="dataset_derivations.value")
            method_data
            spacing = np.array(range(0, len(method_data["dataset_derivations.value"])))
            ax.plot(
                spacing,
                # method_data["dataset_derivations.value"],
                method_data[performance_indicator],
                label=method,
                # alpha=0.5,
                marker="o",
                color=colors[num],
                # ax=ax,
            )
            offset = num * 0.1 - 0.1
            ax2.bar(
                spacing + offset,
                # method_data["dataset_derivations.value"] + offset,
                method_data["true_positives"],
                label=method,
                width=0.1,
                alpha=0.5,
                color=colors[num],
            )

            ax2.bar(
                spacing + offset,
                # method_data["dataset_derivations.value"] + offset,
                method_data["false_positives"],
                bottom=method_data["true_positives"],
                label=method,
                width=0.1,
                alpha=0.5,
                color=lighten_color(colors[num], 0.2),
            )

        # ax.secondary_yaxis("left")
        ax.set_ylim([0, 1])
        # ax2.set_ylim([0, 8])
        ax.set_xticks(ticks=spacing)
        ax.set_xticklabels(labels=method_data["dataset_derivations.value"])
        ax.set_title(f"Sensitivity Analysis ({dataset} {applied_to})")
        ax.set_ylabel(f"Performance Indicator: {performance_indicator}")
        ax2.set_ylabel(f"Performance Indicator: Leaks")
        ax.set_xlabel(f"Derivation of {value}")
        ax.legend()

# str(data_results["dataset_derivations.data"][0])
# plot_data["dataset_derivations.data"][0][0]


# %%


def create_plots(
    hyperparameters: List[str],
    performance_metric: str,
    out_folder,
):
    plot_out_folder = os.path.join(out_folder, method, dataset)
    os.makedirs(plot_out_folder, exist_ok=True)
    plot_data = results[(results["method"] == method) & (results["dataset"] == dataset)]
    hyperparameters = list(map(lambda x: "hyperparameters." + x, hyperparameters))

    hyperparameter_combination = list(itertools.combinations(hyperparameters, 2))
    for param_1, param_2 in hyperparameter_combination:
        fig, ax = plt.subplots()

        pvt = pd.pivot_table(
            plot_data,
            values=performance_metric,
            index=param_1,
            columns=param_2,  # , "hyperparameters.delta"]
        )
        cmap = sns.cm.rocket_r
        sns.heatmap(pvt, ax=ax, cmap=cmap)
        ax.set_title(f"Heatmap of {method}-{dataset}")
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        fig.savefig(os.path.join(plot_out_folder, f"heatmap.png"))
        # plt.close(fig)


derivations = {"data": [{"to": "pressures", "kind": "precision", "value": 1.0}]}

create_plots(
    derivations,
    performance_indicator,
    base_out_folder,
)


# %%
lila = results[(results["method"] == "mnf") & (results["dataset"] == "battledim")]


df_hyperparameters = ["est_length", "C_threshold", "delta"]

for param in df_hyperparameters:
    frame = lila.groupby(f"hyperparameters.{param}").max(numeric_only=True)
    fig, ax = plt.subplots()
    ax.scatter(
        frame.index,
        frame[performance_indicator],
    )
    ax.set_title(f"Hyperparameter: {param}")
    ax.set_xlabel(f"Value of Hyperparameter ({param})")
    ax.set_ylabel(f"Performance metric ({performance_indicator})")
    plt.show()

# %%
# Scatter
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(
    lila["hyperparameters.est_length"],
    lila["hyperparameters.C_threshold"],
    # lila["hyperparameters.delta"],
    lila[performance_indicator],
    alpha=0.2,
    c=lila[performance_indicator],
    cmap="inferno_r",
)
plt.show()


# %%
# Plot other 3D Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_trisurf(
    lila["hyperparameters.est_length"],
    lila["hyperparameters.C_threshold"],
    lila[performance_indicator],
    linewidth=0,
    antialiased=False,
)


# Customize the z axis.
# ax.set_zlim(0, 5)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter("{x:.02f}")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# %%
# Plot 3D plot

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(
    lila["hyperparameters.est_length"],
    lila["hyperparameters.C_threshold"],
    # sparse=True,
)
Z = np.outer(lila[performance_indicator], lila[performance_indicator].T)

# Plot the surface.
surf = ax.plot_surface(
    X,
    Y,
    Z,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter("{x:.02f}")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
Z
# %%
