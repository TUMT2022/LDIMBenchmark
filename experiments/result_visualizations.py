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

results_db_path = os.path.join("grid-search", "evaluation_results", "results.db")
engine = create_engine(f"sqlite:///{results_db_path}")
results = pd.read_sql("results", engine, index_col="_folder")

# results.hyperparameters = results.hyperparameters.astype("str")
# df_hyperparameters = pd.json_normalize(
#     results.hyperparameters.apply(ast.literal_eval)
# ).add_prefix("hyperparameters.")
# df_hyperparameters.index = results.index
# df_hyperparameters
# # results = results.drop(columns=["hyperparameters"])
# results = pd.concat([results, df_hyperparameters], axis=1)

results

# %%
performance_indicator = "F1"


base_out_folder = "out"


def create_plots(
    method: str,
    dataset: str,
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
        ax.set_ylabel(param_1)
        ax.set_xlabel(param_2)
        fig.savefig(os.path.join(plot_out_folder, f"heatmap.png"))
        # plt.close(fig)


# create_plots(
#     "mnf", "gjovik", ["window", "gamma"], performance_indicator, base_out_folder
# )

# create_plots(
#     "lila",
#     "gjovik",
#     [
#         "est_length",
#         "delta",
#         "C_threshold",
#     ],
#     performance_indicator,
#     base_out_folder,
# )

# create_plots(
#     "lila",
#     "battledim",
#     ["est_length", "C_threshold", "delta"],
#     performance_indicator,
#     base_out_folder,
# )
# create_plots(
#     "mnf",
#     "battledim",
#     ["window", "gamma"],
#     performance_indicator,
#     base_out_folder,
# )
create_plots(
    "mnf",
    "gjovik",
    ["window", "gamma"],
    performance_indicator,
    base_out_folder,
)
create_plots(
    "lila",
    "gjovik",
    ["est_length", "C_threshold", "delta"],
    performance_indicator,
    base_out_folder,
)
# create_plots(
#     "mnf", "battledim", ["window", "gamma"], performance_indicator, base_out_folder
# )


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
