# %%
import ast
import os
import numpy as np
import pandas as pd
from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from typing import Dict, List
from ldimbenchmark.evaluation_metrics import f1Score
from ldimbenchmark.methods import LILA, MNF, DUALMethod
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
import itertools
import logging

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(
    level=numeric_level,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


methods = [LILA(), MNF(), DUALMethod()]


out_dir = "out"

# get method meta data
method_meta = {}
for method in methods:
    method_meta[method.name] = method.metadata


frame = pd.DataFrame(method_meta).T
frame = frame.drop(columns=["mimum_dataset_size"])


frame.data_needed = frame.data_needed.astype(str)
normalized_data_needed = pd.json_normalize(frame.data_needed.apply(ast.literal_eval))
normalized_data_needed.index = frame.index

frame = pd.concat(
    [frame, normalized_data_needed],
    axis=1,
)
frame = frame.drop(columns=["data_needed"])

# frame["hyperparamters"] = frame["hyperparameters"].apply(
#     lambda x: ("\n".join(param.name for param in x))
# )


frame[["pressures", "flows", "levels", "model", "demands", "structure"]].style.format(
    escape="latex",
).set_table_styles(
    [
        # {'selector': 'toprule', 'props': ':hline;'},
        {"selector": "midrule", "props": ":hline;"},
        # {'selector': 'bottomrule', 'props': ':hline;'},
    ],
    overwrite=False,
).to_latex(
    os.path.join(out_dir, "methods_needed_data.tex"),
    label="table:methods_needed_data",
    caption="Data used by each method",
    position="H",
)

with pd.option_context("max_colwidth", 1000):
    for method in frame.index:
        method_params = pd.DataFrame(frame.loc[method]["hyperparameters"])
        method_params["name"] = method_params[0].apply(lambda x: x.name)
        method_params["type"] = method_params[0].apply(lambda x: x.type.__name__)
        method_params["default"] = method_params[0].apply(lambda x: x.default)
        method_params["description"] = method_params[0].apply(lambda x: x.description)
        method_params = method_params.drop(columns=[0])
        method_params.style.format(
            escape="latex",
        ).set_table_styles(
            [
                # {'selector': 'toprule', 'props': ':hline;'},
                {"selector": "midrule", "props": ":hline;"},
                # {'selector': 'bottomrule', 'props': ':hline;'},
            ],
            overwrite=False,
        ).hide(axis="index").to_latex(
            os.path.join(out_dir, f"methods_{method}_hyperparameters.tex"),
            label=f"table:methods_hyperparameters_{method}",
            caption=f"Hyperparameters of {method}",
            position="H",
        )
