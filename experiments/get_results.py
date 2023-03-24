# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from sqlalchemy import create_engine
import os


from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.datasets.classes import Dataset


results_db_path = os.path.join("grid-search", "evaluation_results", "results.db")
engine = create_engine(f"sqlite:///{results_db_path}")
results = pd.read_sql("results", engine, index_col="_folder")
results

#############
## Battledim
############

# %%

results_lila = results[
    (results["method"] == "lila") & (results["dataset"] == "battledim")
].sort_values("F1", ascending=False)

# With Evaluation trained Parameters
t1 = results_lila[results["dataset_part"].isna()].iloc[0]
t1["result_set"] = "Parameters from Evaluation Data"

# With Training Data trained Parameters
t2_training = results_lila[results["dataset_part"] == "training"].iloc[0]
print(t2_training["hyperparameters"])

# Parameters do not match completely so we have to supply them manually
hyperparameters = "{'C_threshold': 15, 'delta': 12, 'est_length': 168}"
t2 = results_lila[
    results_lila["dataset_part"].isna()
    & (results_lila["hyperparameters"] == hyperparameters)
].iloc[0]
t2["result_set"] = "Parameters from Training Data"

table_lila = pd.concat([t1, t2], axis=1).T[
    [
        "method",
        "dataset",
        "dataset_part",
        "recall (TPR)",
        "false_positives",
        "F1",
        "hyperparameters",
        "result_set",
    ]
]
table_lila


# table_lila[["hyperparameters"]].to_dict()

# %%

results_dualmethod = results[
    (results["method"] == "dualmethod") & (results["dataset"] == "battledim")
].sort_values("F1", ascending=False)
results
# With Evaluation trained Parameters
t1 = results_dualmethod[results["dataset_part"].isna()].iloc[0]
t1["result_set"] = "Parameters from Evaluation Data"

# With Training Data trained Parameters
t2_training = results_dualmethod[results["dataset_part"] == "training"].iloc[0]
print(t2_training["hyperparameters"])

# Parameters do not match completely so we have to supply them manually
hyperparameters = "{'C_threshold': 0.8, 'delta': 3.0, 'est_length': 888.0}"
t2 = results_dualmethod[
    results_dualmethod["dataset_part"].isna()
    & (results_dualmethod["hyperparameters"] == hyperparameters)
].iloc[0]
t2["result_set"] = "Parameters from Training Data"

table_dualmethod = pd.concat([t1, t2], axis=1).T[
    [
        "method",
        "dataset",
        "dataset_part",
        "recall (TPR)",
        "false_positives",
        "F1",
        "hyperparameters",
        "result_set",
    ]
]
table_dualmethod

# %%

results_mnf = results[
    (results["method"] == "mnf") & (results["dataset"] == "battledim")
].sort_values("F1", ascending=False)
results
# With Evaluation trained Parameters
t1 = results_mnf[results["dataset_part"].isna()].iloc[0]
t1["result_set"] = "Parameters from Evaluation Data"

# With Training Data trained Parameters
t2_training = results_mnf[results["dataset_part"] == "training"].iloc[0]
print(t2_training["hyperparameters"])

# Parameters do not match completely so we have to supply them manually
hyperparameters = "{'window': 10.0, 'gamma': 1}"
t2 = results_mnf[
    results_mnf["dataset_part"].isna()
    & (results_mnf["hyperparameters"] == hyperparameters)
].iloc[0]
t2["result_set"] = "Parameters from Training Data"

table_mnf = pd.concat([t1, t2], axis=1).T[
    [
        "method",
        "dataset",
        "dataset_part",
        "recall (TPR)",
        "false_positives",
        "F1",
        "hyperparameters",
        "result_set",
    ]
]
table_mnf

# table_mnf[["hyperparameters"]].to_dict()

# %%
## Create Graphics


benchmark = LDIMBenchmark(
    hyperparameters={},
    datasets=[
        # Dataset("test_data/datasets/battledim"),
        Dataset("test_data/datasets/gjovik"),
    ],
    results_dir="./grid-search",
)

# benchmark.evaluate_run("dualmethod_0.1.0_battledim-789426bdd3a2afb33b904b293765c5fd_evaluation_a24648898fe78db373db626e00ba8970")

# Gjovik
# benchmark.evaluate_run("lila_0.2.0_gjovik-1aa41b32a8acd6d4a14058de02957c64_evaluation_6967a9c238a413245469c8016118ec71")
benchmark.evaluate_run("mnf_1.3_gjovik-1aa41b32a8acd6d4a14058de02957c64_evaluation_54dab5af2b95fb3bb8581beeaa7ec6bf", pd.Timedelta(days=4))

## For runs with best Parameters from Training Dataset
# %%
benchmark.evaluate_run("lila_0.1.20_battledim_fc810cb4fcac3b5edff336f83aa3c72b")

# %%

benchmark.evaluate_run("dualmethod_0.1.20_battledim_8a1287fd94b275fe42ef3fabcd87a78f")

# %%
benchmark.evaluate_run(
    "mnf_0.1.20_battledim-cf63c0e154c6c2355a362cb8b5905bc2_training_f9c89c6ad15d717a1f7c5d4b9328e8ed"
)

# %%

benchmark.evaluate_run("lila_0.1.20_gjovik_14f72cccdc431db4e2ffa1c3a0e50c92")

# %%
