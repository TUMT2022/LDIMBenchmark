# %%
import pandas as pd
from sqlalchemy import create_engine
import os

results_db_path = os.path.join("grid-search", "evaluation_results", "results.db")
engine = create_engine(f"sqlite:///{results_db_path}")
results = pd.read_sql("results", engine, index_col="_folder")


# %%

results_lila = results[results["method"] == "lila"].sort_values("F1", ascending=False)

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

results_dualmethod = results[results["method"] == "dualmethod"].sort_values(
    "F1", ascending=False
)
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


# table_dualmethod[["hyperparameters"]].to_dict()
