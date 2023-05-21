from ast import Dict
import os

import pandas as pd
from ldimbenchmark.benchmark_evaluation import evaluate_leakages


def load_result(folder: str, try_load_docker_stats=False) -> Dict:
    folder = os.path.join(folder, "")
    detected_leaks_file = os.path.join(folder, "detected_leaks.csv")
    if not os.path.exists(detected_leaks_file):
        return {}

    detected_leaks = pd.read_csv(
        detected_leaks_file,
        parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, utc=True),
    )

    evaluation_dataset_leakages = pd.read_csv(
        os.path.join(folder, "should_have_detected_leaks.csv"),
        parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, utc=True),
    )

    run_info = pd.read_csv(os.path.join(folder, "run_info.csv")).iloc[0]

    # TODO: Ignore Detections outside of the evaluation period
    (evaluation_results, matched_list) = evaluate_leakages(
        evaluation_dataset_leakages, detected_leaks
    )
    evaluation_results["method"] = run_info["method"]
    evaluation_results["method_version"] = run_info.get("method_version", None)
    evaluation_results["dataset"] = run_info["dataset"]
    evaluation_results["dataset_part"] = run_info.get("dataset_part", None)
    evaluation_results["dataset_id"] = run_info["dataset_id"]
    evaluation_results["dataset_derivations"] = run_info["dataset_options"]
    evaluation_results["hyperparameters"] = run_info["hyperparameters"]
    evaluation_results["matched_leaks_list"] = matched_list
    evaluation_results["_folder"] = os.path.basename(os.path.dirname(folder))
    evaluation_results["executed_at"] = run_info.get("executed_at", None)
    evaluation_results["train_time"] = run_info["train_time"]
    evaluation_results["detect_time"] = run_info["detect_time"]
    evaluation_results["time_initializing"] = run_info["time_initializing"]
    evaluation_results["total_time"] = run_info["total_time"]
    evaluation_results["method_time"] = (
        evaluation_results["train_time"] + evaluation_results["detect_time"]
    )

    if try_load_docker_stats:
        stats = pd.read_csv(os.path.join(folder, "stats.csv"))

        # Convert string columns to Dictionary columns
        stats["pids_stats"] = stats["pids_stats"].apply(lambda x: eval(x))
        stats["blkio_stats"] = stats["blkio_stats"].apply(lambda x: eval(x))
        stats["cpu_stats"] = stats["cpu_stats"].apply(lambda x: eval(x))
        stats["precpu_stats"] = stats["precpu_stats"].apply(lambda x: eval(x))
        stats["memory_stats"] = stats["memory_stats"].apply(lambda x: eval(x))
        # stats["networks"] = stats["networks"].apply(lambda x: eval(x))

        flat_stats = pd.json_normalize(stats.to_dict(orient="records"))

        evaluation_results["memory_avg"] = flat_stats["memory_stats.usage"].mean()
        evaluation_results["memory_max"] = flat_stats["memory_stats.usage"].max()

    return evaluation_results
