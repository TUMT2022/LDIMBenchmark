from ast import Dict
import logging
import os
import numpy as np

import pandas as pd
from ldimbenchmark.benchmark_evaluation import evaluate_leakages
from ldimbenchmark.classes import BenchmarkLeakageResult


def load_result(folder: str, try_load_docker_stats=False) -> Dict:
    folder = os.path.join(folder, "")
    index = os.path.basename(os.path.dirname(folder))

    detected_leaks_file = os.path.join(folder, "detected_leaks.csv")
    if not os.path.exists(detected_leaks_file):
        logging.warning(f"No detected_leaks.csv found in {folder}")
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
    matched_frame = pd.DataFrame(matched_list, columns=[0, 1])
    detected_leaks_frame = pd.DataFrame(
        pd.json_normalize(matched_frame[1]).add_prefix("detected."),
        columns=[
            "expected." + key
            for key in list(BenchmarkLeakageResult.__annotations__.keys())
        ]
        + [
            "detected." + key
            for key in list(BenchmarkLeakageResult.__annotations__.keys())
        ],
    )
    detected_leaks_frame["result_id"] = index

    evaluation_results["method"] = run_info["method"]
    evaluation_results["method_version"] = run_info.get("method_version", None)
    evaluation_results["dataset"] = run_info["dataset"]
    evaluation_results["dataset_part"] = run_info.get("dataset_part", None)
    evaluation_results["dataset_id"] = run_info["dataset_id"]
    evaluation_results["dataset_derivations"] = run_info["dataset_options"]
    evaluation_results["hyperparameters"] = run_info["hyperparameters"]
    evaluation_results["matched_leaks_list"] = matched_list
    evaluation_results["detected_leaks_frame"] = detected_leaks_frame

    evaluation_results["_folder"] = index
    evaluation_results["executed_at"] = run_info.get("executed_at", None)
    evaluation_results["train_time"] = run_info["train_time"]
    evaluation_results["detect_time"] = run_info["detect_time"]
    evaluation_results["time_initializing"] = run_info["time_initializing"]
    evaluation_results["total_time"] = run_info["total_time"]
    evaluation_results["method_time"] = (
        evaluation_results["train_time"] + evaluation_results["detect_time"]
    )

    if try_load_docker_stats:
        stats_file = os.path.join(folder, "stats.csv")
        if os.path.exists(stats_file):
            stats = pd.read_csv(stats_file)

            # Convert string columns to Dictionary columns
            stats["pids_stats"] = stats["pids_stats"].apply(lambda x: eval(x))
            stats["blkio_stats"] = stats["blkio_stats"].apply(lambda x: eval(x))
            stats["cpu_stats"] = stats["cpu_stats"].apply(lambda x: eval(x))
            stats["precpu_stats"] = stats["precpu_stats"].apply(lambda x: eval(x))
            stats["memory_stats"] = stats["memory_stats"].apply(lambda x: eval(x))
            # stats["networks"] = stats["networks"].apply(lambda x: eval(x))

            flat_stats = pd.json_normalize(stats.to_dict(orient="records"))

            # According to https://github.com/docker/cli/blob/e57b5f78de635e6e2b688686d10b830c4747c4dc/cli/command/container/stats_helpers.go#L239
            if ("memory_stats.stats.inactive_file" in flat_stats.columns) and (
                flat_stats["memory_stats.stats.inactive_file"]
                .gt(flat_stats["memory_stats.usage"])
                .all()
            ):
                memory = (
                    flat_stats["memory_stats.usage"]
                    - flat_stats["memory_stats.stats.inactive_file"]
                )
            else:
                memory = flat_stats["memory_stats.usage"]
            evaluation_results["memory_avg"] = memory.mean()
            evaluation_results["memory_max"] = memory.max()

    return evaluation_results
