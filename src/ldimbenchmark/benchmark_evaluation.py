"""
Evaluation Methods
"""
from datetime import datetime, timedelta
import itertools

from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np
import pandas as pd


def evaluate_leakages(expected_leaks: pd.DataFrame, detected_leaks: pd.DataFrame):
    """
    Evaluates the detected leaks and returns the results.
    """
    # Map Leakages Start and End Time to Coordinate System

    # Minimize the error between the expected and detected leakages to find the best match
    # Sort list by start time

    # expected_leaks = parse_obj_as(List[BenchmarkLeakageResult], expected_leaks)
    # detected_leaks = parse_obj_as(List[BenchmarkLeakageResult], detected_leaks_unparsed)

    # columns = list(BenchmarkLeakageResult.__fields__.keys())

    right_leak_detected = 0
    non_existing_leak_detected = 0
    existing_leak_not_detected = 0
    time_to_detection = ""
    wrong_pipe_detected = 0

    # Make sure dates are the same

    expected_leaks = expected_leaks.astype(
        {
            "leak_time_start": "datetime64[ns, UTC]",
            "leak_time_end": "datetime64[ns, UTC]",
        }
    )
    detected_leaks = detected_leaks.astype(
        {
            "leak_time_start": "datetime64[ns, UTC]",
            "leak_time_end": "datetime64[ns, UTC]",
        }
    )

    # Match detected Leaks with existing Leaks
    expected_leaks_times = expected_leaks
    expected_leaks_times["type"] = "expected"
    detected_leaks_times = detected_leaks
    detected_leaks_times["type"] = "detected"

    list_of_all = pd.concat([expected_leaks_times, detected_leaks_times])
    list_of_all = list_of_all.sort_values(by="leak_time_start")
    list_of_all["used"] = 0
    list_of_all["index"] = list_of_all.index
    list_of_all = list_of_all.reset_index()
    # print(list_of_all)
    [D1, D2] = np.meshgrid(
        expected_leaks_times["leak_time_start"], detected_leaks_times["leak_time_start"]
    )
    # Get Distance between expected and detected Leaks (with positive values if the detected leak is after the expected leak)
    dist_mat = pd.DataFrame(D2 - D1)
    dist_mat[dist_mat < np.timedelta64(0)] = np.timedelta64("NaT")
    # print(dist_mat)

    matched_list = []
    # TODO: Make sure that the table index (int) is used and not the pipe_id
    for index, leak in list_of_all.iterrows():
        if list_of_all["used"][index] == 1:
            continue
        source_array_index = list_of_all["index"][index]

        if leak["type"] == "expected":
            # If the type is expected try to find the closest leak from the detected array
            timespan_to_other_leaks = dist_mat.iloc[:, source_array_index]
            # print(timespan_to_other_leaks)
            # If all detected leaks ar before the expected Leak there is None to match it to.
            if all(np.isnat(i.to_numpy()) for i in timespan_to_other_leaks):
                matched_list.append(
                    (expected_leaks.loc[source_array_index].to_dict(), None)
                )
                continue

            min_x = timespan_to_other_leaks.idxmin(skipna=True)
            ref = list_of_all[list_of_all["type"] == "detected"]
            ref = ref[ref["index"] == min_x]
            index_new = ref.index

            if dist_mat.iloc[min_x, :].idxmin(skipna=True) == source_array_index and (
                # Leak also has to be within the expected leak time
                expected_leaks.loc[source_array_index].leak_time_end
                >= detected_leaks.loc[min_x].leak_time_start
            ):
                # if (ref["used"].values[0] == 0):
                list_of_all.loc[index_new, "used"] = 1
                # list_of_all.iloc[index_new]["used"] = 1
                matched_list.append(
                    (
                        expected_leaks.loc[source_array_index].to_dict(),
                        detected_leaks.loc[min_x].to_dict(),
                    )
                )
            else:
                matched_list.append(
                    (expected_leaks.loc[source_array_index].to_dict(), None)
                )
            # print(ref)
        else:
            matched_list.append(
                (None, detected_leaks.loc[source_array_index].to_dict())
            )

        # print(list_of_all)
        # for detected_leak, expected_leak in matched_list:
        #     print(detected_leak.pipe_id,
        #           expected_leak.pipe_id if expected_leak != None else None)

        # matched_list.append((expected_leaks[expected_index], leak))
    # for detected_leak, expected_leak in matched_list:
    #     print(detected_leak.pipe_id if detected_leak != None else None,
    #           expected_leak.pipe_id if expected_leak != None else None)

    # print("###########")
    # for detected_leak, expected_leak in itertools.zip_longest(sorted_detected_leaks, sorted_expected_leaks):

    time_to_detection = []
    for expected_leak, detected_leak in matched_list:
        if detected_leak is None:
            existing_leak_not_detected += 1
            continue
        if expected_leak is None:
            non_existing_leak_detected += 1
            continue
        # print(detected_leak.pipe_id, expected_leak.pipe_id)
        if (
            detected_leak["leak_time_start"] >= expected_leak["leak_time_start"]
            and detected_leak["leak_time_end"] <= expected_leak["leak_time_end"]
        ):
            right_leak_detected += 1
            # Calculate TimeSpan Between Detections
            time_to_detection.append(
                (
                    detected_leak["leak_time_start"] - expected_leak["leak_time_start"]
                ).total_seconds()
            )
            if expected_leak["leak_pipe_id"] == detected_leak["leak_pipe_id"]:
                pass
            else:
                wrong_pipe_detected += 1
        else:
            existing_leak_not_detected += 1

        # TODO Refactor and only give back the matched list
        # TODO make another function which takes a list of evaluators and executes them to get an evaluation table
    return (
        {
            "true_positives": right_leak_detected,
            "false_positives": non_existing_leak_detected,
            "true_negatives": None,  # Not applicable, we dont have information about non-leaks
            "false_negatives": existing_leak_not_detected,
            "time_to_detection": None
            if len(time_to_detection) == 0
            else np.sum(time_to_detection),
            "wrong_pipe": wrong_pipe_detected,
        },
        matched_list,
    )
