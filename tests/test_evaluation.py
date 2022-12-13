from ldimbenchmark.benchmark_evaluation import evaluate_leakages
from ldimbenchmark.classes import BenchmarkLeakageResult
from datetime import datetime
import pandas as pd


def test_evaluate_leakages_tn():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                # BenchmarkLeakageResult(
                #     leak_leak_pipe_id: str
                #     leak_time_start: datetime
                #     leak_time_end: datetime
                #     leak_time_peak: datetime
                #     leak_area: float
                #     leak_diameter: float
                #     leak_max_flow: float
                # )
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                }
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 1,
        "false_positives": 0,
        "true_negatives": None,
        "false_negatives": 1,
        "time_to_detection": 0.0,
        "wrong_pipe": 0,
        "score": 0,
    }


def test_evaluate_leakages_fp():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 1,
        "false_positives": 1,
        "true_negatives": None,
        "false_negatives": 0,
        "time_to_detection": 0.0,
        "wrong_pipe": 0,
        "score": 0,
    }


def test_leak_matching_more_detected():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-01-16 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-04",
                    "leak_time_start": datetime.fromisoformat("2022-01-15 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-01-16 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 2,
        "false_positives": 1,
        "true_negatives": None,
        "false_negatives": 0,
        "time_to_detection": 300.0,
        "wrong_pipe": 0,
        "score": 0,
    }


def test_leak_matching_more_expected():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-04",
                    "leak_time_start": datetime.fromisoformat("2022-01-15 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-01-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-01-16 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-01-16 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 2,
        "false_positives": 0,
        "true_negatives": None,
        "false_negatives": 1,
        "time_to_detection": 300.0,
        "wrong_pipe": 0,
        "score": 0,
    }


def test_leak_matching_all_detected_earlier_than_expected():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-04",
                    "leak_time_start": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-01-01 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-01-16 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 0,
        "false_positives": 2,
        "true_negatives": None,
        "false_negatives": 2,
        "time_to_detection": "",
        "wrong_pipe": 0,
        "score": 0,
    }


def test_empty_detected():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-04",
                    "leak_time_start": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
        pd.DataFrame([], columns=list(BenchmarkLeakageResult.__annotations__.keys())),
    )
    assert evaluation_results == {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": None,
        "false_negatives": 2,
        "time_to_detection": "",
        "wrong_pipe": 0,
        "score": 0,
    }


def test_empty_expected():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame([], columns=list(BenchmarkLeakageResult.__annotations__.keys())),
        pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-04",
                    "leak_time_start": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ),
    )
    assert evaluation_results == {
        "true_positives": 0,
        "false_positives": 2,
        "true_negatives": None,
        "false_negatives": 0,
        "time_to_detection": "",
        "wrong_pipe": 0,
        "score": 0,
    }


def test_empty_both():
    evaluation_results, matched_list = evaluate_leakages(
        pd.DataFrame([], columns=list(BenchmarkLeakageResult.__annotations__.keys())),
        pd.DataFrame([], columns=list(BenchmarkLeakageResult.__annotations__.keys())),
    )
    assert evaluation_results == {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": None,
        "false_negatives": 0,
        "time_to_detection": "",
        "wrong_pipe": 0,
        "score": 0,
    }
