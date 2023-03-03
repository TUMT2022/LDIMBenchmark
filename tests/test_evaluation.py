import unittest
from ldimbenchmark.benchmark_evaluation import evaluate_leakages
from ldimbenchmark.classes import BenchmarkLeakageResult
from datetime import datetime
import pandas as pd
from pandas.testing import assert_frame_equal


class MyTestCase(unittest.TestCase):
    def test_evaluate_leakages_tn(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
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
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
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
        }

    def test_evaluate_leakages_fp(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
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
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
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
        }

    def test_leak_matching_more_detected(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-05",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-16 00:00:00"
                        ),
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
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:05:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-04",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-15 00:05:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-17 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-05",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-16 00:05:00"
                        ),
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
            "time_to_detection": 600.0,
            "wrong_pipe": 0,
        }

    def test_leak_matching_more_expected(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-04",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-15 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-17 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-05",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-16 00:00:00"
                        ),
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
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:05:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-05",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-16 00:05:00"
                        ),
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
            "time_to_detection": 600.0,
            "wrong_pipe": 0,
        }

    def test_leak_matching_all_detected_earlier_than_expected(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-04",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-15 00:00:00"
                        ),
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
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:05:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-05",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-16 00:05:00"
                        ),
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
            "time_to_detection": None,
            "wrong_pipe": 0,
        }

    def test_leak_matching_detected_leak_outside_expected(self):
        expected_leaks = pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-03-01 00:00:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-01 23:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-01 10:00:00"),
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
        ).astype(
            {
                "leak_time_start": "datetime64[ns, UTC]",
                "leak_time_end": "datetime64[ns, UTC]",
            }
        )
        expected_leaks["type"] = "expected"

        detected_leaks = pd.DataFrame(
            [
                {
                    "leak_pipe_id": "P-03",
                    "leak_time_start": datetime.fromisoformat("2022-03-02 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-02 23:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-02 10:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
                {
                    "leak_pipe_id": "P-05",
                    "leak_time_start": datetime.fromisoformat("2022-03-16 00:05:00"),
                    "leak_time_end": datetime.fromisoformat("2022-03-17 00:00:00"),
                    "leak_time_peak": datetime.fromisoformat("2022-03-16 05:00:00"),
                    "leak_area": 0.005,
                    "leak_diameter": 0.005,
                },
            ]
        ).astype(
            {
                "leak_time_start": "datetime64[ns, UTC]",
                "leak_time_end": "datetime64[ns, UTC]",
            }
        )
        detected_leaks["type"] = "detected"

        evaluation_results, matched_list = evaluate_leakages(
            expected_leaks=expected_leaks, detected_leaks=detected_leaks
        )

        expected_result = [
            (
                expected_leaks.loc[0],
                None,
            ),
            (
                None,
                detected_leaks.loc[0],
            ),
            (
                expected_leaks.loc[1],
                detected_leaks.loc[1],
            ),
        ]
        assert_frame_equal(pd.DataFrame(matched_list), pd.DataFrame(expected_result))
        assert evaluation_results == {
            "true_positives": 1,
            "false_positives": 1,
            "true_negatives": None,
            "false_negatives": 1,
            "time_to_detection": 86700.0,
            "wrong_pipe": 1,
        }

    def test_evaluate_leakages_time_to_detection(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-02 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-02 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-02-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-02-02 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-02-02 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-01-01 00:10:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-01-02 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-01-02 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-02-01 00:10:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-02-02 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-02-02 00:00:00"),
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
            "false_negatives": 0,
            "time_to_detection": 1200,
            "wrong_pipe": 0,
        }

    def test_empty_detected(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-04",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-15 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-17 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                ]
            ),
            pd.DataFrame(
                [], columns=list(BenchmarkLeakageResult.__annotations__.keys())
            ),
        )
        assert evaluation_results == {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": None,
            "false_negatives": 2,
            "time_to_detection": None,
            "wrong_pipe": 0,
        }

    def test_empty_expected(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [], columns=list(BenchmarkLeakageResult.__annotations__.keys())
            ),
            pd.DataFrame(
                [
                    {
                        "leak_pipe_id": "P-03",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-01 00:00:00"
                        ),
                        "leak_time_end": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_time_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
                        "leak_area": 0.005,
                        "leak_diameter": 0.005,
                    },
                    {
                        "leak_pipe_id": "P-04",
                        "leak_time_start": datetime.fromisoformat(
                            "2022-03-15 00:00:00"
                        ),
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
            "time_to_detection": None,
            "wrong_pipe": 0,
        }

    def test_empty_both(self):
        evaluation_results, matched_list = evaluate_leakages(
            pd.DataFrame(
                [], columns=list(BenchmarkLeakageResult.__annotations__.keys())
            ),
            pd.DataFrame(
                [], columns=list(BenchmarkLeakageResult.__annotations__.keys())
            ),
        )
        assert evaluation_results == {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": None,
            "false_negatives": 0,
            "time_to_detection": None,
            "wrong_pipe": 0,
        }
