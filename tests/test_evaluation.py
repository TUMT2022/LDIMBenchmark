# from evaluation import evaluate_leakages
# from algorithms.classes import BenchmarkAlgorithm, BenchmarkData, BenchmarkLeakageResult
# from datetime import datetime


# def test_evaluate_leakages_tn():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             }
#         ],
#     ) == {
#         "true_positives": 1,
#         "false_positives": 0,
#         "true_negatives": None,
#         "false_negatives": 1,
#         "time_to_detection": 0.0,
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_evaluate_leakages_fp():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             }
#         ],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#     ) == {
#         "true_positives": 1,
#         "false_positives": 1,
#         "true_negatives": None,
#         "false_negatives": 0,
#         "time_to_detection": 0.0,
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_leak_matching_more_detected():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-05",
#                 "leak_start": datetime.fromisoformat("2022-01-16 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-04",
#                 "leak_start": datetime.fromisoformat("2022-01-15 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-17 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-05",
#                 "leak_start": datetime.fromisoformat("2022-01-16 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#     ) == {
#         "true_positives": 2,
#         "false_positives": 1,
#         "true_negatives": None,
#         "false_negatives": 0,
#         "time_to_detection": 300.0,
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_leak_matching_more_expected():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-04",
#                 "leak_start": datetime.fromisoformat("2022-01-15 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-01-17 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-01-15 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-05",
#                 "leak_start": datetime.fromisoformat("2022-01-16 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-05",
#                 "leak_start": datetime.fromisoformat("2022-01-16 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#     ) == {
#         "true_positives": 2,
#         "false_positives": 0,
#         "true_negatives": None,
#         "false_negatives": 1,
#         "time_to_detection": 300.0,
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_leak_matching_all_detected_earlier_than_expected():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-04",
#                 "leak_start": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-17 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-01-01 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-05",
#                 "leak_start": datetime.fromisoformat("2022-01-16 00:05:00"),
#                 "leak_end": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-02-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#     ) == {
#         "true_positives": 0,
#         "false_positives": 2,
#         "true_negatives": None,
#         "false_negatives": 2,
#         "time_to_detection": "",
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_empty_detected():
#     assert evaluate_leakages(
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-04",
#                 "leak_start": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-17 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#         [],
#     ) == {
#         "true_positives": 0,
#         "false_positives": 0,
#         "true_negatives": None,
#         "false_negatives": 2,
#         "time_to_detection": "",
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_empty_expected():
#     assert evaluate_leakages(
#         [],
#         [
#             {
#                 "pipe_id": "P-03",
#                 "leak_start": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-01 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#             {
#                 "pipe_id": "P-04",
#                 "leak_start": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_end": datetime.fromisoformat("2022-03-17 00:00:00"),
#                 "leak_peak": datetime.fromisoformat("2022-03-15 00:00:00"),
#                 "leak_area": 0.005,
#                 "leak_diameter": 0.005,
#             },
#         ],
#     ) == {
#         "true_positives": 0,
#         "false_positives": 2,
#         "true_negatives": None,
#         "false_negatives": 0,
#         "time_to_detection": "",
#         "wrong_pipe": 0,
#         "score": 0,
#     }


# def test_empty_both():
#     assert evaluate_leakages([], []) == {
#         "true_positives": 0,
#         "false_positives": 0,
#         "true_negatives": None,
#         "false_negatives": 0,
#         "time_to_detection": "",
#         "wrong_pipe": 0,
#         "score": 0,
#     }
