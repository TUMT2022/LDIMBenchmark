from typing import Dict


def precision(results: Dict):
    # Positive-Predictive-Value (Precision)
    results["precision"] = results["true_positives"] / (
        results["true_positives"] + results["false_positives"]
    )
    return results


def recall(results: Dict):
    # True-Positive-Rate (Recall)
    results["recall (TPR)"] = results["true_positives"] / (
        results["true_positives"] + results["false_negatives"]
    )
    return results


def specifity(results: Dict):
    # True-Negative-Rate (Specificity)
    results["TNR"] = results["true_negatives"] / (
        results["true_negatives"] + results["false_positives"]
    )
    return results


def falsePositiveRate(results: Dict):
    # False-Positive-Rate (Fall-Out)
    results["FPR"] = results["false_positives"] / (
        results["true_negatives"] + results["false_positives"]
    )
    return results


def falseNegativeRate(results: Dict):
    # False-Negative-Rate (Miss-Rate)
    results["FNR"] = results["false_negatives"] / (
        results["true_positives"] + results["false_negatives"]
    )
    return results


def f1Score(results: Dict):
    results["F1"] = (2 * results["precision"] * results["recall (TPR)"]) / (
        results["precision"] + results["recall (TPR)"]
    )
    return results
