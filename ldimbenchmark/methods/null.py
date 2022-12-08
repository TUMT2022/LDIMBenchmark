from ldimbenchmark import LDIMMethodBase, BenchmarkData, BenchmarkLeakageResult


class NullLeakagedDetectionMethod(LDIMMethodBase):
    """
    Null Algorithm
    """

    def __init__(self):
        super().__init__(
            name="Null",
            version="1.0",
            # hyperparameters={"est_length": "3 days", "C_threshold": 3, "delta": 4},
        )

    def train(self, train_data: BenchmarkData) -> None:
        return

    def detect(self, evaluation_data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        return []

    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        return None
