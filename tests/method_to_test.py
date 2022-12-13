from ldimbenchmark import (
    BenchmarkData,
    BenchmarkLeakageResult,
    FileBasedMethodRunner,
)
from ldimbenchmark.datasets import Dataset

from ldimbenchmark.classes import LDIMMethodBase


class YourCustomLDIMMethod(LDIMMethodBase):
    def __init__(self, additional_output_path="", hyperparameters={}):
        super().__init__(
            name="YourCustomLDIMMethod",
            version="0.1",
            additional_output_path=additional_output_path,
            hyperparameters=hyperparameters,
        )

    def train(self, data: BenchmarkData):
        pass

    def detect(self, data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        return [
            {
                "leak_start": "2020-01-01",
                "leak_end": "2020-01-02",
                "leak_area": 0.2,
                "pipe_id": "test",
            }
        ]

    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        return {}


if __name__ == "__main__":
    runner = FileBasedMethodRunner(YourCustomLDIMMethod())
    runner.run()
