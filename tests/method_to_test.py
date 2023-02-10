from ldimbenchmark import (
    BenchmarkData,
    BenchmarkLeakageResult,
    FileBasedMethodRunner,
)
from ldimbenchmark.datasets import Dataset

from ldimbenchmark.classes import LDIMMethodBase
from typing import List
from ldimbenchmark.classes import MethodMetadata


class YourCustomLDIMMethod(LDIMMethodBase):
    def __init__(self, additional_output_path=""):
        super().__init__(
            name="YourCustomLDIMMethod",
            version="0.1",
            metadata=MethodMetadata(
                data_needed=["pressures", "demands", "flows", "levels"],
                hyperparameters=[],
            ),
        )

    def train(self, data: BenchmarkData):
        pass

    def detect_offline(self, data: BenchmarkData) -> List[BenchmarkLeakageResult]:
        return [
            {
                "leak_start": "2020-01-01",
                "leak_end": "2020-01-02",
                "leak_area": 0.2,
                "pipe_id": "test",
            }
        ]

    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        return {}


if __name__ == "__main__":
    runner = FileBasedMethodRunner(YourCustomLDIMMethod())
    runner.run()
