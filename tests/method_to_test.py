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
    def get_results():
        return [
            BenchmarkLeakageResult(
                leak_pipe_id="test",
                leak_time_start="2020-01-01",
                leak_time_end="2020-01-02",
                leak_time_peak="2020-01-01",
                leak_area=0.2,
                leak_diameter=0.1,
                leak_max_flow=1,
            )
        ]

    def __init__(self, additional_output_path=""):
        super().__init__(
            name="YourCustomLDIMMethod",
            version="0.1",
            metadata=MethodMetadata(
                data_needed=["pressures", "demands", "flows", "levels"],
                hyperparameters=[],
            ),
        )

    def prepare(self, data: BenchmarkData = None) -> None:
        pass

    def detect_offline(self, data: BenchmarkData) -> List[BenchmarkLeakageResult]:
        return YourCustomLDIMMethod.get_results()

    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        return {}


if __name__ == "__main__":
    runner = FileBasedMethodRunner(YourCustomLDIMMethod())
    runner.run()
