import math
from time import sleep
from ldimbenchmark import LDIMMethodBase, BenchmarkData, BenchmarkLeakageResult
from ldimbenchmark.classes import MethodMetadata, MethodMetadataDataNeeded
from typing import List, Union


class TestComplexityLogNLeakagedDetectionMethod(LDIMMethodBase):
    """
    Test Complexity: linear
    """

    def __init__(self):
        super().__init__(
            name="Linear Complexity",
            version="1.0",
            metadata=MethodMetadata(
                data_needed=MethodMetadataDataNeeded(
                    pressures="ignored",
                    flows="ignored",
                    levels="ignored",
                    model="ignored",
                    demands="ignored",
                    structure="ignored",
                ),
                hyperparameters=[],
            )
            # hyperparameters={"est_length": "3 days", "C_threshold": 3, "delta": 4},
        )

    def train(self, train_data: BenchmarkData) -> None:
        sleep(len(train_data.demands[train_data.demands.keys()[0]]))
        return

    def detect_offline(
        self, evaluation_data: BenchmarkData
    ) -> List[BenchmarkLeakageResult]:
        sleep(len(evaluation_data.demands[evaluation_data.demands.keys()[0]]))
        return []

    def detect_online(self, evaluation_data) -> Union[BenchmarkLeakageResult, None]:
        return None
