from ldimbenchmark import LDIMMethodBase, BenchmarkData, BenchmarkLeakageResult
from ldimbenchmark.classes import MethodMetadata, Hyperparameter
from typing import List
import random


class RandomMethod(LDIMMethodBase):
    """
    RandomMethod
    """

    def __init__(self):
        # Provide information about your method in the super call
        super().__init__(
            name="RandomMethod",
            version="1.0",
            metadata=MethodMetadata(
                data_needed=["pressures", "demands", "flows", "levels"],
                hyperparameters=[
                    Hyperparameter(
                        name="random",
                        description="The Random percentage of detecing a leakage",
                        default=0.5,
                        max=1.0,
                        min=0.0,
                        type=float,
                    ),
                ],
            ),
        )

    def train(self, train_data: BenchmarkData) -> None:
        return

    def detect(self, evaluation_data: BenchmarkData) -> List[BenchmarkLeakageResult]:
        return []

    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        # TODO: Update keys to conform to new schema
        return (
            {
                "pipe_id": "Any",
                "leak_start": evaluation_data.pressures.index[0],
                "leak_end": evaluation_data.pressures.index[0],
                "leak_peak": evaluation_data.pressures.index[0],
                "leak_area": 0.0,
                "leak_diameter": 0.0,
            }
            if random() < 0.5
            else None
        )
