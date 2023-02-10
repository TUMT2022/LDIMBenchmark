from ldimbenchmark import LDIMMethodBase, BenchmarkData, BenchmarkLeakageResult
from ldimbenchmark.classes import (
    MethodMetadata,
    Hyperparameter,
    MethodMetadataDataNeeded,
)
from typing import List, Union
from random import random


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
                data_needed=MethodMetadataDataNeeded(
                    pressures="ignored",
                    flows="ignored",
                    levels="ignored",
                    model="ignored",
                    demands="ignored",
                    structure="ignored",
                ),
                hyperparameters=[
                    Hyperparameter(
                        name="random",
                        description="The Random percentage of detecting a leakage",
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

    def detect_offline(
        self, evaluation_data: BenchmarkData
    ) -> List[BenchmarkLeakageResult]:
        return []

    def detect_online(self, evaluation_data) -> Union[BenchmarkLeakageResult, None]:
        # TODO: Update keys to conform to new schema
        if random() < 0.5:
            return BenchmarkLeakageResult(
                leak_pipe_id="Any",
                leak_time_start=evaluation_data.pressures.index[0],
                leak_time_end=evaluation_data.pressures.index[0],
                leak_time_peak=evaluation_data.pressures.index[0],
                leak_area=0.0,
                leak_diameter=0.0,
                leak_max_flow=0.0,
            )
        else:
            return None
