import logging
import re
from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, Optional, TypedDict, Dict, Union, List, Type
from datetime import datetime
from abc import ABC, abstractmethod


class BenchmarkLeakageResult(TypedDict):
    leak_pipe_id: Optional[str]
    leak_time_start: datetime
    leak_time_end: datetime
    leak_time_peak: datetime
    leak_area: float
    leak_diameter: float
    leak_max_flow: float
    description: Optional[str]
