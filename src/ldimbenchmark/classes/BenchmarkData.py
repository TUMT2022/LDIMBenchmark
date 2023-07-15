import logging
import re
from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, Optional, TypedDict, Dict, Union, List, Type
from datetime import datetime
from abc import ABC, abstractmethod

class BenchmarkData:
    """
    Representation of the File Based Benchmark Dataset
    """

    def __init__(
        self,
        pressures: Dict[str, DataFrame],
        demands: Dict[str, DataFrame],
        flows: Dict[str, DataFrame],
        levels: Dict[str, DataFrame],
        model: WaterNetworkModel,
        dmas: Dict[str, List[str]],
    ):
        """
        Initialize the BenchmarkData object.
        """
        self.pressures = pressures
        """Pressures of the System."""
        self.demands = demands
        """Demands of the System."""
        self.flows = flows
        """Flows of the System."""
        self.levels = levels
        """Levels of the System."""
        self.model = model
        """Model of the System (INP)."""
        self.dmas = dmas
        """
        District Metered Areas
        Dictionary with names of the areas as key and list of WN nodes as value.
        """
        self.metadata = {}
        """Metadata of the System. e.g. Metering zones and included sensors."""
