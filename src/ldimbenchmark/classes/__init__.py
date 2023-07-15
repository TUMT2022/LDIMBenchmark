"""
Main Module
"""

from importlib import metadata

__version__ = metadata.version(__package__ or __name__)

from .BenchmarkData import *
from .BenchmarkLeakageResult import *
from .LDIMMethodBase import *
from .MethodMetadata import *
