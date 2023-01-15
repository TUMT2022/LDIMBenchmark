"""
Main Module
"""

from importlib import metadata

__version__ = metadata.version(__package__ or __name__)

from .benchmark import *
from .classes import *
