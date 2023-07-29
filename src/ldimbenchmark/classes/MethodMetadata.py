import logging
import re
from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, Optional, TypedDict, Dict, Union, List, Type
from datetime import datetime
from abc import ABC, abstractmethod


class Hyperparameter:
    """
    Definition of a Hyperparameter for a Leakage Detection Method
    """

    name: str
    type: type
    default: Union[str, int, float, bool]
    description: str
    min: Union[int, float]
    max: Union[int, float]
    options: List[Union[str, int, float]]

    def __init__(
        self,
        name: str,
        description: str,
        value_type: Type,
        required: bool = False,
        default: Union[int, float, bool] = None,
        options: Optional[List[str]] = None,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ):
        """
        ctor.
        """

        self.name = name

        # Warn in name is not lowercase
        # TODO: Rename all hyperparameters to lowercase
        # if self.name != self.name.lower():
        #     logging.warning(
        #         f"Hyperparameter name '{self.name}' is not lowercase. This is not recommended."
        #     )

        self.description = description

        # Validation
        self.type = value_type
        if not (value_type == type(default) or type(None) == type(default)):
            raise ValueError(
                f"Parameter 'default' must be of type {value_type}, but is of type {type(default)}."
            )

        if isinstance(value_type, bool):
            if options is not None and (min is not None or max is not None):
                raise ValueError(
                    f"Parameter 'options' and 'min/max cannot be set if using type 'bool'."
                )

        # if isinstance(value_type, str):
        #     if options is None:
        #         raise ValueError(
        #             f"Parameter 'options' must be set if using type 'str'."
        #         )

        # if isinstance(value_type, int) or isinstance(value_type, float):
        #     if options is None and (min is None or max is None):
        #         raise ValueError(
        #             f"Parameter 'options' or 'min/max' must be set if using type 'int/float'."
        #         )

        if options is not None and (min is not None or max is not None):
            raise ValueError(
                f"Parameters 'options' and 'min/max' cannot be supplied at the same time."
            )
        self.required = required
        self.default = default
        self.options = options
        self.min = min
        self.max = max

    # def __str__(self):
    #     return f"{self.name}: {self.value}"

    # def __repr__(self):
    #     return f"{self.name}: {self.value}"

    # def __eq__(self, other):
    #     return self.name == other.name and self.value == other.value

    # def __hash__(self):
    #     return


class MethodMetadataDataNeeded(TypedDict):
    """
    Describing the necessity of the data for the method.

    necessary - The method needs the data to work, otherwise it would fail.
    optional - The data is not necessary for the method, but its presence would enhance it.
    ignored - The data is not necessary for the method and its presence would not enhance it (simply put it is ignored).

    Depending on what is set for the type of data the

    |Selected Need|Provided by dataset|Result     | Data supplied |
    |:------------|:------------------|-----------|---------------|
    |`necessary`  |yes                |Benchmarked|Yes            |
    |`necessary`  |no                 |Skipped    |No             |
    |`optional`   |yes                |Benchmarked|Yes            |
    |`optional`   |no                 |Benchmarked|No             |
    |`ignored`    |yes                |Benchmarked|No             |
    |`ignored`    |no                 |Benchmarked|No             |
    """

    pressures: Literal["necessary", "optional", "ignored"]
    demands: Literal["necessary", "optional", "ignored"]
    flows: Literal["necessary", "optional", "ignored"]
    levels: Literal["necessary", "optional", "ignored"]
    model: Literal["necessary", "optional", "ignored"]
    structure: Literal["necessary", "optional", "ignored"]


class MethodMetadata(TypedDict):
    data_needed: MethodMetadataDataNeeded
    hyperparameters: List[Hyperparameter]
    capability: Literal["asses", "detect", "identify", "localize", "control"]
    paradigm: Literal["online", "offline"]
