from __future__ import annotations
from enum import Enum


class Structure(str, Enum):
    DISC = "disc"
    CUP = "cup"


class LabelType(str, Enum):
    GT = "gt"
    PRED = "pred"