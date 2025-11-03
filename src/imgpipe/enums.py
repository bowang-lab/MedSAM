from __future__ import annotations
from enum import Enum


class Structure(str, Enum):
    DISC = "disc"
    CUP = "cup"


class LabelType(str, Enum):
    GT = "gt"
    PRED = "pred"

class Eye(str, Enum):
    OS = "os"
    OD = "od"

class Ethnicity(str, Enum):
    CHINESE = "chinese"
    SPANISH = "spanish"