from __future__ import annotations

from enum import Enum
from typing import Type

from dimcat.base import DimcatObject
from dimcat.data.base import DimcatResource
from typing_extensions import Self


class FeatureName(str, Enum):
    Notes = "Notes"
    Annotations = "Annotations"
    KeyAnnotations = "KeyAnnotations"
    Metadata = "Metadata"

    def get_class(self) -> Type[Feature]:
        return DimcatObject._registry[self.name]

    @classmethod
    def _missing_(cls, value) -> Self:
        value_lower = value.lower()
        lc_values = {member.value.lower(): member for member in cls}
        if value_lower in lc_values:
            return lc_values[value_lower]
        for lc_value, member in lc_values.items():
            if lc_value.startswith(value_lower):
                return member
        raise ValueError(f"ValueError: {value_lower!r} is not a valid FeatureName.")


class Feature(DimcatResource):
    _enum_type = FeatureName


class Metadata(Feature):
    pass
