from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Type

from dimcat.base import Configuration, DimcatObject, WrappedDataframe
from typing_extensions import Self


class FeatureName(str, Enum):
    Notes = "Notes"
    Annotations = "Annotations"
    KeyAnnotations = "KeyAnnotations"
    Metadata = "Metadata"

    def get_class(self) -> Type[Feature]:
        return DimcatObject._registry[self.name]

    def get_config(self, **kwargs) -> FeatureConfig:
        config_type = self.get_class()._config_type
        return config_type.from_dict(kwargs)

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


@dataclass(frozen=True)
class FeatureConfig(Configuration):
    _configured_class = "Feature"


class Feature(WrappedDataframe):
    _config_type = FeatureConfig
    _enum_type = FeatureName


@dataclass(frozen=True)
class MetadataConfig(FeatureConfig):
    _configured_class = "Metadata"


class Metadata(Feature):
    _config_type = MetadataConfig
