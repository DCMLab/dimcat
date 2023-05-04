from dataclasses import dataclass
from enum import Enum

from dimcat.features.base import Feature, FeatureConfig, FeatureName


class NotesFormat(str, Enum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    DEGREE = "DEGREE"
    INTERVAL = "INTERVAL"


@dataclass(frozen=True)
class NotesConfig(FeatureConfig):
    _configured_class = FeatureName.Notes
    format: NotesFormat = NotesFormat.NAME
    weight_grace_notes: float = 0.0


class Notes(Feature):
    _config_type = NotesConfig
