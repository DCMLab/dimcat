from enum import Enum

from dimcat.features.base import Feature
from marshmallow import fields


class NotesFormat(str, Enum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    DEGREE = "DEGREE"
    INTERVAL = "INTERVAL"


class Notes(Feature):
    class Schema:
        format = fields.Enum(NotesFormat)
        weight_grace_notes = fields.Float(default=0.0)
