from __future__ import annotations

from enum import Enum
from typing import Optional, Type

import frictionless as fl
from dimcat import get_class
from dimcat.resources.base import D, DimcatResource
from marshmallow import fields
from typing_extensions import Self


class FeatureName(str, Enum):
    Notes = "Notes"
    Annotations = "Annotations"
    KeyAnnotations = "KeyAnnotations"
    Metadata = "Metadata"

    def get_class(self) -> Type[Feature]:
        return get_class(self.name)

    @classmethod
    def _missing_(cls, value) -> Self:
        value_lower = value.lower()
        lc_values = {member.value.lower(): member for member in cls}
        if value_lower in lc_values:
            return lc_values[value_lower]
        for lc_value, member in lc_values.items():
            if lc_value.startswith(value_lower):
                return member
        raise ValueError(f"ValueError: {value!r} is not a valid FeatureName.")

    def __eq__(self, other) -> bool:
        if self.value == other:
            return True
        if isinstance(other, str):
            return other.lower() == self.value.lower()
        return False

    def __hash__(self):
        return hash(self.value)


class Feature(DimcatResource):
    _enum_type = FeatureName


class Metadata(Feature):
    pass


class NotesFormat(str, Enum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    DEGREE = "DEGREE"
    INTERVAL = "INTERVAL"


class Notes(Feature):
    class Schema(Feature.Schema):
        format = fields.Enum(NotesFormat)
        weight_grace_notes = fields.Float()

    def __init__(
        self,
        format: NotesFormat = NotesFormat.FIFTHS,
        weight_grace_notes: float = 0.0,
        df: Optional[D] = None,
        resource_name: Optional[str] = None,
        resource: Optional[fl.Resource | str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        self._format: NotesFormat = format
        self._weight_grace_notes: float = weight_grace_notes
        super().__init__(
            resource=resource,
            resource_name=resource_name,
            df=df,
            column_schema=column_schema,
            basepath=basepath,
            filepath=filepath,
            validate=validate,
        )

    @property
    def format(self) -> NotesFormat:
        return self._format

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes


class Annotations(Feature):
    pass


class KeyAnnotations(Annotations):
    pass
