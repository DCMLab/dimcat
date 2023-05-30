from __future__ import annotations

from enum import Enum
from typing import Optional, Type

import frictionless as fl
import marshmallow as mm
from dimcat import get_class
from dimcat.resources.base import DimcatResource
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
        format = mm.fields.Enum(NotesFormat)
        merge_ties = mm.fields.Boolean(
            load_default=True,
            metadata=dict(
                title="Merge tied notes",
                description="If set, notes that are tied together in the score are merged together, counting them "
                "as a single event of the corresponding length. Otherwise, every note head is counted.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = True,
        weight_grace_notes: float = 0.0,
        resource_name: Optional[str] = None,
        resource: Optional[fl.Resource | str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = True,
    ) -> None:
        self._format: NotesFormat = format
        self._weight_grace_notes: float = weight_grace_notes
        super().__init__(
            resource=resource,
            resource_name=resource_name,
            basepath=basepath,
            filepath=filepath,
            column_schema=column_schema,
            descriptor_filepath=descriptor_filepath,
            auto_validate=auto_validate,
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
