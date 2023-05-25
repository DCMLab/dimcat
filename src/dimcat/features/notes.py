from enum import Enum
from typing import Optional

import frictionless as fl
from dimcat.data.base import D
from dimcat.features.base import Feature
from marshmallow import fields


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
            df=df,
            resource_name=resource_name,
            resource=resource,
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
