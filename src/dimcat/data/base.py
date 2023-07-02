from __future__ import annotations

import logging
import os
from typing import Optional

import marshmallow as mm
from dimcat.base import DimcatObject
from dimcat.utils import resolve_path

logger = logging.getLogger(__name__)


class Data(DimcatObject):
    """
    This base class unites all classes containing data in some way or another.
    """

    class Schema(DimcatObject.Schema):
        basepath = mm.fields.Str(
            required=False,
            allow_none=True,
            metadata=dict(description="The directory where data would be stored."),
        )

    def __init__(
        self,
        basepath: Optional[str] = None,
    ):
        super().__init__()
        self._basepath = None
        if basepath is not None:
            self.basepath = basepath

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        basepath_arg = resolve_path(basepath)
        if not os.path.isdir(basepath_arg):
            raise NotADirectoryError(
                f"basepath {basepath_arg!r} is not an existing directory."
            )
        self._basepath = basepath_arg
        self.logger.debug(
            f"The basepath of this {self.name} has been set to {basepath_arg!r}"
        )
