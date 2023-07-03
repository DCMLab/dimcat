from __future__ import annotations

import logging
from typing import Optional

import marshmallow as mm
from dimcat.base import DimcatObject
from dimcat.utils import _set_new_basepath

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
        self._basepath = _set_new_basepath(basepath, self.logger)
