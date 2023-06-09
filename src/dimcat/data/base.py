from __future__ import annotations

import logging

from dimcat.base import DimcatObject

logger = logging.getLogger(__name__)


class Data(DimcatObject):
    """
    This base class unites all classes containing data in some way or another.
    """

    class Schema(DimcatObject.Schema):
        # basedir = fields.String(required=True)
        pass
