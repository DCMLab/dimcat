import logging

from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .base import (
    Data,
    DimcatConfig,
    DimcatObject,
    DimcatSchema,
    PipelineStep,
    get_class,
    get_schema,
)
from .dataset.base import Dataset, DimcatPackage
from .resources.base import DimcatIndex, DimcatResource, PieceIndex
from .resources.features import FeatureName

logger = logging.getLogger(__name__)
