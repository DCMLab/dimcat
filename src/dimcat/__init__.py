import logging
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .base import DimcatConfig, get_class, get_schema
from .data.dataset.base import Dataset
from .steps.extractor.base import FeatureExtractor
from .steps.pipeline.base import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s -- %(pathname)s (line %(lineno)s) in %(funcName)s():\n\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
