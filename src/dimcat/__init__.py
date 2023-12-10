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

# modules of dimcat.data are not allowed to import from dimcat.steps, so when they do, they use get_class() which
# requires that the respective step was already "seen" and is part of the registry. Hence, although the main purpose
# of the imports here is syntactic sugar, some are required.
from .base import (
    DimcatConfig,
    change_setting,
    deserialize_config,
    deserialize_dict,
    deserialize_json_file,
    deserialize_json_str,
    get_class,
    get_schema,
    get_setting,
    reset_settings,
)
from .data import catalogs, datasets, packages, resources
from .data.datasets.base import Dataset
from .data.resources import PieceIndex
from .steps import analyzers, extractors, filters, groupers, loaders, pipelines, slicers
from .steps.pipelines.base import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s -- %(pathname)s (line %(lineno)s) in %(funcName)s():\n\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
