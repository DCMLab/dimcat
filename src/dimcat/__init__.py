import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .analyzer import (
    ChordSymbolBigrams,
    ChordSymbolUnigrams,
    PitchClassVectors,
    TPCrange,
)
from .data import Corpus
from .filter import IsAnnotatedFilter
from .grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
from .pipeline import Pipeline
from .slicer import LocalKeySlicer, NoteSlicer
from .writer import TSVWriter
