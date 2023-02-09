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
    LocalKeySequence,
    LocalKeyUnique,
    PitchClassVectors,
    TPCrange,
)
from .data import Dataset
from .filter import HasCadenceAnnotationsFilter, IsAnnotatedFilter
from .grouper import CorpusGrouper, ModeGrouper, PieceGrouper, YearGrouper
from .pipeline import Pipeline
from .slicer import (
    ChordCriterionSlicer,
    ChordFeatureSlicer,
    LocalKeySlicer,
    MeasureSlicer,
    NoteSlicer,
    PhraseSlicer,
)
from .writer import TSVWriter
