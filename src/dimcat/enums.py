"""Convenience module collecting DiMCAT's enum types for easy import."""

from dimcat.base import FriendlyEnum, ObjectEnum
from dimcat.data.packages.base import PackageMode, PackageStatus
from dimcat.data.resources.base import FeatureName
from dimcat.data.resources.dc import ResourceStatus, UnitOfAnalysis
from dimcat.data.resources.facets import MuseScoreFacetName
from dimcat.data.resources.features import (
    BassNotesFormat,
    CadenceLabelFormat,
    HarmonyLabelsFormat,
    NotesFormat,
)
from dimcat.data.resources.results import ResultName, TerminalSymbol
from dimcat.plotting import GroupMode
from dimcat.steps.analyzers.base import AnalyzerName
from dimcat.steps.analyzers.counters import NgramTableFormat
from dimcat.steps.loaders.base import FacetName
from dimcat.utils import SortOrder

__all__ = [
    AnalyzerName,
    BassNotesFormat,
    CadenceLabelFormat,
    FacetName,
    FeatureName,
    FriendlyEnum,
    GroupMode,
    HarmonyLabelsFormat,
    MuseScoreFacetName,
    NgramTableFormat,
    NotesFormat,
    ObjectEnum,
    PackageMode,
    PackageStatus,
    ResultName,
    ResourceStatus,
    SortOrder,
    TerminalSymbol,
    UnitOfAnalysis,
]
