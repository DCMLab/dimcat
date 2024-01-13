"""Convenience module collecting DiMCAT's enum types for easy import."""
import logging

from dimcat.base import FriendlyEnum, LowercaseEnum, ObjectEnum
from dimcat.data.packages.base import PackageMode, PackageStatus
from dimcat.data.resources.base import FeatureName
from dimcat.data.resources.dc import Playthrough, ResourceStatus, UnitOfAnalysis
from dimcat.data.resources.facets import MuseScoreFacetName
from dimcat.data.resources.features import (
    BassNotesFormat,
    CadenceLabelFormat,
    HarmonyLabelsFormat,
    NotesFormat,
    PhraseComponentName,
)
from dimcat.data.resources.results import (
    NgramTableFormat,
    PhraseDataFormat,
    ResultName,
    TerminalSymbol,
)
from dimcat.plotting import GroupMode
from dimcat.steps.analyzers.base import AnalyzerName, DispatchStrategy
from dimcat.steps.loaders.base import FacetName
from dimcat.utils import SortOrder

module_logger = logging.getLogger(__name__)

__all__ = [
    AnalyzerName,
    BassNotesFormat,
    CadenceLabelFormat,
    DispatchStrategy,
    FacetName,
    FeatureName,
    FriendlyEnum,
    GroupMode,
    HarmonyLabelsFormat,
    LowercaseEnum,
    MuseScoreFacetName,
    NgramTableFormat,
    NotesFormat,
    ObjectEnum,
    PackageMode,
    PackageStatus,
    PhraseComponentName,
    PhraseDataFormat,
    Playthrough,
    ResultName,
    ResourceStatus,
    SortOrder,
    TerminalSymbol,
    UnitOfAnalysis,
]
