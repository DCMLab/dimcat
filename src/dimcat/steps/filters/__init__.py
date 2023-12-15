"""Subclasses of groupers which eliminate certain groups, excluding them from further processing."""
import logging

from .annotations import HasCadenceAnnotationsFilter, HasHarmonyLabelsFilter
from .base import CorpusFilter, PieceFilter

module_logger = logging.getLogger(__name__)
