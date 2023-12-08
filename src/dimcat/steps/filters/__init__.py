"""Subclasses of groupers which eliminate certain groups, excluding them from further processing."""


from .annotations import HasCadenceAnnotationsFilter, HasHarmonyLabelsFilter
from .base import CorpusFilter, PieceFilter
