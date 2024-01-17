from typing import ClassVar, Optional

from dimcat.data.resources.results import PrevalenceMatrix
from dimcat.steps.analyzers.base import Analyzer


class PrevalenceAnalyzer(Analyzer):
    """Creates what is the equivalent to NLP's "frequency matrix" except that in the case of music,
    the coefficients are not restricted to represent count frequencies (when created from a
    :class:`~.data.resources.results.Counts` object) but can also represent durations (when created
    from a :class:`~.data.resources.results.Durations` object). When the analyzer is applied to
    a :class:`Feature`, its default analysis will be used.
    """

    _default_dimension_column: ClassVar[Optional[str]] = "duration_qb"
    _new_resource_type = PrevalenceMatrix
