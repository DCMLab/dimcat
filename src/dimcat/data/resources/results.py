from __future__ import annotations

import logging
from typing import Optional

from dimcat.base import ObjectEnum
from dimcat.plotting import make_bar_plot, tpc_bubbles
from plotly import graph_objs as go

from .dc import DimcatResource

logger = logging.getLogger(__name__)


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Durations = "Durations"
    Result = "Result"


class Result(DimcatResource):
    _enum_type = ResultName

    def plot(self, layout: Optional[dict] = None, **kwargs) -> go.Figure:
        return self.make_bar_plot(layout=layout, **kwargs)

    def make_bar_plot(self, layout: Optional[dict] = None, **kwargs) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        group_cols = self.get_default_groupby()
        return make_bar_plot(
            df=self.df,
            group_cols=group_cols,
            layout=layout,
            **kwargs,
        )


class Durations(Result):
    def plot(self, layout: Optional[dict] = None, **kwargs) -> go.Figure:
        return self.make_bubble_plot(layout=layout, **kwargs)

    def make_bubble_plot(self, layout: Optional[dict] = None, **kwargs) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        groups = self.get_grouping_levels()
        normalized = self.df.groupby(groups, group_keys=False).apply(
            lambda S: S / S.sum()
        )
        title = "Normalized pitch class durations"
        return tpc_bubbles(normalized, title=title, layout=layout, **kwargs)
