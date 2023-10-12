from __future__ import annotations

import logging
from typing import Optional

from dimcat.base import ObjectEnum
from plotly import express as px
from plotly import graph_objs as go

from .dc import DimcatResource
from .plotting import tpc_bubbles

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
        groups = self.get_default_groupby()
        if len(groups) > 0 and "color" not in kwargs:
            kwargs["color"] = groups[0]
        df = self.df.reset_index()
        fig = px.bar(
            df,
            x=df.columns[-2],
            y=df.columns[-1],
            hover_data=["corpus", "piece"],
            height=500,
            **kwargs,
        )
        figure_layout = dict(
            xaxis_type="category",  # prevent Plotly from interpreting values as dates
        )
        if layout is not None:
            figure_layout.update(layout)
        fig.update_layout(figure_layout)
        return fig


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
