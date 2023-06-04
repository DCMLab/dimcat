from __future__ import annotations

import logging
from typing import Optional

from dimcat.base import ObjectEnum
from plotly import express as px
from plotly import graph_objs as go

from .base import DimcatResource

logger = logging.getLogger(__name__)


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Result = "Result"


class Result(DimcatResource):
    _enum_type = ResultName

    def plot(self, layout: Optional[dict] = None, **kwargs) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        groups = self.get_default_groupby()
        for level in ("corpus", "piece"):
            if level in groups:
                groups.remove(level)
        if len(groups) > 0 and "color" not in kwargs:
            kwargs["color"] = groups[0]
            print(kwargs)
        df = self.df.reset_index()
        fig = px.bar(
            df,
            x=df.columns[-2],
            y=df.columns[-1],
            hover_data=["corpus", "piece"],
            **kwargs,
        )
        figure_layout = dict(
            xaxis_type="category",  # prevent Plotly from interpreting values as dates
        )
        if layout is not None:
            figure_layout.update(layout)
        fig.update_layout(figure_layout)
        return fig
