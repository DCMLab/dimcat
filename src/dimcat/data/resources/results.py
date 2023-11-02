from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from dimcat.base import ObjectEnum
from dimcat.plotting import (
    GroupMode,
    make_bar_plot,
    make_bubble_plot,
    make_lof_bar_plot,
    make_lof_bubble_plot,
)
from plotly import graph_objs as go

from .dc import DimcatResource

logger = logging.getLogger(__name__)


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Durations = "Durations"
    Result = "Result"


class Result(DimcatResource):
    _enum_type = ResultName

    @property
    def y_column(self) -> str:
        return self.df.columns[-1]

    def plot(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        unit_of_analysis = self.get_grouping_levels()
        y_col = unit_of_analysis[-1]
        x_col = self.value_column
        return self.make_bubble_plot(
            x_col=x_col,
            y_col=y_col,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def plot_grouped(
        self,
        group_cols: Optional[str | Iterable[str]] = None,
        group_modes: Iterable[GroupMode] = (
            GroupMode.COLOR,
            GroupMode.ROWS,
            GroupMode.COLUMNS,
        ),
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        if group_cols is None:
            group_cols = self.get_default_groupby()
        if not group_cols:
            x_col = self.value_column
        else:
            x_col, *group_cols = group_cols
        return self.make_bar_plot(
            x_col=x_col,
            group_cols=group_cols,
            group_modes=group_modes,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def make_bar_plot(
        self,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        group_cols: Optional[str | Iterable[str]] = None,
        group_modes: Iterable[GroupMode] = (
            GroupMode.COLOR,
            GroupMode.ROWS,
            GroupMode.COLUMNS,
        ),
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        if y_col is None:
            y_col = self.y_column
        return make_bar_plot(
            df=self.df,
            x_col=x_col,
            y_col=y_col,
            group_cols=group_cols,
            group_modes=group_modes,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def make_bubble_plot(
        self,
        normalize: bool = False,
        flip: bool = False,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        duration_column: str = "duration_qb",
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        width: Optional[int] = 1200,
        height: Optional[int] = 1500,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        if y_col is None:
            y_col = self.y_column
        return make_bubble_plot(
            df=self.df,
            normalize=normalize,
            flip=flip,
            x_col=x_col,
            y_col=y_col,
            duration_column=duration_column,
            title=title,
            labels=labels,
            hover_data=hover_data,
            width=width,
            height=height,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )


class Durations(Result):
    pass


class PitchClassDurations(Durations):
    def make_bar_plot(
        self,
        x_col="tpc",
        y_col=None,
        group_cols: Optional[str | Iterable[str]] = None,
        group_modes: Iterable[GroupMode] = (
            GroupMode.COLOR,
            GroupMode.ROWS,
            GroupMode.COLUMNS,
        ),
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        if y_col is None:
            y_col = self.y_column
        return make_lof_bar_plot(
            df=self.df,
            x_col=x_col,
            y_col=y_col,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def make_bubble_plot(
        self,
        normalize: bool = False,
        flip: bool = False,
        x_col: Optional[str] = "tpc",
        y_col: Optional[str] = None,
        duration_column: str = "duration_qb",
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        width: Optional[int] = 1200,
        height: Optional[int] = 1500,
        layout: Optional[dict] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """

        Args:
            layout: Keyword arguments passed to fig.update_layout()
            **kwargs: Keyword arguments passed to the Plotly plotting function.

        Returns:
            A Plotly Figure object.
        """
        if y_col is None:
            y_col = self.y_column
        return make_lof_bubble_plot(
            df=self.df,
            normalize=normalize,
            flip=flip,
            x_col=x_col,
            y_col=y_col,
            duration_column=duration_column,
            title=title,
            labels=labels,
            hover_data=hover_data,
            width=width,
            height=height,
            layout=layout,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )
