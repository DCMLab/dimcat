from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import pandas as pd
from dimcat.base import ObjectEnum
from dimcat.plotting import (
    GroupMode,
    make_bar_plot,
    make_bubble_plot,
    make_lof_bar_plot,
    make_lof_bubble_plot,
)
from dimcat.utils import SortOrder
from plotly import graph_objs as go

from .dc import DimcatResource

logger = logging.getLogger(__name__)


def clean_axis_labels(*labels: str) -> dict:
    """Clean axis labels for Plotly plots.

    Args:
        *labels: Labels to clean.

    Returns:
        A dictionary with the cleaned labels.
    """
    return {label: label.replace("_", " ") for label in labels}


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Durations = "Durations"
    Result = "Result"
    PitchClassDurations = "PitchClassDurations"


class Result(DimcatResource):
    _enum_type = ResultName

    @property
    def x_column(self) -> str:
        return self.value_column

    @property
    def y_column(self) -> str:
        return self.df.columns[-1]

    def combine(
        self,
        group_cols: Optional[str | Iterable[str]] = None,
        sort_order: Optional[SortOrder] = SortOrder.NONE,
    ):
        if group_cols is None:
            group_cols = self.get_default_groupby()
        elif isinstance(group_cols, str):
            group_cols = [group_cols]
        else:
            group_cols = list(group_cols)
        groupby = group_cols + [self.x_column]
        combined_result = self.df.groupby(groupby).sum()
        if sort_order is None or sort_order == SortOrder.NONE:
            return combined_result
        if not group_cols:
            # no grouping required
            if sort_order == SortOrder.ASCENDING:
                return combined_result.sort_values(self.y_column)
            else:
                return combined_result.sort_values(self.y_column, ascending=False)
        if sort_order == SortOrder.ASCENDING:
            return combined_result.groupby(group_cols, group_keys=False).apply(
                lambda df: df.sort_values(self.y_column)
            )
        else:
            return combined_result.groupby(group_cols, group_keys=False).apply(
                lambda df: df.sort_values(self.y_column, ascending=False)
            )

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
        labels_settings = clean_axis_labels(x_col, y_col)
        if labels is not None:
            labels_settings.update(labels)
        return self.make_bubble_plot(
            x_col=x_col,
            y_col=y_col,
            title=title,
            labels=labels_settings,
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
        elif isinstance(group_cols, str):
            group_cols = [group_cols]
        else:
            group_cols = list(group_cols)
        if not group_cols:
            x_col = self.x_column
            y_col = self.y_column
            labels_settings = clean_axis_labels(x_col, y_col)
            if labels is not None:
                labels_settings.update(labels)
            combined_result = self.combine()
            return self.make_bar_plot(
                df=combined_result,
                x_col=x_col,
                group_cols=group_cols,
                group_modes=group_modes,
                title=title,
                labels=labels_settings,
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
        else:
            y_col = group_cols[-1]
            x_col = self.x_column
            labels_settings = clean_axis_labels(x_col, y_col)
            if labels is not None:
                labels_settings.update(labels)
            return self.make_bubble_plot(
                x_col=x_col,
                y_col=y_col,
                title=title,
                labels=labels_settings,
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
        df: Optional[pd.DataFrame] = None,
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
        if df is None:
            df = self.df
        if y_col is None:
            y_col = self.y_column
        return make_bar_plot(
            df=df,
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
        normalize: bool = True,
        flip: bool = False,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        duration_column: str = "duration_qb",
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
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
        df: Optional[pd.DataFrame] = None,
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
        if df is None:
            df = self.df
        if y_col is None:
            y_col = self.y_column
        return make_lof_bar_plot(
            df=df,
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
        normalize: bool = True,
        flip: bool = False,
        x_col: Optional[str] = "tpc",
        y_col: Optional[str] = None,
        duration_column: str = "duration_qb",
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
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
