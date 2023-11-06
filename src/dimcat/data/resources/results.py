from __future__ import annotations

import logging
from functools import cache, cached_property
from itertools import product
from typing import Iterable, List, Optional

import frictionless as fl
import pandas as pd
from dimcat.base import ObjectEnum
from dimcat.plotting import (
    GroupMode,
    make_bar_plot,
    make_bubble_plot,
    make_lof_bar_plot,
    make_lof_bubble_plot,
    make_transition_heatmap_plots,
)
from dimcat.utils import SortOrder
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
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


def tuple2str(tup: tuple) -> str:
    """Used for displaying n-grams on axes."""
    try:
        return ", ".join(str(e) for e in tup)
    except TypeError:
        return str(tup)


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Durations = "Durations"
    NgramTable = "NgramTable"
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


class NgramTable(Result):
    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        # ToDo: These fields need to be added to the schema for pickling. This will mean adding them to the init
        #  arguments (currently these are added by NgramAnalyzer._post_process_result()).
        self.context_column_names: List[str] = []
        self.feature_column_names: List[str] = []
        self.auxiliary_column_names: List[str] = []

    @cached_property
    def ngram_levels(self) -> List[str]:
        return list(self.df.columns.levels[0])

    def make_bigram_tuples(
        self,
        columns: Optional[str | List[str]] = None,
        split: int = -1,
        as_string: bool = False,
    ) -> pd.DataFrame:
        """Get a list of bigram tuples where each tuple contains two tuples the values of which correspond to the
        specified columns.

        Args:
            columns: Columns from which to construct bigrams. Defaults to feature columns.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                elements are to be part of the left and the right gram. Defaults to -1, i.e. the last element is
                used as the right gram. This is a useful default for settings where the (n-1) previous elements are
                the context for predicting the next element.


        Returns:
            Like :meth:`make_ngram_tuples`, but condensed to two columns.
        """
        if len(self.ngram_levels) == 2:
            result = self.make_ngram_tuples(columns=columns)
        else:
            ngram_tuples = self.make_ngram_tuples(columns=columns).itertuples(
                index=False, name=None
            )
            data = [
                (ngram_tuple[:split], ngram_tuple[split:])
                for ngram_tuple in ngram_tuples
            ]
            result = pd.DataFrame(data, columns=["a", "b"], index=self.df.index)
        if as_string:
            result = result.map(tuple2str)
        return result

    def get_transitions(
        self,
        columns: Optional[str | List[str]] = None,
        split: int = -1,
        as_string: bool = False,
    ) -> pd.DataFrame:
        """Get a Series that counts for each context the number of transitions to each possible following element.

        Args:
            columns: Columns from which to construct bigrams. Defaults to feature columns.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                elements are to be part of the left and the right gram. Defaults to -1, i.e. the last element is
                used as the right gram. This is a useful default for settings where the (n-1) previous elements are
                the context for predicting the next element.
            smooth: Initial count value of all transitions
            as_string: Set to True to convert the tuples to strings.

        Returns:
            Dataframe with columns 'count' and 'proportion', showing each (n-1) previous elements (index level 0),
            the count and proportion of transitions to each possible following element (index level 1).
        """
        bigrams = self.make_bigram_tuples(
            columns=columns, split=split, as_string=as_string
        )
        gpb = bigrams.groupby("a").b
        return pd.concat([gpb.value_counts(), gpb.value_counts(normalize=True)], axis=1)

    @cache
    def make_ngram_tuples(
        self,
        columns: Optional[str | List[str]] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Reduce the selected columns for the n first n-gram levels a, b, ... so that the resulting dataframe
        contains n columns each of which contains tuples.

        Args:
            columns: Use only the selected column(s). Defaults to feature columns.
            n:
                Only make tuples for the first n n-gram levels. If None, use all n-gram levels. Minimum is 2, maximum
                is the number of n-gram levels determined by the :obj:`NgramAnalyzer` used to create the n-gram table.

        Returns:

        """
        if columns is None:
            columns = self.feature_column_names
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)
        if n is not None:
            n = int(n)
            assert 1 < n <= len(self.ngram_levels)
            selected_levels = self.ngram_levels[:n]
        else:
            selected_levels = self.ngram_levels
        selected_columns = list(product(selected_levels, columns))
        selection = self.df[selected_columns]
        return selection.groupby(level=0, axis=1).apply(
            lambda df: df.apply(tuple, axis=1)
        )

    def plot_grouped(
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
    ) -> MatplotlibFigure:
        """The arguments are currently ignored because the heatmaps have not been implemented in Plotly yet."""
        transitions = self.get_transitions(self.x_column, as_string=True)
        transition_matrix = transitions.proportion.unstack(fill_value=0.0)
        unigram_stats = (
            transitions.groupby("a", sort=False)["count"]
            .sum()
            .sort_values(ascending=False)
        )
        unigram_stats /= unigram_stats.sum()

        def sort_by_prevalence(index):
            missing = index.difference(unigram_stats.index)
            if len(missing) > 0:
                highest = unigram_stats.max()
                shared = index.intersection(unigram_stats.index)
                result = unigram_stats.loc[shared].reindex(
                    index, fill_value=highest + 1
                )
            else:
                result = unigram_stats.loc[index]
            return pd.Index(result.values, name=index.name)

        transition_matrix = (
            transition_matrix.sort_index(
                key=sort_by_prevalence, ascending=False
            ).sort_index(axis=1, key=sort_by_prevalence, ascending=False)
        ) * 100
        fig = make_transition_heatmap_plots(
            left_transition_matrix=transition_matrix,
            left_unigrams=unigram_stats,
            frequencies=True,
        )
        if output is not None:
            plt.savefig(output, dpi=400)
        return fig


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
