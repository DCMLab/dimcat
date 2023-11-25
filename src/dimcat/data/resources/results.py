from __future__ import annotations

import logging
from functools import cache, cached_property
from itertools import product
from typing import ClassVar, Iterable, List, Optional, Tuple

import frictionless as fl
import marshmallow as mm
import pandas as pd
from dimcat.base import ObjectEnum
from dimcat.plotting import (
    GroupMode,
    make_bar_plot,
    make_bubble_plot,
    make_lof_bar_plot,
    make_lof_bubble_plot,
    make_transition_heatmap_plots,
    update_plot_grouping_settings,
)
from dimcat.utils import SortOrder
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import graph_objs as go

from .base import D
from .dc import DimcatResource, UnitOfAnalysis

logger = logging.getLogger(__name__)


def tuple2str(tup: tuple) -> str:
    """Used for displaying n-grams on axes."""
    try:
        return ", ".join(str(e) for e in tup)
    except TypeError:
        return str(tup)


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    Counts = "Counts"
    Durations = "Durations"
    NgramTable = "NgramTable"
    Result = "Result"


class Result(DimcatResource):
    _enum_type = ResultName
    _default_group_modes: ClassVar[Tuple[GroupMode, ...]] = (
        GroupMode.COLOR,
        GroupMode.ROWS,
        GroupMode.COLUMNS,
    )
    """If the no other sequence of group_modes is specified when plotting, this default is zipped to the groupby
    columns to determine how the data will be grouped for the plot."""

    class Schema(DimcatResource.Schema):
        analyzed_resource = mm.fields.Nested(DimcatResource.Schema, required=True)
        value_column = mm.fields.Str(required=True)
        dimension_column = mm.fields.Str(required=True)
        formatted_column = mm.fields.Str(required=False)

    def __init__(
        self,
        analyzed_resource: DimcatResource,
        value_column: Optional[str],
        dimension_column: Optional[str],
        formatted_column: Optional[str] = None,
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
        # self._formatted_column and self._value_column are already set by super().__init__()
        self.formatted_column = formatted_column
        self.value_column = value_column
        self.analyzed_resource: DimcatResource = analyzed_resource
        self.dimension_column: Optional[str] = dimension_column
        """Name of the column containing some dimension, e.g. to be interpreted as quantity (durations, counts,
        etc.) or as color."""

    @property
    def formatted_column(self) -> str:
        """Name of the column containing the formatted values, typically for display on the x_axis."""
        return self._formatted_column

    @formatted_column.setter
    def formatted_column(self, formatted_column: str):
        self._formatted_column = formatted_column

    @cached_property
    def uses_line_of_fifths_colors(self) -> bool:
        """Whether or not the plots produced by this Result exhibit a color gradient along the line of fifths.
        This is typically the case for results based intervals, note names, or scale degrees. In these cases,
        the color dimension is lost for discerning between different groups, which are then typically shown in
        different rows or columns.
        """
        resource_format = self.analyzed_resource.format
        # since all format values are of type FriendlyEnum and can be compared with strings, no matter what specific
        # format Enum the analyzed resource was using, it can be checked against these fifths format strings:
        return resource_format in ("FIFTHS", "INTERVAL", "NAME", "SCALE_DEGREE")

    @property
    def value_column(self) -> str:
        """Name of the column containing the values, typically to arrange markers along the x_axis."""
        return self._value_column

    @value_column.setter
    def value_column(self, value_column: str):
        self._value_column = value_column

    @property
    def x_column(self) -> str:
        """Name of the result column from which to create one marker per distinct value to show over the x-axis."""
        return self.value_column

    @property
    def y_column(self) -> str:
        """Name of the numerical result column used for determining each marker's dimension along the y-axis."""
        return self.dimension_column

    def combine_results(
        self,
        group_cols: Optional[str | Iterable[str]] = None,
        sort_order: Optional[SortOrder] = SortOrder.NONE,
    ) -> D:
        """Aggregate results for each group, typically by summing up and normalizing the values. By default,
        the groups correspond to those that had been applied to the analyzed resource. If no Groupers had been
        applied, the entire dataset is treated as a single group.
        """
        if group_cols is None:
            group_cols = self.get_default_groupby()
        elif isinstance(group_cols, str):
            group_cols = [group_cols]
        else:
            group_cols = list(group_cols)
        groupby = group_cols + [self.value_column]
        if self.formatted_column:
            groupby.append(self.formatted_column)
        combined_result = self.df.groupby(groupby).sum()
        if group_cols:
            normalize_by = combined_result.groupby(group_cols).sum()
        else:
            normalize_by = combined_result.sum()
        group_proportions = (combined_result / normalize_by).rename(
            columns=lambda x: "proportion"
        )
        group_proportions_str = (
            group_proportions.mul(100)
            .round(2)
            .astype(str)
            .add(" %")
            .rename(columns=lambda x: "proportion_%")
        )
        combined_result = pd.concat(
            [combined_result, group_proportions, group_proportions_str], axis=1
        )
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

    def _get_color_midpoint(self) -> int:
        if self.analyzed_resource.format == "NAME":
            # if note names are displayed, center the color scale on the note D (2 fifths)
            return 2
        return 0

    def get_grouping_levels(
        self, smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE
    ) -> List[str]:
        """Returns the levels of the grouping index, i.e., all levels until and including 'piece'."""
        smallest_unit = UnitOfAnalysis(smallest_unit)
        if smallest_unit == UnitOfAnalysis.SLICE:
            but_last = 2 if self.has_distinct_formatted_column else 1
            return self.get_level_names()[:-but_last]
        if smallest_unit in (UnitOfAnalysis.PIECE, UnitOfAnalysis.SLICE):
            return self.get_piece_index(max_levels=0).names
        if smallest_unit == UnitOfAnalysis.GROUP:
            return self.get_default_groupby()

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
        return self.make_bubble_plot(
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
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = None,
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
        x_col = self.x_column
        y_col = self.y_column
        combined_result = self.combine_results()
        return self.make_bar_plot(
            df=combined_result,
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
        # # in principle, group distributions can also be displayed as bubble plots:
        # if not group_cols:
        #     ... # bar plot code
        # else:
        #     y_col = group_cols[-1]
        #     x_col = self.x_column
        #     labels_settings = clean_axis_labels(x_col, y_col)
        #     if labels is not None:
        #         labels_settings.update(labels)
        #     return self.make_bubble_plot(
        #         x_col=x_col,
        #         y_col=y_col,
        #         title=title,
        #         labels=labels_settings,
        #         hover_data=hover_data,
        #         height=height,
        #         width=width,
        #         layout=layout,
        #         x_axis=x_axis,
        #         y_axis=y_axis,
        #         color_axis=color_axis,
        #         traces_settings=traces_settings,
        #         output=output,
        #         **kwargs,
        #     )

    def make_bar_plot(
        self,
        df: Optional[pd.DataFrame] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        group_cols: Optional[str | Iterable[str]] = None,
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = None,
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
        if x_col is None:
            x_col = self.x_column
        if y_col is None:
            y_col = self.y_column
        if group_cols is None:
            group_cols = self.get_default_groupby()
        elif isinstance(group_cols, str):
            group_cols = [group_cols]
        if group_cols:
            group_modes = self._resolve_group_modes_arg(group_modes)
            update_plot_grouping_settings(kwargs, group_cols, group_modes)
        layout_update = dict()
        if layout is not None:
            layout_update.update(layout)
        if "xaxis_type" not in layout_update:
            layout_update["xaxis_type"] = "category"
        if self.uses_line_of_fifths_colors:
            color_midpoint = self._get_color_midpoint()
            x_names_col = self.formatted_column
            hover_cols = [x_names_col]
            if hover_data:
                hover_cols.extend(hover_data)
            return make_lof_bar_plot(
                df=df,
                fifths_transform=None,
                x_names_col=x_names_col,
                x_col=x_col,
                y_col=y_col,
                title=title,
                labels=labels,
                shift_color_midpoint=color_midpoint,
                hover_data=hover_cols,
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
            return make_bar_plot(
                df=df,
                x_col=x_col,
                y_col=y_col,
                title=title,
                labels=labels,
                hover_data=hover_data,
                height=height,
                width=width,
                layout=layout_update,
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
        dimension_column: Optional[str] = None,
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
        if x_col is None:
            x_col = self.value_column
        if y_col is None:
            unit_of_analysis = self.get_grouping_levels()
            y_col = unit_of_analysis[-1]
        if dimension_column is None:
            dimension_column = self.dimension_column
        layout_update = dict()
        if layout is not None:
            layout_update.update(layout)
        if "yaxis_type" not in layout_update:
            layout_update["yaxis_type"] = "category"
        resource_format = self.analyzed_resource.format
        if resource_format in ("FIFTHS", "INTERVAL", "NAME", "SCALE_DEGREE"):
            color_midpoint = self._get_color_midpoint()
            x_names_col = self.formatted_column
            hover_cols = [x_names_col]
            if hover_data:
                hover_cols.extend(hover_data)
            return make_lof_bubble_plot(
                df=self.df,
                normalize=normalize,
                flip=flip,
                fifths_col=x_col,
                y_col=y_col,
                duration_column=dimension_column,
                x_names_col=x_names_col,
                title=title,
                labels=labels,
                hover_data=hover_cols,
                shift_color_midpoint=color_midpoint,
                width=width,
                height=height,
                layout=layout_update,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )
        else:
            return make_bubble_plot(
                df=self.df,
                normalize=normalize,
                flip=flip,
                x_col=x_col,
                y_col=y_col,
                duration_column=dimension_column,
                title=title,
                labels=labels,
                hover_data=hover_data,
                width=width,
                height=height,
                layout=layout_update,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )

    def _resolve_group_modes_arg(
        self, group_modes: Optional[GroupMode | Iterable[GroupMode]] = None
    ) -> List[GroupMode]:
        """Turns the argument into a list of GroupMode members and, if the COLOR dimension is occupied by line of
        fifths coloring, removes grouping by COLOR from the list."""
        if group_modes is None:
            group_modes = self._default_group_modes
        elif isinstance(group_modes, str):
            group_modes = [GroupMode(group_modes)]
        else:
            group_modes = [GroupMode(mode) for mode in group_modes]
        if self.uses_line_of_fifths_colors and GroupMode.COLOR in group_modes:
            group_modes = [mode for mode in group_modes if mode != GroupMode.COLOR]
            self.logger.debug(
                f"Removed {GroupMode.COLOR} from group_modes because {self.resource_name!r} uses line-of_fifths "
                f"coloring."
            )
        return group_modes


class Counts(Result):
    pass


class Durations(Result):
    pass


class NgramTable(Result):
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
            columns = self.analyzed_resource._feature_column_names
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
