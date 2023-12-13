from __future__ import annotations

import logging
import math
from functools import cache, cached_property, partial
from itertools import product, repeat
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import frictionless as fl
import marshmallow as mm
import ms3
import numpy as np
import pandas as pd
from dimcat.base import LowercaseEnum, ObjectEnum, get_setting
from dimcat.plotting import (
    CADENCE_COLORS,
    GroupMode,
    make_bar_plot,
    make_bubble_plot,
    make_heatmap,
    make_lof_bar_plot,
    make_lof_bubble_plot,
    make_pie_chart,
    update_figure_layout,
    update_plot_grouping_settings,
    write_image,
)
from dimcat.utils import SortOrder
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy import special
from typing_extensions import Self

from .base import D, S
from .dc import DimcatResource, UnitOfAnalysis

logger = logging.getLogger(__name__)

str_or_sequence = TypeAlias = Union[str, Sequence[str]]


@cache
def logarithm_function(
    base: Literal[10, 2, math.e] = 2,
    numpy=False,
) -> Callable:
    if numpy:
        if base == 2:
            return math.log2
        if base == 10:
            return math.log10
        if base == math.e:
            return math.log
        raise NotImplementedError(f"base {base} not implemented")
    if base == 2:
        return np.log2
    if base == 10:
        return np.log10
    if base == math.e:
        return np.log
    raise NotImplementedError(f"base {base} not implemented")


def compute_entropy_of_observations(
    observations: Iterable[Any],
    base: Literal[10, 2, math.e] = 2,
) -> float:
    """Compute the Shannon entropy of an array of observations by counting the values."""
    return compute_entropy_of_probabilities(
        pd.Series(observations).value_counts(), base, skip_check=True
    )


def compute_entropy_of_occurrences(
    occurrences: Iterable[int],
    base: Literal[10, 2, math.e] = 2,
) -> float:
    """Compute the Shannon entropy of the given absolute frequencies where each integer represents the number of
    observed occurrences of a category."""
    return compute_entropy_of_probabilities(occurrences, base, skip_check=True)


def _entropy(
    pk: np.typing.ArrayLike, base: float | None = None, axis: int = 0
) -> np.number | np.ndarray:
    """This is a copy of scipy.stats.entropy @ v1.11.4 leaving out the `np.asarray` call causing the problem
    reported under https://github.com/pandas-dev/pandas/issues/56472 Tested for unidimensional input only (had to
    drop the 'keepdims' argument). Apparently, this workaround will not be necessary anymore from pandas 2.2 on.
    """
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    # pk = np.asarray(pk)
    pk = 1.0 * pk / np.sum(pk, axis=axis)
    # if qk is None:
    vec = special.entr(pk)
    # else:
    #     qk = np.asarray(qk)
    #     pk, qk = np.broadcast_arrays(pk, qk)
    #     qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
    #     vec = special.rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def compute_entropy_of_probabilities(
    probabilities: Iterable[float] | Iterable[int],
    base: Literal[10, 2, math.e] = 2,
    skip_check: bool = False,
) -> float:
    """Compute the Shannon entropy of the given probability distribution, which is expected to be normalized.

    Args:
        probabilities:
        normalize:
            If True (default), the entropy is normalized to the range [0, 1] based on the maximum entropy for the
            given number of propabilities.
        base: Logarithmic base for computing the entropy.
        skip_check:
            If False (default) the probabilities are asserted to sum to 1. Pass True when you have normalized the
            data yourself or when you're passing occurrences rather than probabilities.

    Returns:
        The absolute or normalized Shannon entropy of the given probability distribution.
    """
    if not skip_check:
        assert math.isclose(
            (p_sum := sum(probabilities)), 1
        ), f"Expecting normalized probabilites, these sum to {p_sum}."
    return _entropy(probabilities, base=base)


class TerminalSymbol(LowercaseEnum):
    """Used to control arguments for n-gram creation. DEFAULT defines the default terminal symbol.
    NA replaces each terminal value with pd.NA values (rather than, say, with a tuple of null values).
    DROP results in terminal n-grams being dropped entirely, that is, those starting with one of the n-1 last n-grams
    of a sequence.
    """

    DEFAULT = get_setting("default_terminal_symbol")
    NA = pd.NA
    DROP = "DROP"
    # Caution: adding options needs to be done with care, in particular with NgramTable._make_ngram_component()


def turn_proportions_into_percentage_strings(
    df: pd.DataFrame | pd.Series, column_name: str = "proportion_%"
) -> pd.DataFrame | pd.Series:
    """Interprets the Series or all columns of the DataFrame as proportions, multiplies them by 100 and turns them
    into strings with a % sign.
    """
    result = df.mul(100).round(2).astype(str).add(" %")
    if isinstance(df, pd.DataFrame):
        return result.rename(columns=lambda x: column_name)
    else:
        return result.rename(column_name)


def tuple2str(
    tup: tuple,
    join_str: Optional[str] = ", ",
    recursive: bool = True,
    keep_parentheses: bool = False,
) -> str:
    """Used for turning n-gram components into strings, e.g. for display on plot axes.

    Args:
        tup: Tuple to be returned as string.
        join_str:
            String to be interspersed between tuple elements. If None, result is ``str(tup)`` and ``recursive`` is
            ignored.
        recursive:
            If True (default) tuple elements that are tuples themselves will be joined together recursively, using the
            same ``join_str`` (except when it's None). Inner tuples always keep their parentheses.
        keep_parentheses: If False (default), the outer parentheses are removed. Pass True to keep them in the string.

    Returns:
        A string representing the tuple.
    """
    try:
        if join_str is None:
            result = str(tup)
            if keep_parentheses:
                return result
            return result[1:-1]
        if recursive:
            result = join_str.join(
                tuple2str(e, join_str=join_str, keep_parentheses=True)
                if isinstance(e, tuple)
                else str(e)
                for e in tup
            )
        else:
            result = join_str.join(str(e) for e in tup)
    except TypeError:
        return str(tup)
    if keep_parentheses:
        return f"({result})"
    return result


class ResultName(ObjectEnum):
    """Identifies the available analyzers."""

    CadenceCounts = "CadenceCounts"
    Counts = "Counts"
    Durations = "Durations"
    NgramTable = "NgramTable"
    NgramTuples = "NgramTuples"
    Result = "Result"
    Transitions = "Transitions"


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
        formatted_column = mm.fields.Str(allow_none=True)

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
        self.analyzed_resource: DimcatResource = analyzed_resource
        self.value_column = value_column
        self.dimension_column: Optional[str] = dimension_column
        """Name of the column containing some dimension, e.g. to be interpreted as quantity (durations, counts,
        etc.) or as color."""
        self.formatted_column = formatted_column
        self.is_combination = False
        """Is True if this Result has been created by Result.combine_results(), in which case the method will return
        :attr:`df` as is (without combining anything)."""

    @property
    def feature_columns(self) -> List[str]:
        """The :attr:`column` and, if distinct, the :attr:`formatted_column`, as a list."""
        result = [self.value_column]
        if self.has_distinct_formatted_column:
            result.append(self.formatted_column)
        return result

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
        return resource_format in (
            "FIFTHS",
            "INTERVAL",
            "NAME",
            "SCALE_DEGREE",
            "SCALE_DEGREE_MAJOR",
            "SCALE_DEGREE_MINOR",
        )

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
        if self.uses_line_of_fifths_colors or not self.formatted_column:
            return self.value_column
        else:
            return self.formatted_column

    @property
    def y_column(self) -> str:
        """Name of the numerical result column used for determining each marker's dimension along the y-axis."""
        return self.dimension_column

    def _add_proportion_columns(self, combined_result: D, normalize_by: S | float) -> D:
        """Normalize the combined results and concatenate them as two new column, 'proportion' and 'proportion_%'."""
        try:
            group_proportions = (combined_result / normalize_by).rename(
                columns=lambda x: "proportion"
            )
        except Exception as e:
            raise RuntimeError(
                f"Normalizing the combined results failed with the following exception:\n{e!r}\n"
                f"We were trying to divide\n{combined_result}\nby\n{normalize_by}"
            )
        group_proportions_str = turn_proportions_into_percentage_strings(
            group_proportions
        )
        combined_result = pd.concat(
            [combined_result, group_proportions, group_proportions_str], axis=1
        )
        return combined_result

    def _combine_results(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ) -> D:
        """Aggregate results for each group, typically by summing up and normalizing the values. By default,
        the groups correspond to those that had been applied to the analyzed resource. If no Groupers had been
        applied, the entire dataset is treated as a single group.
        """
        group_cols = self._resolve_group_cols_arg(group_cols)

        if self.is_combination:
            # this has been combined before, check if the grouping is the same or a subset of the current grouping
            available_columns = set(self.df.columns) | set(self.df.index.names)
            if group_cols == self.default_groupby:
                return self.df
            elif not set(group_cols).issubset(available_columns):
                raise ValueError(
                    f"Cannot group the results that are already combined by {group_cols}. "
                    f"Available columns are {available_columns}"
                )
            else:
                df = self.df[
                    [self.dimension_column]
                ]  # gets rid of existing proportion columns, we will get new ones
        else:
            df = self.df

        groupby = group_cols + self.feature_columns
        combined_result = df.groupby(groupby).sum()
        if group_cols:
            normalize_by = combined_result.groupby(group_cols).sum()
        else:
            normalize_by = combined_result.sum()
        combined_result = self._add_proportion_columns(combined_result, normalize_by)
        return self._sort_combined_result(combined_result, group_cols, sort_order)

    def combine_results(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ) -> Self:
        """Aggregate results for each group, typically by summing up and normalizing the values. By default,
        the groups correspond to those that had been applied to the analyzed resource. If no Groupers had been
        applied, the entire dataset is treated as a single group.
        """
        group_cols = self._resolve_group_cols_arg(group_cols)
        combined_results = self._combine_results(
            group_cols=group_cols, sort_order=sort_order
        )
        new_result = self.__class__.from_resource_and_dataframe(
            self,
            combined_results,
            default_groupby=group_cols,
        )
        new_result.is_combination = True
        return new_result

    def _compute_entropy(
        self, combined_result: D, group_cols: List[str], weighted: bool = False
    ) -> S:
        if group_cols:
            gpb = combined_result.groupby(group_cols)
            group_entropies = gpb.proportion.apply(compute_entropy_of_probabilities)
            if not weighted:
                return group_entropies.rename("entropy")
            group_occurrences = gpb["count"].sum()
            return (
                group_entropies.mul(group_occurrences) / group_occurrences.sum()
            ).rename("weighted_entropy")
        return compute_entropy_of_probabilities(combined_result.proportion).rename(
            "entropy"
        )

    def compute_entropy(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        weighted: bool = False,
    ) -> S:
        """Compute the Shannon entropies of the probability distributions for the default or specified grouping.

        Args:
            group_cols: For which groups to compute entropy values.
            weighted:
                If True, the entropy values will be weighted by the relative prevalence of the respective group. If
                no grouping is specified, this argument has no effect.

        Returns:
            A Series of entropy values, indexed by the group names.
        """
        group_cols = self._resolve_group_cols_arg(group_cols)
        combined_result = self._combine_results(
            group_cols=group_cols, sort_order=SortOrder.NONE
        )
        return self._compute_entropy(combined_result, group_cols, weighted=weighted)

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
        if smallest_unit == UnitOfAnalysis.PIECE:
            return self.get_piece_index(max_levels=0).names
        if smallest_unit == UnitOfAnalysis.GROUP:
            return self.default_groupby

    def make_bar_plot(
        self,
        df: Optional[D] = None,
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
        font_size: Optional[int] = None,
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
            x_col = self.x_column
        if y_col is None:
            y_col = self.y_column
        group_cols = self._resolve_group_cols_arg(group_cols)
        if group_cols:
            group_modes = self._resolve_group_modes_arg(group_modes)
            update_plot_grouping_settings(kwargs, group_cols, group_modes)
        if df is None:
            if group_cols:
                df = self._combine_results(group_cols=group_cols)
            else:
                df = self.df
        layout_update = dict()
        if layout is not None:
            layout_update.update(layout)
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
                font_size=font_size,
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
                font_size=font_size,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )

    def make_bubble_plot(
        self,
        df: Optional[D] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        group_cols: Optional[str | Iterable[str]] = None,
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = (
            GroupMode.ROWS,
            GroupMode.COLUMNS,
        ),
        normalize: bool = True,
        dimension_column: Optional[str] = None,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
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
            x_col = self.x_column
        if y_col is None:
            unit_of_analysis = self.get_grouping_levels()
            y_col = unit_of_analysis[-1]
        if df is None:
            df = self.df
        group_cols = self._resolve_group_cols_arg(group_cols)
        if group_cols:
            group_modes = self._resolve_group_modes_arg(group_modes)
        if dimension_column is None:
            dimension_column = self.dimension_column
        layout_update = dict()
        if layout is not None:
            layout_update.update(layout)
        if self.uses_line_of_fifths_colors:
            color_midpoint = self._get_color_midpoint()
            x_names_col = self.formatted_column
            hover_cols = [x_names_col]
            if hover_data:
                hover_cols.extend(hover_data)
            return make_lof_bubble_plot(
                df=df,
                normalize=normalize,
                x_col=x_col,
                y_col=y_col,
                dimension_column=dimension_column,
                group_cols=group_cols,
                group_modes=group_modes,
                x_names_col=x_names_col,
                title=title,
                labels=labels,
                hover_data=hover_cols,
                shift_color_midpoint=color_midpoint,
                width=width,
                height=height,
                layout=layout_update,
                font_size=font_size,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )
        else:
            return make_bubble_plot(
                df=df,
                normalize=normalize,
                x_col=x_col,
                y_col=y_col,
                dimension_column=dimension_column,
                group_cols=group_cols,
                group_modes=group_modes,
                title=title,
                labels=labels,
                hover_data=hover_data,
                width=width,
                height=height,
                layout=layout_update,
                font_size=font_size,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )

    def make_pie_chart(
        self,
        df: Optional[D] = None,
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
        font_size: Optional[int] = None,
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
        group_cols = self._resolve_group_cols_arg(group_cols)
        if group_cols and not group_modes:
            group_modes = (GroupMode.ROWS, GroupMode.COLUMNS)
        layout_update = dict()
        if layout is not None:
            layout_update.update(layout)
        update_traces = dict(
            textposition="auto",
            textinfo="label+value+percent",
        )
        if traces_settings is not None:
            update_traces.update(traces_settings)
        return make_pie_chart(
            df=df,
            x_col=x_col,
            y_col=y_col,
            group_cols=group_cols,
            group_modes=group_modes,
            title=title,
            labels=labels,
            font_size=font_size,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout_update,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=update_traces,
            output=output,
            **kwargs,
        )

    def make_ranking_table(
        self,
        /,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_column: Optional[str | Tuple[str, ...]] = None,
        sort_order: Literal[
            SortOrder.DESCENDING, SortOrder.ASCENDING
        ] = SortOrder.DESCENDING,
        top_k: Optional[int] = None,
        drop_cols: Optional[str | Iterable[str]] = None,
    ) -> D:
        """Sorts the values

        Args:
            group_cols:
                Ranking tables for groups will be concatenated side-by-side. Defaults to the default groupby.
                To fully prevent grouping, pass False or a falsy value except None.
            sort_column: By which column to rank. Defaults to the :attr:`dimension_column`.
            sort_order: Defaults to "descending", i.e., the highest values will be ranked first.
            top_k: The number of top ranks to retain. Defaults to 50. Pass None to retain all.

        Returns:

        """

        def make_table(
            df,
            drop_columns: Optional[List[str]] = None,
            make_int_nullable: bool = False,
        ):
            if top_k:
                if top_k > 0:
                    ranking = df.nlargest(top_k, sort_column, keep=keep)
                else:
                    ranking = df.nsmallest(-top_k, sort_column, keep=keep)
            else:
                ranking = df.sort_values(sort_column, ascending=ascending)
            ranking = ranking.reset_index()
            if drop_columns:
                ranking = ranking.drop(columns=drop_columns)
            ranking.index = (ranking.index + 1).rename("rank")
            if make_int_nullable:
                conversion = {
                    col: "Int64"
                    for col, dtype in ranking.dtypes.items()
                    if pd.api.types.is_integer_dtype(dtype)
                }
                if conversion:
                    ranking = ranking.astype(conversion)
            return ranking

        if sort_order == SortOrder.DESCENDING:
            ascending = False
        elif sort_order == SortOrder.ASCENDING:
            ascending = True
        else:
            raise ValueError(
                f"sort_order must be 'descending' or 'ascending', not {sort_order}"
            )
        keep = "last" if ascending else "first"
        if sort_column is None:
            sort_column = self.dimension_column
        group_cols = self._resolve_group_cols_arg(group_cols)
        drop_cols = self._resolve_group_cols_arg(drop_cols)
        df = self._combine_results(group_cols)
        if not group_cols:
            return make_table(df)
        ranking_groups = {
            group: make_table(df, group_cols + drop_cols, make_int_nullable=True)
            for group, df in df.groupby(group_cols)
        }
        return pd.concat(ranking_groups, names=group_cols, axis=1)

    def plot(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
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
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def plot_grouped(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = None,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        group_cols = self._resolve_group_cols_arg(group_cols)
        combined_result = self._combine_results(group_cols=group_cols)
        if not group_cols:
            return self.make_bar_plot(
                df=combined_result,
                group_cols=group_cols,
                group_modes=group_modes,
                title=title,
                labels=labels,
                hover_data=hover_data,
                height=height,
                width=width,
                layout=layout,
                font_size=font_size,
                x_axis=x_axis,
                y_axis=y_axis,
                color_axis=color_axis,
                traces_settings=traces_settings,
                output=output,
                **kwargs,
            )
        else:
            if "y_col" in kwargs:
                y_col = kwargs.pop("y_col")
            else:
                y_col = group_cols[-1]
            return self.make_bubble_plot(
                df=combined_result,
                y_col=y_col,
                title=title,
                hover_data=hover_data,
                height=height,
                width=width,
                layout=layout,
                font_size=font_size,
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

    def _sort_combined_result(
        self,
        combined_result: D,
        group_cols: List[str],
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ):
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


class Counts(Result):
    pass


class CadenceCounts(Counts):
    @property
    def x_column(self) -> str:
        return self.formatted_column

    def plot(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
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
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            color_discrete_map=CADENCE_COLORS,
            **kwargs,
        )

    def plot_grouped(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = None,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        group_cols = self._resolve_group_cols_arg(group_cols)
        combined_result = self._combine_results(group_cols=group_cols)
        return self.make_pie_chart(
            df=combined_result,
            group_cols=group_cols,
            group_modes=group_modes,
            title=title,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            color_discrete_map=CADENCE_COLORS,
            **kwargs,
        )


class Durations(Result):
    pass


class NgramTable(Result):
    """A side-by-side concatenation of a feature with one or several shifted version of itself, so that each row
    contains both the original values and those of the n-1 following rows, concatenated on the right.
    This table keeps full flexibility in terms of how you want to create :class:`NgramTuples` from it.
    """

    @cached_property
    def ngram_levels(self) -> List[str]:
        return list(self.df.columns.levels[0])

    def _add_context_columns(
        self,
        df: D,
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
    ) -> D:
        """Concatenates requested context columns to the left side of the computed ngram_table or
        :obj:`NgramTuples`. If terminals are being dropped, this is accomplished by a join to not
        restore the dropped rows.
        """
        context_df = self._get_context_df(context_columns)
        if terminal_symbols == "DROP":
            return context_df.join(df, how="right")
        return pd.concat([context_df, df], axis=1)

    @cache
    def _get_component_missing_mask(
        self,
        level: str,
        columns: Optional[str, Tuple[str, ...]] = None,
    ) -> S:
        """Returns a boolean mask in which those entries are True at which entire rows consist of missing values for a
        a given n-gram component as defined by level and columns.
        This method is cached and calls the cached :meth:`_subselect_component_columns`.
        """
        selection = self._subselect_component_columns(level, columns)
        if isinstance(selection, pd.Series):
            return selection.isna()
        return selection.isna().all(axis=1)

    def _combine_results(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ) -> D:
        raise NotImplementedError(
            "NgramTable does not support this action. Try one of .get_ngram_tuples(), "
            ".get_bigram_tuples(), .get_ngram_table(), .get_bigram_table(), .get_transitions()"
        )

    def compute_information_gain(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[str | bool] = None,
        fillna: Optional[Hashable] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        reverse: bool = False,
    ) -> S | float:
        """Computes the gain in information about  (reduction in entropy of) the consequent from knowing the antecedent.
        This can be interpreted as measure of how much we know on average about the consequent given an antecedent.
        This method provides a shortcut to calling :attr:`TransitionTable.compute_information_gain` on the result of
        :meth:`get_transitions`.


        Args:
            gram_component_columns:
                One or several column specifications. If zero or one are passed, the same specification will be used
                for each n-gram component. The number of specifications can be at most the number of components ('a',
                'b', etc.) that this NgramTable contains. Each specification can be None (default feature columns),
                a single column name, or a tuple of column names.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                components are to be part of the antecedent (context, left) and the consequent (target, right).
                Defaults to -1, i.e. the last component is used as consequent. This is a useful default for
                evaluations where the (n-1) previous components are the context for predicting the next one.
                If you pass an integer within ±[1, n-1], the split will be performed after the indicated component
                and any side (left or ride) that includes only a single component will contain single values (tuples or
                strings). To override this automatic behaviour, you may instead pass a tuple that indicates the split in
                terms of column names ('a, 'b', etc.) in the way that tuples become tuples, individual strings not.
                Example: (('a', 'b'), 'c') corresponds to the default behaviour, where the left side has tuples, the
                right side not. (('a', 'b'), ('c')), on the other hand, would turn the right-hand side into 1-element
                tuples, too.
            join_str:
                Parameter passed to :meth:`make_ngram_table`. It determines whether the antecedent and consequent
                columns contain [tuples of] tuples (the default) or [tuples of] strings. If n == 2, each cell is of
                type (tuple|str), if n > 2, it's Tuple[(tuple|str)].
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last bigram in a sequence ends on (a tuple or string concatenation of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                In the case of bigrams, there is only one consequent component. However, when dealing with bigrams
                constructed by splitting higher-level grams, you can either specify a single value to be used for all
                consequent components (b, c, ...) or a tuple of (n-1) values to obtain different behaviours.  For each
                component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            group_cols: Defines the groups for which to compute the information gain.
            reverse: Reverse the argument: How much more do we know about the antecedent when we know the consequent?

        Returns:
            If group_cols is None or empty or resolves to empty (the default when no groupers have been applied),
            the resulting value is a float expressing the difference in entropy. Otherwise, when a grouping is
            performed, the result is a Series of floats.
        """
        transitions = self.get_transitions(
            *ngram_component_columns,
            split=split,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
            group_cols=group_cols,
        )
        return transitions.compute_information_gain(
            group_cols=group_cols, reverse=reverse
        )

    def _get_context_df(
        self,
        context_columns: Optional[str, Tuple[str, ...]] = None,
    ) -> D:
        """Retrieve context columns to be included in an n-grams table."""
        if context_columns is True or context_columns is None:
            if not self._auxiliary_column_names:
                raise NotImplementedError(
                    f"The _auxiliary_column_names should have been set to the names of the original Feature's context "
                    f"columns by the object that created this {self.name}."
                )
            context_columns = self._auxiliary_column_names
        elif isinstance(context_columns, str):
            context_columns = [context_columns]
        return self._subselect_component_columns(
            level="a", columns=context_columns, droplevel=True
        )

    def get_default_analysis(self) -> Transitions:
        return self.get_transitions()

    def get_grouping_levels(
        self, smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE
    ) -> List[str]:
        # do not follow the behaviour of Result.get_grouping_levels, which assumes that the last one or two levels
        # are value_column or [value_column, formatted_column] and omits these
        return DimcatResource.get_grouping_levels(self, smallest_unit=smallest_unit)

    @cache
    def _get_transitions(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[str | bool] = None,
        fillna: Optional[Hashable] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        group_cols: Optional[str | Iterable[str]] = UnitOfAnalysis.GROUP,
    ) -> D:
        """Get a Series that counts for each antecedent (context) the number of transitions to each possible consequent
        (following element, target).

        Args:
            gram_component_columns:
                One or several column specifications. If zero or one are passed, the same specification will be used
                for each n-gram component. The number of specifications can be at most the number of components ('a',
                'b', etc.) that this NgramTable contains. Each specification can be None (default feature columns),
                a single column name, or a tuple of column names.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                components are to be part of the antecedent (context, left) and the consequent (target, right).
                Defaults to -1, i.e. the last component is used as consequent. This is a useful default for
                evaluations where the (n-1) previous components are the context for predicting the next one.
                If you pass an integer within ±[1, n-1], the split will be performed after the indicated component
                and any side (left or ride) that includes only a single component will contain single values (tuples or
                strings). To override this automatic behaviour, you may instead pass a tuple that indicates the split in
                terms of column names ('a, 'b', etc.) in the way that tuples become tuples, individual strings not.
                Example: (('a', 'b'), 'c') corresponds to the default behaviour, where the left side has tuples, the
                right side not. (('a', 'b'), ('c')), on the other hand, would turn the right-hand side into 1-element
                tuples, too.
            join_str:
                Parameter passed to :meth:`make_ngram_table`. It determines whether the antecedent and consequent
                columns contain [tuples of] tuples (the default) or [tuples of] strings. If n == 2, each cell is of
                type (tuple|str), if n > 2, it's Tuple[(tuple|str)].
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last bigram in a sequence ends on (a tuple or string concatenation of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                In the case of bigrams, there is only one consequent component. However, when dealing with bigrams
                constructed by splitting higher-level grams, you can either specify a single value to be used for all
                consequent components (b, c, ...) or a tuple of (n-1) values to obtain different behaviours.  For each
                component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            group_cols: Determines based for which grouping the transitions should be counted and normalized.

        Returns:
            Dataframe with columns 'count' and 'proportion', showing each (n-1) previous elements (index level 0),
            the count and proportion of transitions to each possible following element (index level 1).
        """
        self._check_ngram_component_columns_arg(ngram_component_columns)
        bigrams = self.make_bigram_df(
            *ngram_component_columns,
            split=split,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
        )
        group_cols = self._resolve_group_cols_arg(group_cols)
        if len(group_cols) == 0 or not group_cols[-1] == "antecedent":
            group_cols.append("antecedent")
        gpb = bigrams.groupby(group_cols).consequent
        counts = gpb.value_counts()
        proportion = gpb.value_counts(normalize=True)
        proportion_str = turn_proportions_into_percentage_strings(proportion)
        return pd.concat([counts, proportion, proportion_str], axis=1)

    def get_transitions(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[str | bool] = None,
        fillna: Optional[Hashable] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        feature_columns: Optional[Tuple[str, str]] = None,
    ) -> Transitions:
        """Get a Series that counts for each context the number of transitions to each possible following element.

        Args:
            gram_component_columns:
                One or several column specifications. If zero or one are passed, the same specification will be used
                for each n-gram component. The number of specifications can be at most the number of components ('a',
                'b', etc.) that this NgramTable contains. Each specification can be None (default feature columns),
                a single column name, or a tuple of column names.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                components are to be part of the antecedent (context, left) and the consequent (target, right).
                Defaults to -1, i.e. the last component is used as consequent. This is a useful default for
                evaluations where the (n-1) previous components are the context for predicting the next one.
                If you pass an integer within ±[1, n-1], the split will be performed after the indicated component
                and any side (left or ride) that includes only a single component will contain single values (tuples or
                strings). To override this automatic behaviour, you may instead pass a tuple that indicates the split in
                terms of column names ('a, 'b', etc.) in the way that tuples become tuples, individual strings not.
                Example: (('a', 'b'), 'c') corresponds to the default behaviour, where the left side has tuples, the
                right side not. (('a', 'b'), ('c')), on the other hand, would turn the right-hand side into 1-element
                tuples, too.
            join_str:
                Parameter passed to :meth:`make_ngram_table`. It determines whether the antecedent and consequent
                columns contain [tuples of] tuples (the default) or [tuples of] strings. If n == 2, each cell is of
                type (tuple|str), if n > 2, it's Tuple[(tuple|str)].
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last bigram in a sequence ends on (a tuple or string concatenation of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                In the case of bigrams, there is only one consequent component. However, when dealing with bigrams
                constructed by splitting higher-level grams, you can either specify a single value to be used for all
                consequent components (b, c, ...) or a tuple of (n-1) values to obtain different behaviours.  For each
                component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            group_cols: Determines based for which grouping the transitions should be counted and normalized.
            feature_columns: Defaults to ["antecedent", "consequent"]. Pass a List with two strings to change.

        Returns:
            Dataframe with columns 'count' and 'proportion', showing each (n-1) previous elements (index level 0),
            the count and proportion of transitions to each possible following element (index level 1).
        """
        transitions = self._get_transitions(
            *ngram_component_columns,
            split=split,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
            group_cols=group_cols,
        )
        if feature_columns:
            feature_columns = list(feature_columns)
            level_names = dict(zip(("antecedent", "consequent"), feature_columns))
            transitions.index.set_names(level_names, inplace=True)
        else:
            feature_columns = ["antecedent", "consequent"]
        new_result = Transitions.from_resource_and_dataframe(
            self,
            transitions,
            feature_columns=feature_columns,
            dimension_column="count",
        )
        return new_result

    @cache
    def make_bigram_df(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[bool | str | Tuple[str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> D:
        """Reduce the selected specified n-gram components to two columns, called 'antecedent' and 'consequent'.
        For NgramTables produced by a :obj:`BigramAnalyzer` or by an :obj:`NgramAnalyzer(n=2) <NgramAnalyzer>`, the
        result is equivalent to :attr:`make_ngram_table`, just with renamed columns. For higher n, the components are
        split split into an antecedent and a consequent part based on the ``split`` parameter.
        as tuples (default, where ``join_str`` is None) or strings.
        If the result corresponds to n=2 (i.e., neither antecedent nor consequent combine n-gram components), the
        columns contain strings or tuples (depending on whether join_str is specified or not); otherwise, both column
        contain tuples thereof.

        Args:
            gram_component_columns:
                One or several column specifications. If zero or one are passed, the same specification will be used
                for each n-gram component. The number of specifications can be at most the number of components ('a',
                'b', etc.) that this NgramTable contains. Each specification can be None (default feature columns),
                a single column name, or a tuple of column names.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                components are to be part of the antecedent (context, left) and the consequent (target, right).
                Defaults to -1, i.e. the last component is used as consequent. This is a useful default for
                evaluations where the (n-1) previous components are the context for predicting the next one.
                If you pass an integer within ±[1, n-1], the split will be performed after the indicated component
                and any side (left or ride) that includes only a single component will contain single values (tuples or
                strings). To override this automatic behaviour, you may instead pass a tuple that indicates the split in
                terms of column names ('a, 'b', etc.) in the way that tuples become tuples, individual strings not.
                Example: (('a', 'b'), 'c') corresponds to the default behaviour, where the left side has tuples, the
                right side not. (('a', 'b'), ('c')), on the other hand, would turn the right-hand side into 1-element
                tuples, too.
            join_str:
                Parameter passed to :meth:`make_ngram_table`. It determines whether the antecedent and consequent
                columns contain [tuples of] tuples (the default) or [tuples of] strings. If n == 2, each cell is of
                type (tuple|str), if n > 2, it's Tuple[(tuple|str)].
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last bigram in a sequence ends on (a tuple or string concatenation of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                In the case of bigrams, there is only one consequent component. However, when dealing with bigrams
                constructed by splitting higher-level grams, you can either specify a single value to be used for all
                consequent components (b, c, ...) or a tuple of (n-1) values to obtain different behaviours.  For each
                component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            context_columns:
                Columns preceding the bigram columns for context, such as measure numbers etc. Pass True to use the
                default context columns or one or several column names to subselect.


        Returns:
            Like :meth:`make_ngram_tuples`, but condensed to two columns.
        """
        self._check_ngram_component_columns_arg(ngram_component_columns)
        ngram_table = self.make_ngram_df(
            *ngram_component_columns,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
        )
        component_names = ngram_table.columns.to_list()
        n_components = len(component_names)
        if isinstance(split, int):
            if not 0 < abs(split) < n_components:
                raise ValueError(
                    f"split must be within ±[1, n-1], not {split} for n={n_components}"
                )
            left_cols, right_cols = component_names[:split], component_names[split:]
            if len(left_cols) == 1:
                left_cols = left_cols[0]
            if len(right_cols) == 1:
                right_cols = right_cols[0]
        else:
            try:
                left_cols, right_cols = split
            except ValueError:
                raise ValueError(
                    f"Since you are requesting bigrams, you need to distribute the components {component_names} on "
                    f"two sides, each of which can be a list or tuple, or a single character. Got: {split}."
                )
        left_tuples = not isinstance(left_cols, str)
        right_tuples = not isinstance(right_cols, str)
        component_selection = []
        if left_tuples:
            component_selection.extend(left_cols)
        else:
            component_selection.append(left_cols)
        if right_tuples:
            component_selection.extend(right_cols)
        else:
            component_selection.append(right_cols)
        if component_selection != component_names:
            if set(component_selection) == set(component_names):
                self.logger.warning(
                    f"The specified split {split} does not bring the gram components ({component_names}) in the "
                    f"correct order."
                )
            else:
                raise ValueError(
                    f"The specified split {split} does not contain exactly the gram components "
                    f"{component_names}."
                )
        if left_tuples:
            left_side = pd.Series(
                ngram_table[list(left_cols)].itertuples(index=False, name=None),
                index=ngram_table.index,
                name="antecedent",
            )
        else:
            left_side = ngram_table[left_cols].rename("antecedent")
        if right_tuples:
            right_side = pd.Series(
                ngram_table[list(right_cols)].itertuples(index=False, name=None),
                index=ngram_table.index,
                name="consequent",
            )
        else:
            right_side = ngram_table[right_cols].rename("consequent")
        result = pd.concat([left_side, right_side], axis=1)
        if context_columns:
            result = self._add_context_columns(
                result, context_columns, terminal_symbols
            )
        return result

    def _check_ngram_component_columns_arg(self, gram_component_columns):
        for component_columns in gram_component_columns:
            if component_columns is not None and not isinstance(
                component_columns, (str, tuple)
            ):
                raise TypeError(
                    f"Component columns must be None, a string or a tuple of strings, got {type(component_columns)}"
                )

    def make_bigram_table(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[bool | str | Tuple[str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> Self:
        """Returns the result of :meth:`make_bigram_df` as a new :class:`NgramTable` object."""
        df = self.make_bigram_df(
            *ngram_component_columns,
            split=split,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
            context_columns=context_columns,
        )
        return self.from_resource_and_dataframe(
            resource=self,
            df=df,
        )

    def make_bigram_tuples(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        split: int | Tuple[str_or_sequence, str_or_sequence] = -1,
        join_str: Optional[bool | str | Tuple[str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        drop_identical: bool = False,
        n_gram_column_name: str = "n_gram",
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> NgramTuples:
        """Get a Resource with a single column that contains bigram tuples, where each element is a tuple or string
        based on the specified (or default) columns. If this object represents trigrams or higher, it is always
        tuples of tuples (never of strings). See :meth:`make_bigram_table` for details.

        Args:
            gram_component_columns:
                One or several column specifications. If zero or one are passed, the same specification will be used
                for each n-gram component. The number of specifications can be at most the number of components ('a',
                'b', etc.) that this NgramTable contains. Each specification can be None (default feature columns),
                a single column name, or a tuple of column names.
            split:
                Relevant only for NgramAnalyzer with n > 2: Then the value can be modified to decide how many
                components are to be part of the antecedent (context, left) and the consequent (target, right).
                Defaults to -1, i.e. the last component is used as consequent. This is a useful default for
                evaluations where the (n-1) previous components are the context for predicting the next one.
                If you pass an integer within ±[1, n-1], the split will be performed after the indicated component
                and any side (left or ride) that includes only a single component will contain single values (tuples or
                strings). To override this automatic behaviour, you may instead pass a pair that indicates the split in
                terms of column names ('a, 'b', etc.) in the way that tuples become tuples, individual strings not.
                Example: (('a', 'b'), 'c') corresponds to the default behaviour, where the left side has tuples, the
                right side not. (('a', 'b'), ('c')), on the other hand, would turn the right-hand side into 1-element
                tuples, too.
            join_str:
                Parameter passed to :meth:`make_ngram_table`. It determines whether the antecedent and consequent
                columns contain [tuples of] tuples (the default) or [tuples of] strings. If n == 2, each cell is of
                type (tuple|str), if n > 2, it's Tuple[(tuple|str)].
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last bigram in a sequence ends on (a tuple or string concatenation of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                In the case of bigrams, there is only one consequent component. However, when dealing with bigrams
                constructed by splitting higher-level grams, you can either specify a single value to be used for all
                consequent components (b, c, ...) or a tuple of (n-1) values to obtain different behaviours.  For each
                component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            drop_identical: Pass True to drop all tuples where left and right gram are identical.
            n_gram_column_name: Name of the value_column in the resulting :class:`NgramTuples` object.
            context_columns:
                Columns preceding the bigram columns for context, such as measure numbers etc. Pass True to use the
                default context columns or one or several column names to subselect.


        Returns:

        """

        self._check_ngram_component_columns_arg(ngram_component_columns)
        table = self.make_bigram_df(
            *ngram_component_columns,
            split=split,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
        )
        return self._make_tuples_from_table(
            table, terminal_symbols, drop_identical, n_gram_column_name, context_columns
        )

    @cache
    def _make_ngram_component(
        self,
        level: str,
        columns: Optional[str, Tuple[str, ...]] = None,
        join_str: Optional[str | bool] = None,
        fillna: Optional[Hashable] = None,
        terminal_symbols: Optional[TerminalSymbol | Hashable] = None,
    ) -> S:
        """Create one of the components for :attr:`make_ngram_table` as a subset of the NgramTable with the requested
        columns (if specified) for one of the n-gram levels 'a', 'b', etc. Such components, concatenated sideways
        make up the n_gram table.
        """
        selection = self._subselect_component_columns(level, columns)
        return_tuples = not isinstance(selection, pd.Series)
        if fillna is not None:
            selection = selection.fillna(fillna)
        if return_tuples:
            selection = pd.Series(
                selection.itertuples(index=False, name=None), index=selection.index
            )
            if join_str is not None:
                if not isinstance(join_str, str):
                    if join_str is True:
                        join_str = ", "
                    elif join_str is False:
                        join_str = ""
                    else:
                        raise TypeError(
                            f"join_str must be a string or a boolean, got {join_str!r} ({type(join_str)})"
                        )
                selection = ms3.transform(selection, tuple2str, join_str=join_str)
        elif join_str is not None:
            selection = selection.astype("string")
        result = selection.rename(level)

        # deal with terminal grams if required
        if terminal_symbols is None or terminal_symbols == TerminalSymbol.DROP:
            return result
        terminal_mask = self._get_component_missing_mask(level, columns)
        replace_terminals = terminal_mask.any()  # false if nothing to replace
        if not replace_terminals:
            return result
        if terminal_symbols == TerminalSymbol.DEFAULT:
            replace_with = get_setting("default_terminal_symbol")
        elif terminal_symbols == TerminalSymbol.NA:
            replace_with = None
        else:
            # at this point, all other members of TerminalSymbol have been dealt with, so that any other will be
            # accepted as fill value. If DROP, replace_terminals should be False, so we never get here.
            replace_with = terminal_symbols
        if replace_with is None:
            return result.where(~terminal_mask)
        if join_str is None and return_tuples:
            replace_with = (replace_with,) * len(columns)
        replacement_series = pd.Series([replace_with] * len(result), index=result.index)
        return result.where(~terminal_mask, other=replacement_series)

    @cache
    def make_ngram_df(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        n: Optional[int] = None,
        join_str: Optional[bool | str | Tuple[bool | str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> D:
        """Reduce the selected columns for the n first n-gram levels a, b, ... so that the resulting dataframe
        contains n columns, each of which contains tuples or strings. You may pass several column specifications to
        create n-gram components from differing columns, e.g. to evaluate how well one feature predicts another.

        Args:
            gram_component_columns:
                One or several column specifications. If one (or only the default, None) is passed, the same
                specification will be used for each n-gram component, otherwise the number of specifications must
                match ``n``. Each specification can be None (default feature columns), a single column name, or a
                tuple of column names.
            n:
                Only make columns for the first n n-gram components. If None, use all n-gram levels. Minimum is 2,
                maximum is the number of n-gram levels determined by the :obj:`NgramAnalyzer` used to create the n-gram
                table.
            join_str:
                By default (None), the resulting columns contain tuples. If you want them to contain strings,
                you may pass a single specification (bool or string) to use for all n-gram components, or a tuple
                thereof to use different specifications for each component. True stands for concatenating the tuple
                values for a given n-gram component separated by ", " -- yielding strings that look like tuples without
                parentheses. False stands for concatenating without any value in-between the values. If a string is
                passed, it will be used as the separator between the tuple values.
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``
            terminal_symbols:
                By default, the last n-1 n-grams in a sequence end on (tuples or string concatenations of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                You can either specify a single value to be used for all consequent components (b, c, ...) or a tuple
                of (n-1) values to obtain different behaviours. In the case of bigrams, there is only one consequent
                component. For each component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            context_columns:
                Columns preceding the bigram columns for context, such as measure numbers etc. Pass True to use the
                default context columns or one or several column names to subselect.

        Returns:

        """
        # region prepare parameters
        n_level_specs = len(ngram_component_columns)
        if n is not None:
            n = int(n)
            assert (
                1 < n <= len(self.ngram_levels)
            ), f"n needs to be between 2 and {len(self.ngram_levels)}, got {n}"
            if n_level_specs > 1:
                if n != n_level_specs:
                    raise ValueError(
                        f"When n is specified, the number of column specifications needs to be either zero, one or n.\n"
                        f"n={n}, but {n_level_specs} column specifications were passed: {ngram_component_columns}"
                    )
            selected_levels = self.ngram_levels[:n]
        else:
            selected_levels = self.ngram_levels
        n = len(selected_levels)
        if len(ngram_component_columns) == 0:
            component_columns = [self.feature_columns] * n
        elif len(ngram_component_columns) == 1:
            component_columns = [ngram_component_columns[0]] * n
        else:
            component_columns = ngram_component_columns
        # ensure that all collections are tuples
        component_columns = [
            arg if arg is None or isinstance(arg, str) else tuple(arg)
            for arg in component_columns
        ]

        if isinstance(join_str, tuple):
            assert (
                len(join_str) == n
            ), f"If you specify 'join_str' as a tuple it needs to have n ({n}) elements, not {len(join_str)}."
            join_strings = join_str
        else:
            join_strings = repeat(join_str)
        if isinstance(fillna, tuple):
            assert (
                len(fillna) == n
            ), f"If you specify 'fillna' as a tuple it needs to have n ({n}) elements, not {len(fillna)}."
            fillna_values = fillna
        else:
            fillna_values = repeat(fillna)
        drop_terminals_for_components = []
        if isinstance(terminal_symbols, tuple):
            assert len(terminal_symbols) == n - 1, (
                f"If you specify 'terminal_symbols' as a tuple it needs to have n-1 ({n - 1}) elements, not "
                f"{len(terminal_symbols)}."
            )
            terminal_symbols = (None,) + terminal_symbols
            drop_terminals_for_components = [
                level
                for level, terminal_symbol in zip(selected_levels, terminal_symbols)
                if terminal_symbol == TerminalSymbol.DROP
            ]
        else:
            if terminal_symbols == TerminalSymbol.DROP:
                drop_terminals_for_components = selected_levels[1:]
            terminal_symbols = repeat(terminal_symbols)
        # endregion prepare parameters

        ngram_components = []
        for level, columns, join_string, fillna_val, terminal in zip(
            selected_levels,
            component_columns,
            join_strings,
            fillna_values,
            terminal_symbols,
        ):
            ngram_components.append(
                self._make_ngram_component(
                    level, columns, join_string, fillna_val, terminal
                )
            )

        if context_columns:
            ngram_components = [
                self._get_context_df(context_columns)
            ] + ngram_components
        result = pd.concat(ngram_components, axis=1)
        if drop_terminals_for_components:
            drop_mask = pd.Series(False, index=result.index)
            for level, columns in zip(selected_levels, component_columns):
                if level in drop_terminals_for_components:
                    drop_mask |= self._get_component_missing_mask(level, columns)
            result = result[~drop_mask]
        return result

    def make_ngram_table(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        n: Optional[int] = None,
        join_str: Optional[bool | str | Tuple[bool | str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> Self:
        """Returns the result of :attr:`make_ngram_df` as a new :class:`NgramTable` object."""
        df = self.make_ngram_df(
            *ngram_component_columns,
            n=n,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
            context_columns=context_columns,
        )
        return self.__class__.from_resource_and_dataframe(resource=self, df=df)

    def make_ngram_tuples(
        self,
        *ngram_component_columns: Optional[str | Tuple[str, ...]],
        n: Optional[int] = None,
        join_str: Optional[bool | str | Tuple[str, ...]] = None,
        fillna: Optional[Hashable | Tuple[Hashable, ...]] = None,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        drop_identical: bool = False,
        n_gram_column_name: str = "n_gram",
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> NgramTuples:
        """Get a Resource with a single column that contains n-gram tuples, where each element is a tuple or string
        based on the specified (or default) columns.

        Args:
            gram_component_columns:
                One or several column specifications. If one (or only the default, None) is passed, the same
                specification will be used for each n-gram component, otherwise the number of specifications must
                match ``n``. Each specification can be None (default feature columns), a single column name, or a
                tuple of column names.
            n:
                Make tuples from the first n n-gram components only. If None, use all n-gram levels. Minimum is 2,
                maximum is the number of n-gram levels determined by the :obj:`NgramAnalyzer` used to create the n-gram
                table.
            join_str:
                By default (None), the resulting columns contain tuples. If you want them to contain strings,
                you may pass a single specification (bool or string) to use for all n-gram components, or a tuple
                thereof to use different specifications for each component. True stands for concatenating the tuple
                values for a given n-gram component separated by ", " -- yielding strings that look like tuples without
                parentheses. False stands for concatenating without any value in-between the values. If a string is
                passed, it will be used as the separator between the tuple values.
            fillna:
                Pass a value to replace all missing values with it. Pass a tuple tuple of n values to fill missing
                values differently for the n components (e.g. (None, '') to fill missing values with empty strings
                only for the second n-gram components). "" is often a good choice for components for which ``join_str``
                is specified to avoid strings looking like ``"value<NA>"``.
            terminal_symbols:
                By default, the last n-1 n-grams in a sequence end on (tuples or string concatenations of) missing
                values. These rows can either be dropped, or the missing components replaced with a terminal symbol.
                You can either specify a single value to be used for all consequent components (b, c, ...) or a tuple
                of (n-1) values to obtain different behaviours. In the case of bigrams, there is only one consequent
                component. For each component to be left untouched, pass None (the default). To drop terminal rows for
                a component, pass "DROP". To replace all terminal cells with pd.NA (independent of whether they would
                be tuples or strings), pass "NA". To replace them with the default_terminal_symbol, pass "DEFAULT".
                Or, pass a string or other Hashable to replace terminals with that string. In the latter two cases,
                the terminal cells will be tuples of terminal strings if ``join_str`` is None, or strings otherwise.
            drop_identical: Pass True to drop all tuples where all elements are identical.
            n_gram_column_name: Name of the value_column in the resulting :class:`NgramTuples` object.
            context_columns:
                Columns preceding the bigram columns for context, such as measure numbers etc. Pass True to use the
                default context columns or one or several column names to subselect.


        Returns:

        """
        self._check_ngram_component_columns_arg(ngram_component_columns)
        table = self.make_ngram_df(
            *ngram_component_columns,
            n=n,
            join_str=join_str,
            fillna=fillna,
            terminal_symbols=terminal_symbols,
        )
        return self._make_tuples_from_table(
            table, terminal_symbols, drop_identical, n_gram_column_name, context_columns
        )

    def make_ranking_table(
        self,
        /,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_column: Optional[str | Tuple[str, ...]] = None,
        sort_order: Literal[
            SortOrder.DESCENDING, SortOrder.ASCENDING
        ] = SortOrder.DESCENDING,
        top_k: Optional[int] = None,
        drop_cols: Optional[str | Iterable[str]] = None,
    ):
        """Shortcut for creating the default :class:`NgramTuples` object and calling
        :meth:`~NgramTuples.make_ranking_table` on it. For more fine-grained control on the n-gram tuples,
        use :meth:`make_ngram_tuples` or :meth:`make_bigram_tuples`.
        """
        n_gram_tuples = self.make_ngram_tuples()
        n_gram_counts = n_gram_tuples.apply_step("Counter")
        return n_gram_counts.make_ranking_table(
            group_cols=group_cols,
            sort_column=sort_column,
            sort_order=sort_order,
            top_k=top_k,
            drop_cols=drop_cols,
        )

    def _make_tuples_from_table(
        self,
        table: D,
        terminal_symbols: Optional[
            TerminalSymbol | Hashable, Tuple[TerminalSymbol | Hashable, ...]
        ] = None,
        drop_identical: bool = False,
        n_gram_column_name: str = "n_gram",
        context_columns: Optional[Literal[True], str, Tuple[str, ...]] = None,
    ) -> NgramTuples:
        """Boilerplate used by :meth:`make_ngram_tuples` and :meth:`make_bigram_tuples`."""
        df = table.apply(tuple, axis=1).to_frame(n_gram_column_name)
        if drop_identical:
            keep_mask = df[n_gram_column_name].map(lambda tup: len(set(tup)) > 1)
            df = df[keep_mask]
        if context_columns:
            df = self._add_context_columns(df, context_columns, terminal_symbols)
        result = NgramTuples.from_resource_and_dataframe(
            self,
            df,
            value_column=n_gram_column_name,
        )
        result.formatted_column = None
        return result

    @overload
    def _subselect_component_columns(
        self, level: str, columns: str, droplevel: bool
    ) -> S:
        ...

    @overload
    def _subselect_component_columns(
        self, level: str, columns: Tuple[str, ...], droplevel: bool
    ) -> D:
        ...

    @overload
    def _subselect_component_columns(
        self, level: str, columns: Literal[None], droplevel: bool
    ) -> D | S:
        ...

    @cache
    def _subselect_component_columns(
        self,
        level: str,
        columns: Optional[str, Tuple[str, ...]] = None,
        droplevel: bool = True,
    ) -> D | S:
        """Retrieve the specified columns for the specified n-gram level ('a, 'b', etc.) from the NgramTable."""
        return_series = False
        if columns is None:
            columns = self.feature_columns
            if len(columns) == 1:
                return_series = True
        elif isinstance(columns, str):
            return_series = True
            columns = [columns]
        else:
            columns = list(columns)
        column_names = list(product([level], columns))
        missing = [col for col in column_names if col not in self.df.columns]
        n_missing = len(missing)
        if n_missing:
            if n_missing == len(column_names):
                msg = f"None of the requested columns {columns} are present in the NgramTable."
            else:
                msg = f"The following columns are not present in the NgramTable: {missing}"
            raise ValueError(msg)
        if return_series:
            selection = self.df.loc[:, column_names[0]]
        else:
            selection = self.df.loc[:, column_names]
        if droplevel:
            if return_series:
                selection = selection.rename(columns[0])
            else:
                selection = selection.droplevel(0, axis=1)
        else:
            selection = selection.copy()
        return selection

    def plot(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        transitions = self.get_transitions(join_str=True, group_cols=None)
        return transitions.plot(
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def plot_grouped(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        transitions = self.get_transitions(
            join_str=True,
            group_cols=UnitOfAnalysis.GROUP,
        )
        return transitions.plot_grouped(
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )


class NgramTuples(Result):
    """Result that has a :attr:`value_column` containing tuples and no `dimension_column`."""

    _default_analyzer = "Counter"

    def make_ranking_table(
        self,
        /,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_column: Optional[str | Tuple[str, ...]] = None,
        sort_order: Literal[
            SortOrder.DESCENDING, SortOrder.ASCENDING
        ] = SortOrder.DESCENDING,
        top_k: Optional[int] = None,
        drop_cols: Optional[str | Iterable[str]] = None,
    ):
        n_gram_counts = self.apply_step("Counter")
        return n_gram_counts.make_ranking_table(
            group_cols=group_cols,
            sort_column=sort_column,
            sort_order=sort_order,
            top_k=top_k,
            drop_cols=drop_cols,
        )

    def plot(self):
        raise NotImplementedError

    def plot_grouped(self):
        raise NotImplementedError


class Transitions(Result):
    class Schema(Result.Schema):
        feature_columns = mm.fields.List(
            mm.fields.Str(), required=True, validate=mm.validate.Length(min=2, max=2)
        )

    def __init__(
        self,
        analyzed_resource: DimcatResource,
        feature_columns: List[str, str],
        value_column: Optional[str] = None,
        dimension_column: Optional[str] = None,
        formatted_column: Optional[str] = None,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            analyzed_resource=analyzed_resource,
            value_column=value_column,
            dimension_column=dimension_column,
            formatted_column=formatted_column,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        self._feature_columns = feature_columns

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_columns)

    @feature_columns.setter
    def feature_columns(self, feature_columns: List[str]):
        if not isinstance(feature_columns, list):
            raise TypeError(f"Expected a list, got {feature_columns!r}")
        assert len(feature_columns) == 2, (
            "Expects exactly 2 column names, one for the antecedent, one for the "
            "consequent"
        )
        self._feature_columns = feature_columns

    @property
    def x_column(self) -> str:
        raise NotImplementedError(
            "x_column not defined for Transitions because it could be 'antecedent' or 'consequent'."
        )

    def _combine_results(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ) -> D:
        """Aggregate results for each group, typically by summing up and normalizing the values. By default,
        the groups correspond to those that had been applied to the analyzed resource. If no Groupers had been
        applied, the entire dataset is treated as a single group.
        """
        group_cols = self._resolve_group_cols_arg(group_cols)

        if self.is_combination:
            # this has been combined before, check if the grouping is the same or a subset of the current grouping
            available_columns = set(self.df.columns) | set(self.df.index.names)
            if group_cols == self.default_groupby:
                return self.df
            elif not set(group_cols).issubset(available_columns):
                raise ValueError(
                    f"Cannot group the results that are already combined by {group_cols}. "
                    f"Available columns are {available_columns}"
                )
        df = self.df[
            [self.dimension_column]
        ]  # gets rid of existing proportion columns, we will get new ones

        groupby = group_cols + self.feature_columns
        groups_to_treat = groupby[:-1]  # normalize by and sort by antecedent groups
        combined_result = df.groupby(groupby).sum()
        normalize_by = combined_result.groupby(groups_to_treat).sum()
        combined_result = self._add_proportion_columns(combined_result, normalize_by)
        return self._sort_combined_result(combined_result, group_cols, sort_order)

    def _compute_entropy(
        self, combined_result: D, group_cols: List[str], weighted: bool = False
    ) -> S:
        antecedent_col, _ = self.feature_columns
        super_method = partial(super()._compute_entropy, group_cols=antecedent_col)
        if not group_cols:
            return super_method(combined_result, weighted=weighted)
        return combined_result.groupby(group_cols).apply(
            super_method, weighted=weighted
        )

    @overload
    def compute_information_gain(
        self, group_cols: Optional[Literal[False]], reverse: bool
    ) -> float:
        ...

    @overload
    def compute_information_gain(
        self, group_cols: UnitOfAnalysis | str | Iterable[str], reverse: bool
    ) -> S:
        ...

    def compute_information_gain(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        reverse: bool = False,
    ) -> S | float:
        """Computes the gain in information about  (reduction in entropy of) the consequent from knowing the antecedent.
        This can be interpreted as measure of how much we know on average about the consequent given an antecedent.

        It is typically explained as the difference between the entropy of the consequents' frequency distribution
        and the weighted frequency-weighted sum of entropies of each antecedent's consequent distribution (which
        is considered as a 'split' in the context of decision trees).


        Args:
            group_cols: Defines the groups for which to compute the information gain.
            reverse: Reverse the argument: How much more do we know about the antecedent when we know the consequent?

        Returns:
            If group_cols is None or empty or resolves to empty (the default when no groupers have been applied),
            the resulting value is a float expressing the difference in entropy. Otherwise, when a grouping is
            performed, the result is a Series of floats.
        """
        group_cols = self._resolve_group_cols_arg(group_cols)
        combined_result = self._combine_results(group_cols=group_cols)
        weighted_entropies = self.compute_entropy(group_cols=group_cols, weighted=True)
        if reverse:
            consequent, antecedent = self.feature_columns
        else:
            antecedent, consequent = self.feature_columns

        def make_original_entropy(df):
            return compute_entropy_of_occurrences(df.groupby(consequent)["count"].sum())

        if group_cols:
            # result will be Series
            original_entropies = combined_result.groupby(group_cols).apply(
                make_original_entropy
            )
            conditioned_entropies = weighted_entropies.groupby(group_cols).sum()
            self.logger.debug(
                f"H({consequent})={original_entropies}\n"
                f"H({consequent}|{antecedent})={conditioned_entropies}"
            )
            return original_entropies - conditioned_entropies

        # result will be float
        original_entropy = make_original_entropy(combined_result)
        conditioned_entropy = weighted_entropies.sum()
        self.logger.debug(
            f"H({consequent})={original_entropy}\n"
            f"H({consequent}|{antecedent})={conditioned_entropy}"
        )
        return original_entropy - conditioned_entropy

    def get_grouping_levels(
        self, smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE
    ) -> List[str]:
        """Returns the levels of the grouping index, i.e., all levels until and including 'piece' or 'slice'."""
        smallest_unit = UnitOfAnalysis(smallest_unit)
        if smallest_unit == UnitOfAnalysis.SLICE:
            return self.get_level_names()[:-2]
        if smallest_unit == UnitOfAnalysis.PIECE:
            return self.get_piece_index(max_levels=0).names
        if smallest_unit == UnitOfAnalysis.GROUP:
            return self.default_groupby

    def make_heatmap(
        self,
        df: Optional[D] = None,
        max_x: Optional[int] = None,
        max_y: Optional[int] = None,
        x_title: Optional[str] = "consequent",
        y_title: Optional[str] = "antecedent",
        facet_row: Optional[str] = None,
        facet_col: Optional[str] = None,
        column_colorscales: Optional[List[str] | Dict[str, str]] = None,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ):
        if df is None:
            df = self.df
        if labels is not None:
            raise NotImplementedError(
                "Changing labels not implemented for heatmaps. You can use x_title and y_title or pass a dict with a "
                "'hovertemplate' to traces_settings, or a dict with 'title_text' to x_axis or y_axis."
            )
        if hover_data is not None:
            raise NotImplementedError(
                "Including more hover_data not implemented for heatmaps."
            )

        return make_heatmaps_from_transitions(
            df,
            max_x=max_x,
            max_y=max_y,
            x_title=x_title,
            y_title=y_title,
            facet_row=facet_row,
            facet_col=facet_col,
            column_colorscales=column_colorscales,
            title=title,
            # labels=labels,
            # hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
        )

    def plot(
        self,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        df = self._combine_results(group_cols=None)
        return self.make_heatmap(
            df=df,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def plot_grouped(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
        group_modes: Optional[GroupMode | Iterable[GroupMode]] = None,
        title: Optional[str] = None,
        labels: Optional[dict] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        layout: Optional[dict] = None,
        font_size: Optional[int] = None,
        x_axis: Optional[dict] = None,
        y_axis: Optional[dict] = None,
        color_axis: Optional[dict] = None,
        traces_settings: Optional[dict] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        group_cols = self._resolve_group_cols_arg(group_cols)

        facet_row, facet_col, column_colorscales = None, None, None
        if not group_cols:
            pass
        elif len(group_cols) == 1:
            facet_row = group_cols[0]
        elif len(group_cols) == 2:
            if group_cols[0] == "mode":
                facet_col, facet_row = group_cols
            else:
                facet_row, facet_col = group_cols
            if facet_col == "mode" and column_colorscales is None:
                column_colorscales = dict(major="Blues", minor="Reds")
        else:
            raise NotImplementedError(
                f"Cannot show heatmaps for more than two groupings: {group_cols!r}"
            )
        return self.make_heatmap(
            facet_row=facet_row,
            facet_col=facet_col,
            column_colorscales=column_colorscales,
            title=title,
            labels=labels,
            hover_data=hover_data,
            height=height,
            width=width,
            layout=layout,
            font_size=font_size,
            x_axis=x_axis,
            y_axis=y_axis,
            color_axis=color_axis,
            traces_settings=traces_settings,
            output=output,
            **kwargs,
        )

    def _sort_combined_result(
        self,
        combined_result: D,
        group_cols: List[str],
        sort_order: Optional[SortOrder] = SortOrder.DESCENDING,
    ):
        if sort_order is None or sort_order == SortOrder.NONE:
            return combined_result

        antecedent, consequent = self.feature_columns
        group_cols = self._resolve_group_cols_arg(group_cols)
        ascending = sort_order == SortOrder.ASCENDING

        def sort_transitions(df):
            gpb = df.groupby(antecedent)
            # order antecedents by overall occurrence
            antecedent_order = (
                gpb[self.dimension_column].sum().sort_values(ascending=ascending).index
            )
            # then, order each antecedent group by occurrence of consequents
            sorted_groups = [
                gpb.get_group(antecedent_group).sort_values(
                    self.dimension_column,
                    ascending=ascending,
                )
                for antecedent_group in antecedent_order
            ]
            return pd.concat(sorted_groups, names=[antecedent])

        if group_cols:
            gpb = combined_result.groupby(group_cols, group_keys=False)
            return gpb.apply(sort_transitions)
        return sort_transitions(combined_result)


def prepare_transitions(
    df: D, max_x: Optional[int] = None, max_y: Optional[int] = None
) -> Tuple[D, D, D]:
    """Turns transitions that come in long format into wide format (transition matrix), optionally subselecting
    the first n columns (max_x) and rows (max_y). Transitions are expected to be sorted, have the consequents (the new
    columns) in the last (right-most) index level, and come with the columns "count", "proportion" and "proportion_%".
    """
    make_subset = (max_x is not None) or (max_y is not None)
    x_slice = slice(None) if max_x is None else slice(None, max_x)
    y_slice = slice(None) if max_y is None else slice(None, max_y)
    counts = df["count"].unstack(sort=False)
    proportions = df["proportion"].unstack(sort=False)
    proportions_str = df["proportion_%"].unstack(sort=False)
    if make_subset:
        counts = counts.iloc[y_slice, x_slice]
        proportions = proportions.iloc[y_slice, x_slice]
        proportions_str = proportions_str.iloc[y_slice, x_slice]
    return proportions, counts, proportions_str


def make_heatmaps_from_transitions(
    transitions_df: D,
    max_x: Optional[int] = None,
    max_y: Optional[int] = None,
    x_title: Optional[str] = "consequent",
    y_title: Optional[str] = "antecedent",
    facet_col: Optional[str] = None,
    facet_row: Optional[str] = None,
    column_colorscales: Optional[List[str] | Dict[str, str]] = None,
    title: Optional[str] = None,
    # labels: Optional[dict] = None,
    # hover_data: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layout: Optional[dict] = None,
    font_size: Optional[int] = None,
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
    output: Optional[str] = None,
) -> go.Figure:
    groupby = []
    make_facet_rows = facet_row is not None
    make_facet_cols = facet_col is not None
    if make_facet_rows:
        groupby.append(facet_row)
    if make_facet_cols:
        groupby.append(facet_col)
    figure_layout = dict()
    if title:
        figure_layout["title_text"] = title
    if height:
        figure_layout["height"] = height
    if width:
        figure_layout["width"] = width
    xaxis_settings = dict(scaleanchor="y", constrain="domain")
    yaxis_settings = dict(scaleanchor="x", constrain="domain", autorange="reversed")
    if x_axis:
        xaxis_settings.update(x_axis)
    if y_axis:
        yaxis_settings.update(y_axis)

    hovertemplate = (
        f"{y_title}: <b>%{{y}}</b><br>"
        f"{x_title}: <b>%{{x}}</b><br>"
        f"proportion: <b>%{{text}}</b><br>"
        f"count: <b>%{{customdata}}</b><br>"
    )
    texttemplate = "%{text}"
    traces_update = dict(hovertemplate=hovertemplate, texttemplate=texttemplate)
    if traces_settings:
        traces_update.update(traces_settings)

    if not groupby:
        # no subplots needed, return single heatmap
        proportions, counts, proportions_str = prepare_transitions(
            transitions_df, max_x=max_x, max_y=max_y
        )
        fig = go.Figure(
            data=make_heatmap(
                proportions, customdata=counts, text=proportions_str, name="Transition"
            )
        )
        update_figure_layout(
            fig=fig,
            layout=layout,
            font_size=font_size,
            x_axis=xaxis_settings,
            y_axis=yaxis_settings,
            color_axis=color_axis,
            traces_settings=traces_update,
        )
        if output:
            write_image(fig=fig, filename=output, width=width, height=height)
        return fig

    # prepare subplots according to groupby
    facet_row_names, facet_col_names = [], []
    group2row_col = {}
    group2data, group2customdata, group2text = {}, {}, {}

    def _update_facet_names(row_name=None, col_name=None) -> Tuple[int, int]:
        if row_name is not None:
            if row_name in facet_row_names:
                row = facet_row_names.index(row_name) + 1
            else:
                facet_row_names.append(row_name)
                row = len(facet_row_names)
        else:
            row = 1
        if col_name is not None:
            if col_name in facet_col_names:
                col = facet_col_names.index(col_name) + 1
            else:
                facet_col_names.append(col_name)
                col = len(facet_col_names)
        else:
            col = 1
        return row, col

    def update_facet_names(group):
        if make_facet_rows and make_facet_cols:
            row_name, col_name = group
            row, col = _update_facet_names(row_name, col_name)
        elif make_facet_rows:
            row, col = _update_facet_names(row_name=group)
        elif make_facet_cols:
            row, col = _update_facet_names(col_name=group)
        else:
            raise RuntimeError("Shouldn't have reached here.")
        group2row_col[group] = row, col

    # prepare the transition data
    for group, group_df in transitions_df.groupby(groupby, group_keys=False):
        if not isinstance(group, str):
            if isinstance(group, tuple):
                group = ", ".join(str(g) for g in group)
            else:
                group = str(group)
        proportions, counts, proportions_str = prepare_transitions(
            group_df, max_x=max_x, max_y=max_y
        )
        group2data[group] = proportions
        group2customdata[group] = counts
        group2text[group] = proportions_str
        update_facet_names(group)

    # prepare the colorscales
    colorscale_list = []
    if column_colorscales is not None:
        if isinstance(column_colorscales, list):
            assert len(column_colorscales) >= len(facet_col_names), (
                f"length of column_colorscales ({len(column_colorscales)}) needs to be at least the number of columns "
                f"({len(facet_row_names)})."
            )
            colorscale_list = column_colorscales
        elif isinstance(column_colorscales, dict):
            if make_facet_cols:
                for col_name in facet_col_names:
                    if col_name not in column_colorscales:
                        print(f"No colorscale defined for group {col_name}.")
                        colorscale_list.append(None)
                    else:
                        colorscale_list.append(column_colorscales[col_name])
            else:
                print("facet_colorscales has no effect if facet_col is False")
        else:
            raise TypeError(
                f"Expected list or dict for column_colorscales, got {type(column_colorscales)}"
            )

    # make subplots figure
    n_rows = max(1, len(facet_row_names))
    n_cols = max(1, len(facet_col_names))
    row_titles = facet_row_names if make_facet_rows else None
    col_titles = facet_col_names if make_facet_cols else None
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        row_titles=row_titles,
        column_titles=col_titles,
        x_title=x_title,
        y_title=y_title,
    )

    # populate figure with heatmaps
    for group, proportions in group2data.items():
        row, col = group2row_col[group]
        if colorscale_list:
            colorscale = colorscale_list[col - 1]
        else:
            colorscale = "Blues"
        heatmap = make_heatmap(
            proportions=proportions,
            customdata=group2customdata[group],
            text=group2text[group],
            colorscale=colorscale,
            name=group,
        )
        fig.add_trace(heatmap, row, col)

    # layout and return
    update_figure_layout(
        fig=fig,
        layout=layout,
        font_size=font_size,
        x_axis=xaxis_settings,
        y_axis=yaxis_settings,
        color_axis=color_axis,
        traces_settings=traces_update,
    )
    if output:
        write_image(fig=fig, filename=output, width=width, height=height)
    return fig
