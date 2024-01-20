from typing import ClassVar, Iterable, List, Optional

import pandas as pd
from dimcat.base import ListOfStringsField
from dimcat.data.resources import Feature
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs, UnitOfAnalysis
from dimcat.data.resources.results import PrevalenceMatrix, Result
from dimcat.steps.analyzers.base import Analyzer, DispatchStrategy
from dimcat.steps.base import D


class PrevalenceAnalyzer(Analyzer):
    """Creates what is the equivalent to NLP's "frequency matrix" except that in the case of music,
    the coefficients are not restricted to represent count frequencies (when created from a
    :class:`~.data.resources.results.Counts` object) but can also represent durations (when created
    from a :class:`~.data.resources.results.Durations` object). When the analyzer is applied to
    a :class:`Feature`, its default analysis will be used.
    """

    _default_dimension_column: ClassVar[Optional[str]] = "duration_qb"
    _new_resource_type = PrevalenceMatrix

    @staticmethod
    def compute(
        resource: D | DimcatResource,
        index: Optional[str | Iterable[str]] = None,
        columns: Optional[str | Iterable[str]] = None,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        dimension_column: Optional[str] = None,
        **kwargs,
    ) -> D:
        """Computes the prevalence matrix from the given resource. This is basically a wrapper
        around :meth:`pandas.DataFrame.pivot_table` with ``aggfunc="sum"``.

        Args:
            resource:
                A dataframe, :class:`Feature` or :class:`Result` which will be pivoted to produce
                a prevalence with ``index`` index level(s) and ``columns`` column level(s),
                summing up the respective values contained in ``dimension_column``.
            index:
                Column(s) and/or index level name(s) that will make up the index values of the
                :class:`~.data.resources.results.PrevalenceMatrix` (akin to a groupby). By default,
                all but the last level will be used.
            columns:
                Column(s) and/or index level name(s) that will make up the column names of the
                :class:`~.data.resources.results.PrevalenceMatrix`. By default, the
                :attr:`~.data.resources.Resource.value_column` will be used.
            smallest_unit:
                The smallest unit to consider for analysis. Relevant only when ``index`` is not
                specified and ``resource`` is a :class:`~.data.resources.DimcatResource`.
            dimension_column:
                Name of the column that represents absolute prevalence values, typically "duration_qb"
                or "count". Required only when ``resource`` is a dataframe.
            **kwargs:

        Returns:
            A pivot table with summed (=absolute) prevalence coefficients. For the analogy with
            NLP's frequency matrix, the ``index`` will correspond to documents and the ``columns``
            to the vocabulary (words/tokens).
        """
        is_dataframe = isinstance(resource, pd.DataFrame)
        assert not (is_dataframe and dimension_column is None), (
            "When passing a dataframe as resource, you need to specify the dimension column containing the "
            "absolute prevalence values."
        )
        if not index:
            if is_dataframe:
                index = resource.index.names[:-1]
            else:
                index = resource.get_grouping_levels(smallest_unit)
        elif isinstance(index, str):
            index = [index]
        else:
            index = list(index)
        if not columns:
            if not is_dataframe:
                columns = [resource.value_column]
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)
        if dimension_column is None:
            dimension_column = getattr(resource, "dimension_column")
        result = resource.pivot_table(
            index=index,
            columns=columns,
            values=dimension_column,
            aggfunc="sum",
            sort=False,
        )
        # sort columns by their overall prevalence
        result = PrevalenceMatrix._sort_combined_result(result, sort_order="descending")
        return result

    class Schema(Analyzer.Schema):
        index = ListOfStringsField(allow_none=True)
        columns = ListOfStringsField(allow_none=True)

    def __init__(
        self,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        columns: Optional[str | Iterable[str]] = None,
        index: Optional[str | Iterable[str]] = None,
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        dimension_column: str = None,
    ):
        """

        Args:
            features:
                The Feature objects you want this Analyzer to process. If not specified, it will try to process all
                features present in a given Dataset's Outputs catalog.
            strategy: Currently, only the default strategy GROUPBY_APPLY is implemented.
            smallest_unit:
                The smallest unit to consider for analysis. Defaults to SLICE, meaning that slice segments are analyzed
                if a slicer has been previously applied, piece units otherwise. The results for larger units can always
                be retrospectively retrieved by using :meth:`Result.combine_results()`, but not the other way around.
                Use this setting to reduce compute time by setting it to PIECE, CORPUS_GROUP, or GROUP where the latter
                uses the default groupby if a grouper has been previously applied, or the entire dataset, otherwise.
            dimension_column:
                Name of the column containing some dimension, e.g. to be interpreted as quantity (durations, counts,
                etc.) or as color.
        """
        super().__init__(
            features=features,
            strategy=strategy,
            smallest_unit=smallest_unit,
            dimension_column=dimension_column,
        )
        self._columns = None
        self._index = None
        if columns:
            self.columns = columns
        if index:
            self.index = index

    @property
    def columns(self) -> List[str]:
        if self._columns is None:
            return []
        return list(self._columns)

    @columns.setter
    def columns(self, value: Optional[str | Iterable[str]]):
        if isinstance(value, str):
            value = [value]
        self._columns = list(value)

    @property
    def index(self) -> List[str]:
        if self._index is None:
            return []
        return list(self._index)

    @index.setter
    def index(self, value: Optional[str | Iterable[str]]):
        if isinstance(value, str):
            value = [value]
        self._index = list(value)

    def groupby_apply(
        self,
        feature: Result | Feature,
        groupby: Optional[str | Iterable[str]] = None,
        **kwargs,
    ) -> D:
        settings = self.to_config()
        if groupby is not None:
            settings["index"] = groupby
        return self.compute(feature, **settings)
