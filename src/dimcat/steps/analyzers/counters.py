import logging
from typing import Iterable, Optional

import marshmallow as mm
import pandas as pd
from dimcat.base import FriendlyEnumField
from dimcat.data.resources import Feature
from dimcat.data.resources.base import DR, D, SomeDataframe, SomeSeries
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs, UnitOfAnalysis
from dimcat.data.resources.results import (
    CadenceCounts,
    Counts,
    NgramTable,
    NgramTableFormat,
    PhraseData,
)
from dimcat.steps.analyzers.base import Analyzer, DispatchStrategy

module_logger = logging.getLogger(__name__)


class Counter(Analyzer):
    _default_dimension_column = "count"
    _new_resource_type = Counts

    @staticmethod
    def compute(feature: Feature, **kwargs) -> D:
        count_columns = [feature.value_column]
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in count_columns
        ):
            count_columns.append(feature.formatted_column)
        result = feature.df.value_counts(subset=count_columns)
        result = result.to_frame(Counter._default_dimension_column)
        return result

    class Schema(Analyzer.Schema):
        dimension_column = mm.fields.Str(
            load_default="count", allow_none=True, metadata=dict(expose=False)
        )

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_grouping_levels(self.smallest_unit)
            self.logger.debug(
                f"Using the {feature.resource_name}'s grouping levels {groupby!r}"
            )
        if not groupby:
            return self.compute(feature, **kwargs)
        count_columns = [feature.value_column]
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in groupby
        ):
            count_columns.append(feature.formatted_column)
        result = feature.groupby(groupby).value_counts(subset=count_columns)
        result = result.to_frame(self.dimension_column)

        return result

    def resource_name_factory(self, resource: DR) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.counted"


class CadenceCounter(Counter):
    _new_resource_type = CadenceCounts


class NgramAnalyzer(Analyzer):
    _allowed_features = (Feature, PhraseData)
    _new_resource_type = NgramTable

    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        return len(feature.index)

    class Schema(Analyzer.Schema):
        n = mm.fields.Integer(
            load_default=2,
            validate=mm.validate.Range(min=2),
            metadata=dict(
                expose=True,
                description="The n in n-grams, i.e., how many consecutive elements are grouped in one entity.",
            ),
        )
        format = FriendlyEnumField(
            NgramTableFormat,
            load_default=NgramTableFormat.CONVENIENCE,
            metadata=dict(expose=False),
        )

    def __init__(
        self,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        n: int = 2,
        format: NgramTableFormat = NgramTableFormat.CONVENIENCE,
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        dimension_column: str = None,
    ):
        """

        Args:
            features:
                The Feature objects you want this Analyzer to process. If not specified, it will try to process all
                features present in a given Dataset's Outputs catalog.
            n:
                Specify the n-1 number of subsequent elements you want the table to contain for each element.
                n = 2 (the default) corresponds to bigrams, 3 to trigrams, etc.
            format:
                Controls the amount of columns you want to include for the n-1 consequents from the bare minimum
                (FEATURES) to the full duplication including context columns (FULL).
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
        self._n = None
        self.n = n
        self._format = None
        self.format = format

    @property
    def format(self) -> NgramTableFormat:
        return self._format

    @format.setter
    def format(self, format: NgramTableFormat):
        self._format = NgramTableFormat(format)

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, n: int):
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, not {type(n)}")
        if n < 2:
            raise ValueError(f"n must be at least 2, not {n}")
        self._n = n

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_grouping_levels(UnitOfAnalysis.SLICE)
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        include_context_columns = self.format in (
            NgramTableFormat.FEATURES_CONTEXT,
            NgramTableFormat.CONVENIENCE_CONTEXT,
            NgramTableFormat.AUXILIARY_CONTEXT,
            NgramTableFormat.FULL,
        )
        include_auxiliary_columns = self.format in (
            NgramTableFormat.AUXILIARY,
            NgramTableFormat.AUXILIARY_CONTEXT,
            NgramTableFormat.FULL_WITHOUT_CONTEXT,
            NgramTableFormat.FULL,
        )
        include_convenience_columns = self.format in (
            NgramTableFormat.CONVENIENCE,
            NgramTableFormat.CONVENIENCE_CONTEXT,
            NgramTableFormat.FULL_WITHOUT_CONTEXT,
            NgramTableFormat.FULL,
        )
        if hasattr(feature, "get_available_column_names"):
            columns_to_shift = feature.get_available_column_names(
                context_columns=include_context_columns,
                auxiliary_columns=include_auxiliary_columns,
                convenience_columns=include_convenience_columns,
                feature_columns=True,
            )
            df_to_shift = feature.df[columns_to_shift]
        else:
            df_to_shift = feature.df
        concatenate_this = {"a": feature.df}
        for i in range(1, self.n):
            key = chr(ord("a") + i)
            df_to_shift = df_to_shift.groupby(groupby, group_keys=False).apply(
                lambda df: df.shift(-1)
            )
            concatenate_this[key] = df_to_shift
        return pd.concat(concatenate_this, axis=1)

    def _post_process_result(
        self,
        result: NgramTable,
        original_resource: Feature,
    ) -> NgramTable:
        if hasattr(original_resource, "get_available_column_names"):
            result._auxiliary_columns = original_resource.get_available_column_names(
                context_columns=True
            )
            result._convenience_columns = original_resource.get_available_column_names(
                auxiliary_columns=True, convenience_columns=True
            )
        return result

    def resource_name_factory(self, resource: DR) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.ngram_table"


class BigramAnalyzer(NgramAnalyzer):
    def resource_name_factory(self, resource: DR) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.bigram_table"

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, n: int):
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, not {type(n)}")
        if n < 2:
            raise ValueError(f"n must be at least 2, not {n}")
        if n > 2:
            self.logger.debug(f"BigramAnalyzer with n=={n}? You do you.")
        self._n = n
