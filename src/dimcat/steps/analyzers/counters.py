import logging
from typing import Any, Iterable, Optional

import marshmallow as mm
import pandas as pd
from dimcat.base import FriendlyEnum
from dimcat.data.resources import Feature
from dimcat.data.resources.base import D, SomeDataframe, SomeSeries
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs, UnitOfAnalysis
from dimcat.data.resources.results import Counts, NgramTable
from dimcat.steps.analyzers.base import Analyzer, DispatchStrategy

logger = logging.getLogger(__name__)


class Counter(Analyzer):
    _dimension_column_name = "count"
    _new_resource_type = Counts

    @staticmethod
    def compute(feature: Feature, **kwargs) -> D:
        groupby = [feature.value_column]
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in groupby
        ):
            groupby.append(feature.formatted_column)
        result = feature.groupby(groupby)[Counter._dimension_column_name].value_counts(
            dropna=False
        )
        result = result.to_frame(Counter._dimension_column_name)
        return result

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_grouping_levels(self.smallest_unit)
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        groupby.append(feature.value_column)
        if (
            feature.formatted_column is not None
            and feature.formatted_column not in groupby
        ):
            groupby.append(feature.formatted_column)
        result = feature.groupby(groupby).size()
        result = result.to_frame(self._dimension_column_name)
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.counted"


class NgramTableFormat(FriendlyEnum):
    """The format of the ngram table determining how many columns are copied for each of the n-1 shifts.
    The original columns are always copied.
    This setting my have a significant effect on the performance when creating the NgramTable.
    """

    FEATURES = "FEATURES"
    FEATURES_CONTEXT = "FEATURES_CONTEXT"
    CONVENIENCE = "CONVENIENCE"
    CONVENIENCE_CONTEXT = "CONVENIENCE_CONTEXT"
    AUXILIARY = "AUXILIARY"
    AUXILIARY_CONTEXT = "AUXILIARY_CONTEXT"
    FULL_WITHOUT_CONTEXT = "FULL_WITHOUT_CONTEXT"
    FULL = "FULL"


class NgramAnalyzer(Analyzer):
    _new_resource_type = NgramTable

    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        return len(feature.index)

    class Schema(Analyzer.Schema):
        n = mm.fields.Integer(load_default=2)
        format = mm.fields.Enum(
            NgramTableFormat, load_default=NgramTableFormat.CONVENIENCE
        )

    def __init__(
        self,
        n: int = 2,
        format: NgramTableFormat = NgramTableFormat.CONVENIENCE,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        fill_na: Any = None,
    ):
        super().__init__(
            features=features,
            strategy=strategy,
            smallest_unit=smallest_unit,
            fill_na=fill_na,
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
        columns_to_shift = feature.get_available_column_names(
            context_columns=include_context_columns,
            auxiliary_columns=include_auxiliary_columns,
            convenience_columns=include_convenience_columns,
            feature_columns=True,
        )
        df_to_shift = feature.df[columns_to_shift]
        concatenate_this = {"a": feature.df}
        for i in range(1, self.n):
            key = chr(ord("a") + i)
            df_to_shift = df_to_shift.groupby(groupby, group_keys=False).apply(
                lambda df: df.shift(-1)
            )
            concatenate_this[key] = df_to_shift
        return pd.concat(concatenate_this, axis=1)

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.ngram_table"


class BigramAnalyzer(NgramAnalyzer):
    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.bigram_table"
