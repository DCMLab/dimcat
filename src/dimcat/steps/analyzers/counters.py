import logging
from typing import Any, Iterable, Optional

import marshmallow as mm
import pandas as pd
from dimcat.base import FriendlyEnum
from dimcat.data.resources.base import SomeDataframe, SomeSeries
from dimcat.data.resources.dc import DimcatResource, UnitOfAnalysis
from dimcat.data.resources.features import Feature, FeatureSpecs
from dimcat.data.resources.results import NgramTable
from dimcat.steps.analyzers.base import Analyzer, DispatchStrategy

logger = logging.getLogger(__name__)


class Counter(Analyzer):
    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        return len(feature.index)

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            groupby = feature.get_grouping_levels(self.smallest_unit)
            self.logger.debug(
                f"Using the {feature.resource_name}'s default groupby {groupby!r}"
            )
        return (
            feature.groupby(groupby).size().to_frame(f"{feature.resource_name} counts")
        )

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_counted"


class NgramTableFormat(FriendlyEnum):
    """The format of the ngram table."""

    FEATURES = "FEATURES"
    FEATURES_CONTEXT = "FEATURES_CONTEXT"
    AUXILIARY = "AUXILIARY"
    AUXILIARY_CONTEXT = "AUXILIARY_CONTEXT"
    FULL = "FULL"


class NgramAnalyzer(Analyzer):
    new_resource_type = NgramTable

    @staticmethod
    def compute(feature: DimcatResource | SomeDataframe, **kwargs) -> int:
        return len(feature.index)

    class Schema(Analyzer.Schema):
        n = mm.fields.Integer(load_default=2)
        format = mm.fields.Enum(
            NgramTableFormat, load_default=NgramTableFormat.AUXILIARY_CONTEXT
        )

    def __init__(
        self,
        n: int = 2,
        format: NgramTableFormat = NgramTableFormat.AUXILIARY_CONTEXT,
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

        if self.format in (
            NgramTableFormat.FEATURES,
            NgramTableFormat.FEATURES_CONTEXT,
        ):
            right_df = feature.df[feature.feature_column_names]
        elif self.format in (
            NgramTableFormat.AUXILIARY,
            NgramTableFormat.AUXILIARY_CONTEXT,
        ):
            right_df = feature.df[
                feature.auxiliary_column_names + feature.feature_column_names
            ]
        elif self.format == NgramTableFormat.FULL:
            right_df = feature.df[feature.get_column_names()]
        if self.format in (
            NgramTableFormat.FEATURES,
            NgramTableFormat.AUXILIARY,
            NgramTableFormat.FULL,
        ):
            left_df = right_df
        else:
            left_df = pd.concat(
                [feature.df[feature.context_column_names], right_df], axis=1
            )
        concatenate_this = {"a": left_df}
        for i in range(1, self.n):
            key = chr(ord("a") + i)
            right_df = right_df.groupby(groupby, group_keys=False).apply(
                lambda df: df.shift(-1)
            )
            concatenate_this[key] = right_df
        return pd.concat(concatenate_this, axis=1)

    def _post_process_result(
        self,
        result: NgramTable,
        original_resource: Feature,
    ) -> DimcatResource:
        """Change the default_groupby value of the returned Feature."""
        result.context_column_names = original_resource.context_column_names
        result.feature_column_names = original_resource.feature_column_names
        result.auxiliary_column_names = original_resource.auxiliary_column_names
        return result

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_ngram_table"


class BigramAnalyzer(NgramAnalyzer):
    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}_bigram_table"
