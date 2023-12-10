"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, ClassVar, Iterable, Optional, Type, TypeVar

import marshmallow as mm
from dimcat.base import FriendlyEnumField, ObjectEnum
from dimcat.data.datasets.processed import AnalyzedDataset
from dimcat.data.resources import Feature
from dimcat.data.resources.base import DR, Rs, SomeSeries
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs, UnitOfAnalysis
from dimcat.data.resources.results import Result
from dimcat.steps.base import FeatureProcessingStep

logger = logging.getLogger(__name__)


R = TypeVar("R")


class AnalyzerName(ObjectEnum):
    """Identifies the available analyzers."""

    Analyzer = "Analyzer"
    BigramAnalyzer = "BigramAnalyzer"
    Counter = "Counter"
    PitchClassVectors = "PitchClassVectors"
    Proportions = "Proportions"


class DispatchStrategy(str, Enum):
    GROUPBY_APPLY = "GROUPBY_APPLY"
    ITER_STACK = "ITER_STACK"


class Analyzer(FeatureProcessingStep):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed.
    The base class performs no analysis, instantiating it serves mere testing purpose.
    """

    _default_dimension_column: ClassVar[Optional[str]] = None
    """Name of a column, contained in the Results produced by this analyzer, containing some dimension,
    e.g. one to be interpreted as quantity (durations, counts, etc.) or as color."""
    _enum_type: ClassVar[Type[Enum]] = AnalyzerName
    _new_dataset_type = AnalyzedDataset
    _new_resource_type = Result
    _output_package_name = "results"
    _applicable_to_empty_datasets = False
    _requires_at_least_one_feature = True

    # assert_all: ClassVar[Tuple[str]] = tuple()
    # """Each of these :obj:`PipelineSteps <.PipelineStep>` needs to be matched by at least one PipelineStep previously
    #  applied to the :obj:`.Dataset`, otherwise :meth:`process_data` raises a ValueError."""
    #
    # # assert_previous_step: ClassVar[Tuple[str]] = tuple()
    # # """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    # # :obj:`_Dataset` does not match any of these types."""
    #
    # excluded_steps: ClassVar[Tuple[str]] = tuple()
    # """:meth:`process_data` raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    # :obj:`.Dataset` matches one of these types."""

    @staticmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    def compute(feature: Feature, **kwargs) -> Any:
        """Static method that performs the actual computation on a single unit of analysis (slice, piece, or group).
        The result of analyzing a resource should be tantamount to a concatenation of the results of applying
        self.compute() to each contained unit, turned into a Feature object in its own right.
        In practice, the analyzers .groupby_apply() method re-implements the same computation and performs it on the
        entire DataFrame at once using .groupby(). In other words, it would be redundant to turn each group into a
        Feature first. self.compute(), however, cannot take a DataFrame as input because it is a static method that
        needs to rely on the Feature object to know which column(s) to process.
        """
        return feature

    # @classmethod
    # def _check_asserted_pipeline_steps(cls, dataset: Dataset):
    #     """Returns None if the check passes.
    #
    #     Raises:
    #         ValueError: If one of the asserted PipelineSteps has not previously been applied to the Dataset.
    #     """
    #     if len(cls.assert_all) == 0:
    #         return True
    #     assert_steps = typestrings2types(cls.assert_all)
    #     missing = []
    #     for step in assert_steps:
    #         if not any(
    #             isinstance(previous_step, step)
    #             for previous_step in dataset.pipeline_steps
    #         ):
    #             missing.append(step)
    #     if len(missing) > 0:
    #         missing_names = ", ".join(m.__name__ for m in missing)
    #         raise ValueError(
    #             f"Applying a {cls.name} requires previous application of: {missing_names}."
    #         )
    #
    # @classmethod
    # def _check_excluded_pipeline_steps(cls, dataset: Dataset):
    #     """Returns None if the check passes.
    #
    #     Raises:
    #         ValueError: If any of the PipelineSteps applied to the Dataset matches one of the ones excluded.
    #     """
    #     if len(cls.excluded_steps) == 0:
    #         return
    #     excluded_steps = typestrings2types(cls.excluded_steps)
    #     excluded = []
    #     for step in excluded_steps:
    #         if any(
    #             isinstance(previous_step, step)
    #             for previous_step in dataset.pipeline_steps
    #         ):
    #             excluded.append(step)
    #     if len(excluded) > 0:
    #         excluded_names = ", ".join(e.__name__ for e in excluded)
    #         raise ValueError(f"{cls.name} cannot be applied after {excluded_names}.")

    class Schema(FeatureProcessingStep.Schema):
        strategy = FriendlyEnumField(DispatchStrategy, metadata=dict(expose=False))
        smallest_unit = FriendlyEnumField(
            UnitOfAnalysis,
            load_default=UnitOfAnalysis.SLICE,
            metadata=dict(expose=False),
        )  # not to be exposed until Slicers become available in the UI
        dimension_column = mm.fields.Str(allow_none=True, metadata=dict(expose=False))

        @mm.pre_load()
        def features_as_list(self, obj, **kwargs):
            """Ensure that features is a list."""
            features = self.get_attribute(obj, "features", None)
            if features is not None and not isinstance(features, list):
                try:
                    obj.features = [obj.features]
                except AttributeError:
                    obj["features"] = [obj["features"]]
            return obj

    def __init__(
        self,
        features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        dimension_column: str = None,
    ):
        super().__init__(features=features)
        self._strategy: DispatchStrategy = None
        self.strategy = strategy
        self._smallest_unit: UnitOfAnalysis = None
        self.smallest_unit = smallest_unit
        self._dimension_column = None
        self.dimension_column = dimension_column

    @property
    def dimension_column(self) -> Optional[str]:
        """Name of a column, contained in the Results produced by this analyzer, containing some dimension,
        e.g. one to be interpreted as quantity (durations, counts, etc.) or as color."""
        return self._dimension_column

    @dimension_column.setter
    def dimension_column(self, dimension_column: Optional[str]):
        if dimension_column is None:
            self._dimension_column = self._default_dimension_column
            return
        if not isinstance(dimension_column, str):
            raise TypeError(
                f"dimension_column must be a string, not {type(dimension_column)}"
            )
        self._dimension_column = dimension_column

    @property
    def strategy(self) -> DispatchStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DispatchStrategy):
        if not isinstance(strategy, DispatchStrategy):
            strategy = DispatchStrategy(strategy)
        self._strategy = strategy

    @property
    def smallest_unit(self) -> UnitOfAnalysis:
        return self._smallest_unit

    @smallest_unit.setter
    def smallest_unit(self, smallest_unit: UnitOfAnalysis):
        if not isinstance(smallest_unit, UnitOfAnalysis):
            smallest_unit = UnitOfAnalysis(smallest_unit)
        self._smallest_unit = smallest_unit

    def _make_new_resource(self, resource: Feature) -> Rs:
        """Dispatch the passed resource to the appropriate method."""
        if self.strategy == DispatchStrategy.ITER_STACK:  # more cases to follow
            raise NotImplementedError()
        if not self.strategy == DispatchStrategy.GROUPBY_APPLY:
            raise ValueError(f"Unknown dispatch strategy '{self.strategy!r}'")
        result_constructor: Type[Result] = self._get_new_resource_type(resource)
        results = self.groupby_apply(resource)
        result_name = self.resource_name_factory(resource)
        value_column = resource.value_column
        if resource.has_distinct_formatted_column:
            formatted_column = resource.formatted_column
        else:
            formatted_column = None
        result = result_constructor.from_dataframe(
            analyzed_resource=resource,
            value_column=value_column,
            dimension_column=self.dimension_column,
            formatted_column=formatted_column,
            df=results,
            resource_name=result_name,
            default_groupby=resource.default_groupby,
        )
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
        return feature.groupby(groupby).apply(self.compute, **self.to_dict())

    def _pre_process_resource(self, resource: DR) -> DR:
        """Perform some pre-processing on a resource before processing it."""
        resource = super()._pre_process_resource(resource)
        resource.load()
        return resource

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.analyzed"
