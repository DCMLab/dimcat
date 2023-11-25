"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, ClassVar, Iterable, Optional, Type, TypeVar

import marshmallow as mm
from dimcat.base import ObjectEnum
from dimcat.data.datasets.processed import AnalyzedDataset
from dimcat.data.resources import Feature
from dimcat.data.resources.base import SomeSeries
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

    _dimension_column_name: ClassVar[Optional[str]] = None
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
        """Static method that performs the actual computation."""
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
        strategy = mm.fields.Enum(DispatchStrategy, metadata={"expose": False})
        smallest_unit = mm.fields.Enum(UnitOfAnalysis, metadata={"expose": False})
        fill_na = mm.fields.Raw(allow_none=True, metadata={"expose": False})

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
        fill_na: Any = None,
    ):
        super().__init__(features=features)
        self._strategy: DispatchStrategy = None
        self.strategy = strategy
        self._smallest_unit: UnitOfAnalysis = None
        self.smallest_unit = smallest_unit
        self.fill_na: Any = fill_na

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

    def _make_new_resource(self, resource: Feature) -> Result:
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
            dimension_column=self._dimension_column_name,
            formatted_column=formatted_column,
            df=results,
            resource_name=result_name,
            default_groupby=resource.get_default_groupby(),
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

    def _pre_process_resource(self, resource: DimcatResource) -> DimcatResource:
        """Perform some pre-processing on a resource before processing it."""
        resource = super()._pre_process_resource(resource)
        resource.load()
        return resource

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Returns a name for the resource based on its name and the name of the pipeline step."""
        return f"{resource.resource_name}.analyzed"


# def typestrings2types(
#     typestrings: Union[Union[str, Enum], Collection[Union[str, Enum]]]
# ) -> Tuple[type]:
#     """Turns one or several names of classes into a tuple of references to these classes."""
#     if isinstance(typestrings, (str, Enum)):
#         typestrings = [typestrings]
#     result = [typestring2type(typestring) for typestring in typestrings]
#     return tuple(result)
#
#
# def typestring2type(typestring: Union[str, Enum]) -> type:
#     if isinstance(typestring, Enum):
#         typestring = typestring.value
#     if typestring in DimcatObject._registry:
#         return DimcatObject._registry[typestring]
#     raise KeyError(
#         f"Typestring '{typestring}' does not correspond to a known subclass of DimcatObject:\n"
#         f"{DimcatObject._registry}"
#     )
