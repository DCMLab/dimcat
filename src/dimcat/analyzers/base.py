"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import logging
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Collection,
    Iterable,
    List,
    MutableMapping,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import marshmallow as mm
import plotly.express as px
from dimcat.base import DimcatConfig, DimcatObject, PipelineStep, get_class
from dimcat.dataset import Dataset
from dimcat.resources.base import DimcatResource, SomeSeries
from dimcat.resources.features import Feature
from typing_extensions import Self

logger = logging.getLogger(__name__)


R = TypeVar("R")


class AnalyzerName(str, Enum):
    """Identifies the available analyzers."""

    Analyzer = "Analyzer"
    Counter = "Counter"

    def get_class(self) -> Type[Feature]:
        return get_class(self.name)

    @classmethod
    def _missing_(cls, value) -> Self:
        value_lower = value.lower()
        lc_values = {member.value.lower(): member for member in cls}
        if value_lower in lc_values:
            return lc_values[value_lower]
        for lc_value, member in lc_values.items():
            if lc_value.startswith(value_lower):
                return member
        raise ValueError(f"ValueError: {value!r} is not a valid AnalyzerName.")

    def __eq__(self, other) -> bool:
        if self.value == other:
            return True
        if isinstance(other, str):
            return other.lower() == self.value.lower()
        return False

    def __hash__(self):
        return hash(self.value)


class ResultName(str, Enum):
    """Identifies the available analyzers."""

    Result = "Result"


class Result(DimcatResource):
    _enum_type = ResultName

    def plot(self):
        df = self.df.reset_index()
        return px.bar(
            df,
            y=df.columns[-1],
            hover_data=["corpus", "fname"],
            labels=dict(index="piece"),
        )


class DispatchStrategy(str, Enum):
    GROUPBY_APPLY = "GROUPBY_APPLY"
    ITER_STACK = "ITER_STACK"


class UnitOfAnalysis(str, Enum):
    SLICE = "SLICE"
    PIECE = "PIECE"
    GROUP = "GROUP"


class Orientation(str, Enum):
    WIDE = "WIDE"
    LONG = "LONG"


class Analyzer(PipelineStep):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed.
    The base class performs no analysis, instantiating it serves mere testing purpose.
    """

    _enum_type: ClassVar[Type[Enum]] = AnalyzerName
    _result_type: ResultName = ResultName.Result

    assert_all: ClassVar[Tuple[str]] = tuple()
    """Each of these :obj:`PipelineSteps <.PipelineStep>` needs to be matched by at least one PipelineStep previously
     applied to the :obj:`.Dataset`, otherwise :meth:`process_data` raises a ValueError."""

    # assert_previous_step: ClassVar[Tuple[str]] = tuple()
    # """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    # :obj:`_Dataset` does not match any of these types."""

    excluded_steps: ClassVar[Tuple[str]] = tuple()
    """:meth:`process_data` raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    :obj:`.Dataset` matches one of these types."""

    class Schema(PipelineStep.Schema):
        features = mm.fields.List(
            mm.fields.Nested(DimcatConfig.Schema),
            required=True,
            validate=mm.validate.Length(min=1),
        )
        strategy = mm.fields.Enum(DispatchStrategy)
        smallest_unit = mm.fields.Enum(UnitOfAnalysis)
        orientation = mm.fields.Enum(Orientation)
        fill_na: mm.fields.Raw(allow_none=True)

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
        features: DimcatConfig | Iterable[DimcatConfig],
        strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY,
        smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE,
        orientation: Orientation = Orientation.WIDE,
        fill_na: Any = None,
    ):
        self._features: List[DimcatConfig] = []
        self.features = features
        self._strategy: DispatchStrategy = None
        self.strategy = strategy
        self._smallest_unit: UnitOfAnalysis = None
        self.smallest_unit = smallest_unit
        self._orientation: Orientation = None
        self.orientation = orientation
        self.fill_na: Any = fill_na

    @property
    def features(self) -> List[DimcatConfig]:
        return self._features

    @features.setter
    def features(self, features):
        if isinstance(features, (MutableMapping, Feature)):
            features = [features]
        configs, not_configs = [], []
        for config in features:
            if isinstance(config, DimcatConfig):
                configs.append(config)
            elif isinstance(config, Feature):
                cfg = config.to_config()
                configs.append(cfg)
            elif isinstance(config, MutableMapping):
                cfg = DimcatConfig(config)
                configs.append(cfg)
            else:
                not_configs.append(config)
        if len(not_configs) > 0:
            logger.warning(f"Not a configuration of a Feature: {not_configs}")
        if len(configs) == 0:
            raise ValueError(
                f"Did not receive any DimcatConfig, not setting {self.name}.features."
            )
        self._features = configs

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

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: Orientation):
        if not isinstance(orientation, Orientation):
            orientation = Orientation(orientation)
        self._orientation = orientation

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

    def groupby_apply(self, feature: Feature, groupby: SomeSeries = None, **kwargs):
        """Static method that performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            return feature.groupby(level=[0, 1]).apply(self.compute, **self.to_dict())
        return feature.groupby(groupby).apply(self.compute, **self.to_dict())

    def dispatch(self, dataset: Dataset) -> Result:
        """The logic how and to what the compute method is applied, based on the config and the Dataset."""
        if self.strategy == DispatchStrategy.ITER_STACK:  # more cases to follow
            raise NotImplementedError()
        if self.strategy == DispatchStrategy.GROUPBY_APPLY:
            stacked_feature = self.pre_process(dataset.load_feature(self.features[0]))
            results = self.groupby_apply(stacked_feature)
            return Result(df=results)
        raise ValueError(f"Unknown dispatch strategy '{self.strategy!r}'")

    @classmethod
    def _check_asserted_pipeline_steps(cls, dataset: Dataset):
        """Returns None if the check passes.

        Raises:
            ValueError: If one of the asserted PipelineSteps has not previously been applied to the Dataset.
        """
        if len(cls.assert_all) == 0:
            return True
        assert_steps = typestrings2types(cls.assert_all)
        missing = []
        for step in assert_steps:
            if not any(
                isinstance(previous_step, step)
                for previous_step in dataset.pipeline_steps
            ):
                missing.append(step)
        if len(missing) > 0:
            missing_names = ", ".join(m.__name__ for m in missing)
            raise ValueError(
                f"Applying a {cls.name} requires previous application of: {missing_names}."
            )

    @classmethod
    def _check_excluded_pipeline_steps(cls, dataset: Dataset):
        """Returns None if the check passes.

        Raises:
            ValueError: If any of the PipelineSteps applied to the Dataset matches one of the ones excluded.
        """
        if len(cls.excluded_steps) == 0:
            return
        excluded_steps = typestrings2types(cls.excluded_steps)
        excluded = []
        for step in excluded_steps:
            if any(
                isinstance(previous_step, step)
                for previous_step in dataset.pipeline_steps
            ):
                excluded.append(step)
        if len(excluded) > 0:
            excluded_names = ", ".join(e.__name__ for e in excluded)
            raise ValueError(f"{cls.name} cannot be applied after {excluded_names}.")

    def process(self, dataset: Dataset) -> Dataset:
        """Returns an :obj:`AnalyzedData` copy of the Dataset with the added analysis result."""
        self._check_asserted_pipeline_steps(dataset)
        self._check_excluded_pipeline_steps(dataset)
        new_dataset = Dataset(dataset)
        stacked_result = self.dispatch(dataset)
        stacked_result = self.post_process(stacked_result)
        new_dataset.result = stacked_result
        return new_dataset

    def pre_process(self, feature: Feature) -> Feature:
        """Whatever needs to be done before analyzing the feature, e.g. transforming it based on
        the config. The method needs to work both on a Feature and a StackedFeature.
        """
        return feature

    def post_process(self, result):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        return result


def typestrings2types(
    typestrings: Union[Union[str, Enum], Collection[Union[str, Enum]]]
) -> Tuple[type]:
    """Turns one or several names of classes into a tuple of references to these classes."""
    if isinstance(typestrings, (str, Enum)):
        typestrings = [typestrings]
    result = [typestring2type(typestring) for typestring in typestrings]
    return tuple(result)


def typestring2type(typestring: Union[str, Enum]) -> type:
    if isinstance(typestring, Enum):
        typestring = typestring.value
    if typestring in DimcatObject._registry:
        return DimcatObject._registry[typestring]
    raise KeyError(
        f"Typestring '{typestring}' does not correspond to a known subclass of DimcatObject:\n"
        f"{DimcatObject._registry}"
    )
