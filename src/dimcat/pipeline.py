from __future__ import annotations

import logging
from typing import ClassVar, Iterable, List, Optional, Tuple, Type, Union, overload

from dimcat.exceptions import (
    EmptyDatasetError,
    EmptyResourceError,
    FeatureUnavailableError,
    NoFeaturesActiveError,
)
from marshmallow import fields, validate

from .base import Data, DimcatConfig, DimcatObject, deserialize_dict
from .dataset.base import Dataset, DimcatPackage
from .resources.base import DimcatResource
from .resources.features import FeatureSpecs, features_argument2config_list

logger = logging.getLogger(__name__)


class PipelineStep(DimcatObject):
    """
    This base class unites all classes able to transform some data in a pre-defined way.

    The initializer will set some parameters of the transformation, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.


    """

    new_dataset_type: Optional[ClassVar[Type[Dataset]]] = None
    """If specified, :meth:`process_dataset` will return Datasets of this type, otherwise same as input type."""
    new_resource_type: Optional[ClassVar[Type[DimcatResource]]] = None
    """If specified, :meth:`process_resource` will return Resources of this type, otherwise same as input type."""
    output_package_name: Optional[str] = None
    """Name of the package in which to store the outputs of this step. If None, the PipeLine step will replace the
    'features' package of the given dataset."""
    applicable_to_empty_datasets: ClassVar[bool] = True
    """If False, :meth:`check_dataset` will raise an EmptyDatasetError if no data has been loaded yet. This makes sense
    for PipelineSteps that are dependent on the data, e.g. because they use :meth:`fit_to_dataset`."""
    requires_at_least_one_feature: ClassVar[bool] = False

    class Schema(DimcatObject.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            allow_none=True,
        )

    def __init__(
        self, features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None, **kwargs
    ):
        self._features: List[DimcatConfig] = []
        self.features = features

    @property
    def features(self) -> List[DimcatConfig]:
        return self._features

    @features.setter
    def features(self, features):
        configs = features_argument2config_list(features)
        if len(self._features) > 0:
            self.logger.info(
                f"Previously selected features {self._features} overwritten by {configs}."
            )
        self._features = configs

    @property
    def is_transformation(self) -> bool:
        """True if this PipelineStep transforms features, replacing the dataset.outputs['features'] package."""
        return (
            self.output_package_name is None or self.output_package_name == "features"
        )

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns:
            True if the passed data is eligible.
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def check_dataset(self, dataset: Dataset) -> None:
        """Check if the dataset is eligible for processing.

        Raises:
            EmptyDatasetError: If the dataset has no features.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset)}")
        if not self.applicable_to_empty_datasets:
            if dataset.n_features_available == 0:
                raise EmptyDatasetError
        required_features = self.get_required_features()
        if self.requires_at_least_one_feature:
            if len(required_features) == 0 and dataset.n_active_features == 0:
                raise NoFeaturesActiveError
        for feature in required_features:
            if not dataset.check_feature_availability(feature):
                raise FeatureUnavailableError

    def check_resource(self, resource: DimcatResource) -> None:
        if not isinstance(resource, DimcatResource):
            raise TypeError(f"Expected DimcatResource, got {type(resource)}")
        if resource.is_empty:
            raise EmptyResourceError
        # ToDo: check if eligible for processing
        return

    def dispatch(self, resource: DimcatResource) -> DimcatResource:
        """Dispatch the passed resource to the appropriate method."""
        resource_constructor = self.get_new_resource_type(resource)
        # This is where the input resource is being processed
        resource_name = self.resource_name_factory(resource)
        return resource_constructor.from_resource(resource, resource_name=resource_name)

    def get_new_resource_type(self, resource):
        if self.new_resource_type is None:
            resource_constructor: Type[DimcatResource] = resource.__class__
        else:
            resource_constructor: Type[DimcatResource] = self.new_resource_type
        return resource_constructor

    def fit_to_dataset(self, dataset: Dataset) -> None:
        """Adjust the PipelineStep to the passed dataset.

        Args:
            dataset: The dataset to adjust to.
        """
        return

    def get_required_features(self) -> List[DimcatConfig]:
        """Return a list of feature names required for this PipelineStep."""
        return self.features

    def _make_new_dataset(self, dataset):
        if self.new_dataset_type is None:
            dataset_constructor: Type[Dataset] = dataset.__class__
        else:
            dataset_constructor: Type[Dataset] = self.new_dataset_type
        new_dataset = dataset_constructor.from_dataset(dataset)
        self.logger.debug(
            f"Created new dataset {new_dataset} of type {dataset_constructor.__name__}."
        )
        return new_dataset

    def _make_new_package(self) -> DimcatPackage:
        if self.output_package_name is None:
            return DimcatPackage(package_name="features")
        return DimcatPackage(package_name=self.output_package_name)

    def pre_process_resource(self, resource: DimcatResource) -> DimcatResource:
        """Perform some pre-processing on a resource before processing it."""
        resource.load()
        return resource

    def post_process_result(self, result: DimcatResource) -> DimcatResource:
        """Perform some post-processing on a resource after processing it."""
        return result

    @overload
    def process(self, data: Data) -> Data:
        ...

    @overload
    def process(self, data: Iterable[Data]) -> List[Data]:
        ...

    def process(self, data: Union[Data, Iterable[Data]]) -> Union[Data, List[Data]]:
        """Same as process_data(), with the difference that an Iterable is accepted."""
        if isinstance(data, Data):
            return self.process_data(data)
        return [self.process_data(d) for d in data]

    def process_data(self, data: Data) -> Data:
        """
        Perform a transformation on an input Data object. This should never alter the
        Data or its properties in place, instead returning a copy or view of the input.

        Args:
            data: The data to be transformed. Must not be altered in place.

        Returns:
            A copy of the input Data, potentially transformed or enhanced in some way defined by this PipelineStep.
        """
        if isinstance(data, Dataset):
            return self.process_dataset(data)
        if isinstance(data, DimcatResource):
            return self.process_resource(data)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        resources = list(new_dataset.iter_features(self.features))
        new_package = self._make_new_package()
        for resource in resources:
            new_resource = self.process_resource(resource)
            new_package.add_resource(new_resource)
        if len(new_package) < len(resources):
            if len(new_package) == 0:
                self.logger.warning(
                    f"None of the {len(resources)} features were successfully transformed."
                )
            else:
                self.logger.warning(
                    f"Transformation was successful only on {len(new_package)} of the "
                    f"{len(resources)} features."
                )
        if self.is_transformation:
            new_dataset.outputs.replace_package(new_package)
        else:
            new_dataset.outputs.extend_package(new_package)
        return new_dataset

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        self.check_dataset(dataset)
        return self._process_dataset(dataset)

    def _process_resource(self, resource: DimcatResource) -> DimcatResource:
        """Apply this PipelineStep to a :class:`Resource` and return a copy containing the output(s)."""
        resource = self.pre_process_resource(resource)
        result = self.dispatch(resource)
        return self.post_process_result(result)

    def process_resource(self, resource: DimcatResource) -> DimcatResource:
        self.check_resource(resource)
        return self._process_resource(resource)

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Creates a unique name for the new resource based on the input resource."""
        return resource.resource_name


class FeatureExtractor(PipelineStep):
    output_package_name = "features"

    class Schema(PipelineStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            validate=validate.Length(min=1),
        )

    @property
    def features(self) -> List[DimcatConfig]:
        return self._features

    @features.setter
    def features(self, features):
        configs = features_argument2config_list(features)
        if len(self._features) > 0:
            self.logger.info(
                f"Previously selected features {self._features} overwritten by {configs}."
            )
        self._features = configs


class DimcatObjectField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if isinstance(value, DimcatConfig):
            return dict(value)
        return value.to_dict()

    def _deserialize(self, value, attr, data, **kwargs):
        return deserialize_dict(value)


class Pipeline(PipelineStep):
    class Schema(PipelineStep.Schema):
        steps = fields.List(DimcatObjectField())

    def __init__(
        self,
        steps: Optional[
            PipelineStep | DimcatConfig | Iterable[PipelineStep | DimcatConfig]
        ],
        **kwargs,
    ):
        self._steps: List[PipelineStep] = []
        if steps is not None:
            self.steps = steps

    @property
    def steps(self) -> List[PipelineStep]:
        return self._steps

    @steps.setter
    def steps(
        self, steps: PipelineStep | DimcatConfig | Iterable[PipelineStep | DimcatConfig]
    ) -> None:
        if isinstance(steps, (DimcatObject, dict)):
            steps = [steps]
        for step in steps:
            self.add_step(step)

    def add_step(self, step: PipelineStep | DimcatConfig) -> None:
        if isinstance(step, dict):
            assert "dtype" in step, (
                "PipelineStep dict must be config-like and have a 'dtype' key mapping to the "
                "name of a PipelineStep."
            )
            step = DimcatConfig(step)
        if isinstance(step, DimcatConfig):
            step = step.create()
        if not isinstance(step, PipelineStep):
            raise TypeError(f"Pipeline acceppts only PipelineSteps, not {type(step)}.")
        self._steps.append(step)
