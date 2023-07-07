from __future__ import annotations

import logging
from typing import (
    ClassVar,
    Iterable,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    overload,
)

from dimcat.base import DimcatConfig, DimcatObject, get_class
from dimcat.data.base import Data
from dimcat.data.dataset.base import Dataset, DimcatPackage
from dimcat.data.resources.base import (
    DimcatResource,
    Resource,
    ResourceSpecs,
    resource_specs2resource,
)
from dimcat.data.resources.features import (
    FeatureName,
    FeatureSpecs,
    features_argument2config_list,
)
from dimcat.data.resources.utils import ensure_level_named_piece
from dimcat.exceptions import (
    EmptyDatasetError,
    EmptyResourceError,
    FeatureUnavailableError,
    NoFeaturesActiveError,
    ResourceNotProcessableError,
)
from marshmallow import fields, pre_load

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=Data)


class PipelineStep(DimcatObject):
    """
    This base class unites all classes able to transform some data in a pre-defined way.

    The initializer will set some parameters of the processing, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.
    """

    new_dataset_type: Optional[ClassVar[Type[Dataset]]] = None
    """If specified, :meth:`process_dataset` will return Datasets of this type, otherwise same as input type."""

    new_resource_type: Optional[ClassVar[Type[DimcatResource]]] = None
    """If specified, :meth:`process_resource` will return Resources of this type, otherwise same as input type."""

    applicable_to_empty_datasets: ClassVar[bool] = True
    """If False, :meth:`check_dataset` will raise an EmptyDatasetError if no data has been loaded yet. This makes sense
    for PipelineSteps that are dependent on the data, e.g. because they use :meth:`fit_to_dataset`."""

    class Schema(DimcatObject.Schema):
        pass

    @property
    def is_transformation(self) -> Literal[False]:
        """True if this PipelineStep transforms features, replacing the dataset.outputs['features'] package."""
        return False

    def check_dataset(self, dataset: Dataset) -> None:
        """Check if the dataset is eligible for processing.

        Raises:
            TypeError: if the given dataset is not a Dataset
            EmptyDatasetError: if :attr:`applicable_to_empty_datasets` is False and the given dataset is empty
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset)}")
        if not self.applicable_to_empty_datasets:
            if dataset.n_features_available == 0:
                raise EmptyDatasetError

    def check_resource(self, resource: DimcatResource) -> None:
        """Check if the resource is eligible for processing.

        Raises:
            TypeError: if the given resource is not a DimcatResource
            EmptyResourceError: if the given resource is empty
        """
        if not isinstance(resource, DimcatResource):
            raise TypeError(f"Expected DimcatResource, got {type(resource)}")
        if resource.is_empty:
            raise EmptyResourceError

    def fit_to_dataset(self, dataset: Dataset) -> None:
        """Adjust the PipelineStep to the passed dataset.

        Args:
            dataset: The dataset to adjust to.
        """
        return

    def _make_new_resource(self, resource: DimcatResource) -> DimcatResource:
        """Dispatch the passed resource to the appropriate method."""
        resource_constructor = self._get_new_resource_type(resource)
        # This is where the input resource is being processed
        resource_name = self.resource_name_factory(resource)
        new_resource = resource_constructor.from_resource(
            resource, resource_name=resource_name
        )
        self.logger.debug(
            f"Created new resource {new_resource} of type {resource_constructor.name}."
        )
        return new_resource

    def _get_new_resource_type(self, resource: DimcatResource) -> Type[DimcatResource]:
        if self.new_resource_type is None:
            resource_constructor: Type[DimcatResource] = resource.__class__
        else:
            resource_constructor: Type[DimcatResource] = self.new_resource_type
        return resource_constructor

    def _make_new_dataset(self, dataset: Dataset) -> Dataset:
        if self.new_dataset_type is None:
            dataset_constructor: Type[Dataset] = dataset.__class__
        else:
            dataset_constructor: Type[Dataset] = self.new_dataset_type
        new_dataset = dataset_constructor.from_dataset(dataset)
        self.logger.debug(
            f"Created new dataset {new_dataset} of type {dataset_constructor.__name__}."
        )
        return new_dataset

    def _post_process_result(self, result: DimcatResource) -> DimcatResource:
        """Perform some post-processing on a resource after processing it."""
        return result

    def _pre_process_resource(self, resource: DimcatResource) -> DimcatResource:
        """Perform some pre-processing on a resource before processing it."""
        resource.load()
        if "piece" not in resource.index.names:
            # ToDo: This can go once the feature extractor does this systematically
            resource.df.index, _ = ensure_level_named_piece(resource.df.index)
        return resource

    @overload
    def process(self, data: D) -> D:
        ...

    @overload
    def process(self, data: Iterable[D]) -> List[D]:
        ...

    def process(self, data: D | Iterable[D]) -> D | List[D]:
        """Same as process_data(), with the difference that an Iterable is accepted."""
        if isinstance(data, Data):
            return self.process_data(data)
        return [self.process_data(d) for d in data]

    @overload
    def process_data(self, data: Dataset) -> Dataset:
        ...

    @overload
    def process_data(self, data: DimcatResource) -> DimcatResource:
        ...

    def process_data(self, data: Dataset | DimcatResource) -> Dataset | DimcatResource:
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
        raise TypeError(f"Expected Dataset or DimcatResource, got {type(data)}")

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        # create a new package and add it to the dataset
        return new_dataset

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        self.check_dataset(dataset)
        return self._process_dataset(dataset)

    def _process_resource(self, resource: Resource) -> Resource:
        """Apply this PipelineStep to a :class:`Resource` and return a copy containing the output(s)."""
        resource = self._pre_process_resource(resource)
        result = self._make_new_resource(resource)
        return self._post_process_result(result)

    def process_resource(self, resource: ResourceSpecs) -> DimcatResource:
        resource = resource_specs2resource(resource)
        self.check_resource(resource)
        return self._process_resource(resource)

    def resource_name_factory(self, resource: DimcatResource) -> str:
        """Creates a unique name for the new resource based on the input resource."""
        return resource.resource_name


class FeatureProcessingStep(PipelineStep):
    """
    This class unites all PipelineSteps that work on one or all features that can be or have been extracted from a
    Dataset. They can be instantiated with the ``features`` argument, with the behaviour defined by class variables.

    """

    allowed_features: Optional[ClassVar[Tuple[FeatureName]]] = None
    """If set, this FeatureProcessingStep can only be initialized with features that are in this tuple."""

    output_package_name: Optional[str] = None
    """Name of the package in which to store the outputs of this step. If None, the PipeLine step will replace the
    'features' package of the given dataset. FeatureProcessingSteps that replace the 'features' packages are called
    transformations internally."""

    requires_at_least_one_feature: ClassVar[bool] = False
    """If set to True, this PipelineStep cannot be initialized without specifying at least one feature."""

    class Schema(PipelineStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            allow_none=True,
        )

        @pre_load
        def deal_with_single_item(self, data, **kwargs):
            if isinstance(data, MutableMapping) and "features" in data:
                if isinstance(data["features"], list):
                    features = data["features"]
                else:
                    features = [data["features"]]
                feature_list = []
                for feature in features:
                    if isinstance(feature, dict):
                        # this seems to be a manually created config
                        feature = DimcatConfig(feature)
                    feature_list.append(feature)
                data = dict(
                    data, features=feature_list
                )  # make sure to not modify data inplace
            return data

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
        configs = features_argument2config_list(
            features, allowed_features=self.allowed_features
        )
        if len(self._features) > 0:
            self.logger.info(
                f"Previously selected features {self._features} overwritten by {configs}."
            )
        self._features = configs

    @property
    def is_transformation(self) -> bool:
        """True if this PipelineStep replaces the :attr:`output_package_name` in dataset.outputs rather than extending
        it. Currently, this is the case only if :attr:`output_package_name` 'features' or None, defaulting to
        'features')."""
        return (
            self.output_package_name is None or self.output_package_name == "features"
        )

    def check_dataset(self, dataset: Dataset) -> None:
        """Check if the dataset is eligible for processing.

        Raises:
            TypeError: if the given dataset is not a Dataset
            EmptyDatasetError: if :attr:`applicable_to_empty_datasets` is False and the given dataset is empty
            NoFeaturesActiveError: if :attr:`requires_at_least_one_feature` is True and no features are active
            FeatureUnavailableError: if any of the required features is not available in the dataset.
        """
        super().check_dataset(dataset)
        required_features = self.get_feature_specs()
        if self.requires_at_least_one_feature:
            if len(required_features) == 0 and dataset.n_active_features == 0:
                raise NoFeaturesActiveError
        for feature in required_features:
            if not dataset.check_feature_availability(feature):
                raise FeatureUnavailableError

    def check_resource(self, resource: DimcatResource) -> None:
        """Check if the resource is eligible for processing.

        Raises:
            TypeError: if the given resource is not a DimcatResource
            EmptyResourceError: if the given resource is empty
            FeatureNotProcessableError: if the given resource cannot be processed by this step
        """
        super().check_resource(resource)
        if self.allowed_features:
            if not any(
                issubclass(resource.__class__, get_class(f))
                for f in self.allowed_features
            ):
                raise ResourceNotProcessableError(resource.name, self.name)

    def _iter_features(self, dataset: Dataset) -> Iterable[DimcatResource]:
        """Iterate over all features that are required for this PipelineStep.
        If :meth:`get_feature_specs` returns None, the Dataset will return an iterator over all active features.
        """
        feature_specs = self.get_feature_specs()
        return dataset.iter_features(feature_specs)

    def get_feature_specs(self) -> List[DimcatConfig]:
        """Return a list of feature names required for this PipelineStep."""
        return self.features

    def _make_new_package(self, package_name: Optional[str] = None) -> DimcatPackage:
        """Create a new package for the output of this PipelineStep, based on :attr:`output_package_name`."""
        if package_name is not None:
            return DimcatPackage(package_name=package_name)
        if self.output_package_name is None:
            return DimcatPackage(package_name="features")
        return DimcatPackage(package_name=self.output_package_name)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = super()._process_dataset(dataset)
        resources = self._iter_features(new_dataset)
        new_package = self._make_new_package()
        n_processed = 0
        for n_processed, resource in enumerate(resources, 1):
            new_resource = self.process_resource(resource)
            new_package.add_resource(new_resource)
        if new_package.n_resources < n_processed:
            if new_package.n_resources == 0:
                self.logger.warning(
                    f"None of the {n_processed} features were successfully transformed."
                )
            else:
                self.logger.warning(
                    f"Transformation was successful only on {new_package.n_resources} of the "
                    f"{n_processed} features."
                )
        if self.is_transformation:
            new_dataset.outputs.replace_package(new_package)
        else:
            new_dataset.outputs.extend_package(new_package)
        new_dataset._pipeline.add_step(self)
        return new_dataset
