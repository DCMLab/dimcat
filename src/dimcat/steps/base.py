from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from itertools import repeat
from pprint import pformat
from typing import (
    ClassVar,
    Iterable,
    Iterator,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import marshmallow as mm
import pandas as pd
from dimcat.base import (
    DimcatConfig,
    DimcatObject,
    ObjectEnum,
    get_class,
    get_schema,
    is_instance_of,
)
from dimcat.data.base import Data
from dimcat.data.datasets.base import Dataset
from dimcat.data.datasets.processed import _AnalyzedMixin
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.resources.base import (
    DR,
    F,
    FeatureName,
    Resource,
    ResourceSpecs,
    resource_specs2resource,
)
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs
from dimcat.data.resources.utils import (
    feature_specs2config,
    features_argument2config_list,
)
from dimcat.dc_exceptions import (
    EmptyDatasetError,
    EmptyResourceError,
    FeatureUnavailableError,
    NoFeaturesActiveError,
    ResourceAlreadyTransformed,
    ResourceNotProcessableError,
)
from dimcat.dc_warnings import OrderOfPipelineStepsWarning
from marshmallow import fields, pre_load

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=Data)


class PipelineStep(DimcatObject):
    """
    This base class unites all classes able to transform some data in a pre-defined way.

    The initializer will set some parameters of the processing, and then the
    :meth:`process` method is used to transform an input Data object, returning a copy.
    """

    _new_dataset_type: ClassVar[Optional[Type[Dataset]]] = None
    """If specified, :meth:`process_dataset` will return Datasets of this type, otherwise same as input type."""

    _new_resource_type: ClassVar[Optional[Type[DR]]] = None
    """If specified, :meth:`process_resource` will return Resources of this type, otherwise same as input type."""

    _applicable_to_empty_datasets: ClassVar[bool] = True
    """If False, :meth:`check_dataset` will raise an EmptyDatasetError if no data has been loaded yet. This makes sense
    for PipelineSteps that are dependent on the data, e.g. because they use :meth:`fit_to_dataset`."""

    class Schema(DimcatObject.Schema):
        """PipelineSteps do not depend on previously serialized data, so their serialization can be validated by
        default after dumping them to a dict-like structure. For Data objects, this default is safe only for their
        PickleSchema, which PipelineSteps do not use.
        """

        @mm.post_dump()
        def validate_dump(self, data, **kwargs):
            """Make sure to never return invalid serialization data."""
            if "dtype" not in data:
                msg = (
                    f"{self.name}: The serialized data doesn't have a 'dtype' field, meaning that DiMCAT would "
                    f"not be able to deserialize it."
                )
                raise mm.ValidationError(msg)
            dtype_schema = get_schema(data["dtype"])
            report = dtype_schema.validate(data)
            if report:
                raise mm.ValidationError(
                    f"Dump of {data['dtype']} created with a {self.name} could not be validated by "
                    f"{dtype_schema.name}."
                    f"\n\nDUMP:\n{pformat(data, sort_dicts=False)}"
                    f"\n\nREPORT:\n{pformat(report, sort_dicts=False)}"
                )
            return data

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
        if isinstance(dataset, _AnalyzedMixin) and not is_instance_of(
            self.name, "Analyzer"
        ):
            warnings.warn(
                f"You're applying a {self.name} to an AnalyzedDataset. As things stand, Analyzers should "
                f"always be the last thing to be applied to a Dataset. Consider a different Pipeline.",
                OrderOfPipelineStepsWarning,
            )
        if not self._applicable_to_empty_datasets:
            if dataset.n_features_available == 0:
                raise EmptyDatasetError

    def check_resource(self, resource: Resource) -> None:
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

    def _make_new_resource(self, resource: DimcatResource) -> DR:
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

    def _get_new_resource_type(self, resource: DimcatResource) -> Type[DR]:
        if self._new_resource_type is None:
            resource_constructor = resource.__class__
        else:
            resource_constructor = self._new_resource_type
        return resource_constructor

    def _make_new_dataset(self, dataset: Dataset) -> Dataset:
        if self._new_dataset_type is None:
            dataset_constructor: Type[Dataset] = dataset.__class__
        else:
            dataset_constructor: Type[Dataset] = self._new_dataset_type
        new_dataset = dataset_constructor.from_dataset(dataset)
        self.logger.debug(
            f"Created new dataset {new_dataset} of type {dataset_constructor.__name__}."
        )
        return new_dataset

    def _post_process_result(
        self,
        result: DR,
        original_resource: DimcatResource,
    ) -> DR:
        """Perform some post-processing on a resource after processing it."""
        return result

    def _pre_process_resource(self, resource: DR) -> DR:
        """Perform some pre-processing on a resource before processing it."""
        return resource

    @overload
    def process(self, data: D) -> D:
        ...

    @overload
    def process(self, data: List[D] | Tuple[D]) -> List[D]:
        ...

    @overload
    def process(self, *data: D) -> List[D]:
        ...

    def process(self, *data: D) -> D | List[D]:
        """Same as process_data(), with the difference that arbitrarily many objects are accepted."""
        if not data:
            raise ValueError("Please pass a Dataset or a Resource to process.")
        if len(data) == 1:
            single_obj = data[0]
            if isinstance(single_obj, (Tuple, List)):
                data = single_obj
            else:
                # a single object was given which is neither a list nor a tuple, this is the
                # case where not to return a list
                return self.process_data(single_obj)
        return [self.process_data(d) for d in data]

    @overload
    def process_data(self, data: Dataset) -> Dataset:
        ...

    @overload
    def process_data(self, data: DimcatResource) -> DR:
        ...

    def process_data(self, data: Dataset | DimcatResource) -> Dataset | DR:
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

    def _process_resource(self, resource: Resource) -> DR:
        """Apply this PipelineStep to a :class:`Resource` and return a copy containing the output(s)."""
        resource = self._pre_process_resource(resource)
        result = self._make_new_resource(resource)
        return self._post_process_result(result, resource)

    def process_resource(self, resource: ResourceSpecs) -> DR:
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

    _allowed_features: ClassVar[Optional[Tuple[FeatureName, ...]]] = None
    """If set, this FeatureProcessingStep can only be initialized with features that are in this tuple."""

    _output_package_name: ClassVar[Optional[str]] = None
    """Name of the package in which to store the outputs of this step. If None, the PipeLine step will replace the
    'features' package of the given dataset. FeatureProcessingSteps that replace the 'features' packages are called
    transformations internally."""

    _requires_at_least_one_feature: ClassVar[bool] = False
    """If set to True, this PipelineStep cannot be initialized without specifying at least one feature."""

    class Schema(PipelineStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            allow_none=True,
            metadata=dict(
                expose=True,
                description="The Feature objects you want this PipelineStep to process. If not specified, "
                "the step will try to process all features in a given Dataset's Outputs catalog.",
            ),
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
                    feature_list.append(feature_specs2config(feature))
                data = dict(
                    data, features=feature_list
                )  # make sure to not modify data inplace
            return data

    def __init__(
        self, features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None, **kwargs
    ):
        self._features: List[DimcatConfig] = []
        self.features = features
        if len(kwargs) > 0:
            self.logger.warning(f"Ignored unknown keyword arguments: {kwargs}")

    @property
    def features(self) -> List[DimcatConfig]:
        """The Feature objects you want this PipelineStep to process. If not specified, the step will try to process
        all features in a given Dataset's Outputs catalog."""
        return self._features

    @features.setter
    def features(self, features):
        configs = features_argument2config_list(
            features, allowed_features=self._allowed_features
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
            self._output_package_name is None or self._output_package_name == "features"
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
        if self._requires_at_least_one_feature:
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
        if self._allowed_features:
            if not any(
                issubclass(resource.__class__, get_class(f))
                for f in self._allowed_features
            ):
                raise ResourceNotProcessableError(resource.name, self.name)

    def _iter_features(self, dataset: Dataset) -> Iterator[DimcatResource]:
        """Iterate over all features that are required for this PipelineStep.
        If :meth:`get_feature_specs` returns None, the Dataset will return an iterator over all active features.
        """
        feature_specs = self.get_feature_specs()
        return dataset.iter_features(feature_specs)

    def _iter_resources(self, dataset: Dataset) -> Iterator[Tuple[str, DimcatResource]]:
        """Iterate over all resources in the dataset's OutputCatalog."""
        return dataset.outputs.iter_resources()

    def get_feature_specs(self) -> List[DimcatConfig]:
        """Return a list of feature names required for this PipelineStep."""
        return self.features

    def _make_new_package(self, package_name: Optional[str] = None) -> DimcatPackage:
        """Create a new package for the output of this PipelineStep, based on :attr:`output_package_name`."""
        if package_name is not None:
            return DimcatPackage(package_name=package_name)
        if self._output_package_name is None:
            return DimcatPackage(package_name="features")
        return DimcatPackage(package_name=self._output_package_name)

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


class ResourceTransformation(FeatureProcessingStep):
    """The subclasses either transform the features specified upon initialization, returning a Dataset containing
    only these, or, if no features are specified, transform all resources in the outputs catalog.
    """

    def _make_new_resource(self, resource: F) -> DR:
        """Create a new resource by transforming the existing one."""
        result_constructor = self._get_new_resource_type(resource)
        result_df = self.transform_resource(resource)
        result_name = self.resource_name_factory(resource)
        try:
            new_resource = result_constructor.from_resource_and_dataframe(
                resource=resource, df=result_df, resource_name=result_name
            )
        except Exception as e:
            print(
                f"Calling {result_constructor.name}.from_dataframe() on the following DataFrame that has the index "
                f"levels {resource.get_level_names()} resulted in the exception\n{e!r}:"
            )
            print(result_df)
            raise
        # new_resource = result_constructor.from_dataframe(
        #     df=result_df,
        #     resource_name=result_name,
        #     **resource_kwargs
        # )
        # print(f"NEW RESOURCE STATUS: {new_resource.status}")
        self.logger.debug(
            f"Created new resource {new_resource} of type {result_constructor.name}."
        )
        return new_resource

    def _pre_process_resource(self, resource: DR) -> DR:
        """Perform some pre-processing on a resource before processing it."""
        resource = super()._pre_process_resource(resource)
        resource.load()
        return resource

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        new_dataset._pipeline.add_step(self)
        feature_specs = self.get_feature_specs()
        if feature_specs:
            resource_iterator = self._iter_features(new_dataset)
            package_name_resource_iterator = zip(repeat("features"), resource_iterator)
        else:
            package_name_resource_iterator = self._iter_resources(new_dataset)
        processed_resources = defaultdict(list)
        for package_name, resource in package_name_resource_iterator:
            try:
                new_resource = self.process_resource(resource)
            except ResourceNotProcessableError as e:
                self.logger.warning(
                    f"Resource {resource.resource_name!r} could not be transformed and is not included in "
                    f"the new Dataset due to the following error: {e!r}"
                )
                continue
            except ResourceAlreadyTransformed:
                new_resource = resource
            processed_resources[package_name].append(new_resource)
        for package_name, resources in processed_resources.items():
            new_package = self._make_new_package(package_name)
            new_package.extend(resources)
            n_processed = len(resources)
            if new_package.n_resources < n_processed:
                if new_package.n_resources == 0:
                    self.logger.warning(
                        f"None of the {n_processed} {package_name} were successfully transformed."
                    )
                else:
                    self.logger.warning(
                        f"Transformation was successful only on {new_package.n_resources} of the "
                        f"{n_processed} features."
                    )
            new_dataset.outputs.replace_package(new_package)
        return new_dataset

    def transform_resource(self, resource: DimcatResource) -> pd.DataFrame:
        """Apply the transformation to a Resource and return the transformed dataframe."""
        return resource.df


StepSpecs: TypeAlias = Union[
    PipelineStep | Type[PipelineStep] | DimcatConfig | dict | ObjectEnum | str
]
