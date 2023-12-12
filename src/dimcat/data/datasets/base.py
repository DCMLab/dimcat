"""
The principal Data object is called Dataset and is the one that users will interact with the most.
The Dataset provides convenience methods that are equivalent to applying the corresponding PipelineStep.
Every PipelineStep applied to it will return a new Dataset that can be serialized and deserialized to re-start the
pipeline from that point. To that aim, every Dataset stores a serialization of the applied PipelineSteps and of the
original Dataset that served as initial input. This initial input is specified as a DimcatCatalog which is a
collection of DimcatPackages, each of which is a collection of DimcatResources, as defined by the Frictionless
Data specifications. The preferred structure of a DimcatPackage is a .zip and a datapackage.json file,
where the former contains one or several .tsv files (resources) described in the latter. Since the data that DiMCAT
transforms and analyzes comes from very heterogeneous sources, each original corpus is pre-processed and stored as a
frictionless data package together with the metadata relevant for reproducing the pre-processing.
It follows that the Dataset is mainly a container for DimcatResources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    overload,
)

import marshmallow as mm
from dimcat.base import DimcatConfig, DimcatObjectField, get_class, resolve_object_specs
from dimcat.data.base import Data
from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.catalogs.inputs import InputsCatalog
from dimcat.data.catalogs.outputs import OutputsCatalog
from dimcat.data.packages.base import Package, PackageSpecs
from dimcat.data.packages.dc import DimcatPackage
from dimcat.data.resources.base import F, FeatureName
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs
from dimcat.data.resources.features import Metadata
from dimcat.data.resources.utils import (
    feature_specs2config,
    features_argument2config_list,
)
from dimcat.dc_exceptions import NoMatchingResourceFoundError, PackageNotFoundError

if TYPE_CHECKING:
    from dimcat.data.resources.results import Result
    from dimcat.steps.base import StepSpecs
    from dimcat.steps.loaders.base import Loader
    from dimcat.steps.pipelines import Pipeline

logger = logging.getLogger(__name__)


# region DimcatPackage


# endregion DimcatPackage
# region DimcatCatalog


# endregion DimcatCatalog
# region Dataset

Ds = TypeVar("Ds", bound="Dataset")


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

    @classmethod
    def from_catalogs(
        cls,
        inputs: DimcatCatalog | List[DimcatPackage],
        outputs: DimcatCatalog | List[DimcatPackage],
        pipeline: Optional[Pipeline] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        """Instantiate by copying existing catalogs."""
        new_dataset = cls(basepath=basepath, **kwargs)
        if pipeline is not None:
            new_dataset._pipeline = pipeline
        new_dataset.inputs.basepath = inputs.basepath
        new_dataset.outputs.basepath = outputs.basepath
        new_dataset.inputs.extend(inputs)
        new_dataset.outputs.extend(outputs)
        return new_dataset

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs) -> Dataset:
        """Instantiate from this Dataset by copying its fields, empty fields otherwise."""
        return cls.from_catalogs(
            inputs=dataset.inputs,
            outputs=dataset.outputs,
            pipeline=dataset.pipeline,
            **kwargs,
        )

    @classmethod
    def from_loader(cls, loader: Loader) -> Dataset:
        dataset = cls() if not loader.basepath else cls(basepath=loader.basepath)
        return loader.process_dataset(dataset)

    @classmethod
    def from_package(cls, package: PackageSpecs) -> Dataset:
        """Instantiate from a PackageSpecs by loading it into the inputs catalog."""
        dataset = cls()
        dataset.load_package(package=package)
        return dataset

    class PickleSchema(Data.Schema):
        """Dataset serialization schema."""

        inputs = mm.fields.Nested(DimcatCatalog.Schema, load_default=[])
        outputs = mm.fields.Nested(DimcatCatalog.Schema, load_default=[])
        pipeline = (
            DimcatObjectField()
        )  # mm.fields.Nested(Pipeline.Schema) would cause circular import

        @mm.post_load
        def init_object(self, data, **kwargs) -> Dataset:
            return Dataset.from_catalogs(
                inputs=data["inputs"],
                outputs=data["outputs"],
            )

    class Schema(PickleSchema, Data.Schema):
        pass

    def __init__(
        self,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of.

        Args:
            **kwargs: Dataset is cooperative and calls super().__init__(data=dataset, **kwargs)
        """
        self._inputs = InputsCatalog(basepath=basepath)
        self._outputs = OutputsCatalog(basepath=basepath)
        self._pipeline: Pipeline = None
        self.reset_pipeline()
        super().__init__(basepath=basepath, **kwargs)  # calls the Mixin's __init__

    @property
    def extractable_features(self) -> Set[FeatureName]:
        """The dtypes of all features that can be extracted from the facet resources included in the input packages."""
        f_name_sets = [package.extractable_features for package in self.inputs]
        return set().union(*f_name_sets)

    @property
    def inputs(self) -> InputsCatalog:
        """The inputs catalog."""
        return self._inputs

    @property
    def n_active_features(self) -> int:
        """The number of features extracted and stored in the outputs catalog."""
        if self.outputs.has_package("features"):
            return self.outputs.get_package_by_name("features").n_resources
        return 0

    @property
    def n_features_available(self) -> int:
        """The number of features (potentially) available from this Dataset."""
        # ToDo: Needs to take into account overlap between packages
        return sum(package.n_resources for package in self.inputs)

    @property
    def outputs(self) -> OutputsCatalog:
        """The outputs catalog."""
        return self._outputs

    @property
    def pipeline(self) -> Pipeline:
        """A copy of the pipeline representing the steps that have been applied to this Dataset so far.
        To add a PipelineStep to the pipeline of this Dataset, use :meth:`apply`.
        """
        Constructor = get_class("Pipeline")
        return Constructor.from_pipeline(self._pipeline)

    def add_output(
        self,
        resource: DimcatResource,
        package_name: Optional[str] = None,
    ) -> None:
        """Adds a resource to the outputs catalog.

        Args:
            resource: Resource to be added.
            package_name:
                Name of the package to add the resource to.
                If unspecified, the package is inferred from the resource type.
        """
        if package_name is None:
            if resource.name == "DimcatResource":
                raise ValueError(
                    "Cannot infer package name from resource type 'DimcatResource'. "
                    "Please specify package_name."
                )
            if isinstance(resource, Result):
                package_name = "results"
            else:
                raise NotImplementedError(
                    f"Cannot infer package name from resource type {type(resource)}."
                )
        self.outputs.add_resource(resource=resource, package_name=package_name)

    @overload
    def apply_step(self, step: StepSpecs | List | Tuple) -> Ds:
        ...

    @overload
    def apply_step(self, *step: StepSpecs) -> Ds:
        ...

    def apply_step(self, *step: StepSpecs) -> Ds:
        """Applies one or several pipeline steps to this dataset.For backward compatibility, when only a single
        argument is passed, the method accepts it to be a list or tuple of step specs, too.
        """
        if len(step) == 1:
            single_step = step[0]
            if isinstance(single_step, (list, tuple)):
                return self.apply_step(*single_step)
            step_obj = resolve_object_specs(single_step, "PipelineStep")
            return step_obj.process_dataset(self)
        Constructor = get_class("Pipeline")
        pipeline = Constructor(steps=step)
        return pipeline.process_dataset(self)

    def check_feature_availability(self, feature: FeatureSpecs) -> bool:
        """Checks whether the given feature specs are available from this Dataset.

        Args:
            feature: FeatureSpecs to be checked.
        """
        # ToDo: feature_config = feature_specs2config(feature)
        return True

    def copy(self) -> Dataset:
        """Returns a copy of this Dataset."""
        return Dataset.from_dataset(self)

    def _extract_feature(self, feature_config: DimcatConfig) -> F:
        """Extracts a feature from the Dataset's input catalog, sends it through its pipeline and returns the result,
        without storing it.

        Args:
            feature: FeatureSpecs to be extracted.
        """
        extracted = self.inputs.extract_feature(feature_config)
        if len(self._pipeline) == 0:
            self.logger.debug("Pipeline empty, returning extracted feature as is.")
            return extracted
        self.logger.debug(
            f"Applying pipeline to extracted feature: {self._pipeline.steps}."
        )
        return self._pipeline._process_resource(
            extracted,
            ignore_exceptions=True,
            skip_step_types=["FeatureExtractor"],
        )

    def extract_feature(self, feature: FeatureSpecs) -> F:
        """Extracts a feature from this Dataset's input catalog, sends it through its pipeline, adds the result to the
        OutputsCatalog, and adds the corresponding FeatureExtractor to the dataset's pipeline.

        Args:
            feature: FeatureSpecs to be extracted.
        """
        feature_config = feature_specs2config(feature)
        Constructor = get_class("FeatureExtractor")
        feature_extractor = Constructor(feature_config)
        extracted = self._extract_feature(feature_config)
        self.add_output(resource=extracted, package_name="features")
        self._pipeline.add_step(feature_extractor)
        return extracted

    def get_feature(self, feature: Optional[FeatureSpecs] = None) -> F:
        """High-level method that first looks up a feature fitting the specs in the outputs catalog,
        and adds a FeatureExtractor to the dataset's pipeline otherwise."""
        feature_config = feature_specs2config(feature)
        try:
            return self.outputs.get_feature(feature_config)
        except (
            PackageNotFoundError,
            NoMatchingResourceFoundError,
            NoMatchingResourceFoundError,
        ):
            pass
        return self.extract_feature(feature_config)

    def iter_features(
        self, features: FeatureSpecs | Iterable[FeatureSpecs] = None
    ) -> Iterator[DimcatResource]:
        if not features:
            if self.n_active_features == 0:
                yield from []
            else:
                yield from self.outputs.get_package_by_name("features")
        configs = features_argument2config_list(features)
        for config in configs:
            yield self.get_feature(config)

    # def make_features_package(
    #     self,
    #     features: FeatureSpecs | Iterable[FeatureSpecs] = None,
    # ) -> DimcatPackage:
    #     """Returns a DimcatPackage containing the requested or currently active features.
    #
    #     Args:
    #         features:
    #
    #     Returns:
    #
    #     """
    #     if not features:
    #         if self.n_active_features == 0:
    #             raise NoFeaturesActiveError
    #         return self.outputs.get_package_by_name("features")
    #     new_package = DimcatPackage(package_name="features")
    #     for feature in self.iter_features(features):
    #         new_package.add_resource(feature)
    #     return new_package

    def get_metadata(
        self,
        raw: bool = False,
    ) -> Metadata:
        metadata = self.inputs.get_metadata()
        if raw:
            return metadata
        else:
            return self.get_feature(DimcatConfig(dtype="Metadata"))

    def load(
        self,
        package: PackageSpecs,
    ):
        """High-level method that tries to infer what it is that you want to load."""
        self.load_package(package=package)

    def load_package(
        self,
        package: PackageSpecs,
        package_name: Optional[str] = None,
        **options,
    ):
        """Loads a package into the inputs catalog.

        Args:
            package: Typically a path to a datapackage.json descriptor.
            package_name:
                If you want to assign a different name to the package than given in the descriptor. The package_name
                is relevant for addressing the package in the catalog.
            **options:

        Returns:

        """
        if isinstance(package, (str, Path)):
            package = DimcatPackage.from_descriptor_path(package, **options)
        elif isinstance(package, dict):
            package = DimcatPackage.from_descriptor(package, **options)
        elif isinstance(package, Package):
            pass
        else:
            raise TypeError(
                f"Package must be a path to a descriptor or a Package instance, not {type(package)}."
            )
        if package_name is None:
            package_name = package.name
            assert (
                package_name is not None
            ), "Descriptor did not contain package name and no name was given."
        else:
            package.package_name = package_name
        self.inputs.add_package(package)
        self.logger.debug(
            f"Package with basepath {package.basepath} loaded into inputs catalog "
            f"with basepath {self.inputs.basepath}."
        )

    def load_feature(self, feature: FeatureSpecs) -> F:
        """ToDo: Harmonize with FeatureExtractor"""
        feature = self.get_feature(feature)
        feature.load()
        return feature

    def reset_pipeline(self) -> None:
        """Resets the pipeline by replacing it with an empty one."""
        if self._pipeline is None:
            self.logger.debug("Initializing empty Pipeline.")
        else:
            self.logger.debug("Resetting Pipeline.")
        Constructor = get_class("Pipeline")
        self._pipeline = Constructor()

    def summary_dict(self) -> dict:
        """Returns a summary of the dataset."""
        summary = dict(
            inputs=self.inputs.summary_dict(),
            outputs=self.outputs.summary_dict(),
            pipeline=[step.name for step in self._pipeline],
        )
        return summary


# endregion Dataset
