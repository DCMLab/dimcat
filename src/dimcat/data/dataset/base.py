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
import os
from enum import IntEnum, auto
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    overload,
)

import frictionless as fl
import marshmallow as mm
from dimcat.base import DimcatConfig, DimcatObjectField, FriendlyEnum, get_class
from dimcat.data.base import Data
from dimcat.data.resources.base import D, DimcatResource, ResourceStatus, SomeDataframe
from dimcat.data.resources.features import (
    Feature,
    FeatureName,
    FeatureSpecs,
    feature_specs2config,
    features_argument2config_list,
)
from dimcat.data.resources.utils import make_rel_path
from dimcat.exceptions import (
    EmptyCatalogError,
    EmptyPackageError,
    NoFeaturesActiveError,
    NoMatchingResourceFoundError,
    PackageNotFoundError,
    ResourceNotFoundError,
)
from dimcat.utils import check_file_path, check_name, get_default_basepath, resolve_path
from typing_extensions import Self

if TYPE_CHECKING:
    from dimcat.data.resources.results import Result
    from dimcat.steps.base import FeatureStep
    from dimcat.steps.pipelines import Pipeline

logger = logging.getLogger(__name__)


# region DimcatPackage


class AddingBehaviour(FriendlyEnum):
    """The behaviour of a DimcatPackage when adding a resource with incompatible paths."""

    RAISE = "RAISE"
    """Raises an error when adding a resource with an incompatible path."""
    LOAD = "LOAD"
    """Loads the resource from the given path, detaching it from the physical source."""
    SERIALIZE = "SERIALIZE"
    """Load the resource and add a physical copy to the package ZIP."""


class PackageStatus(IntEnum):
    EMPTY = 0
    NOT_SERIALIZED = auto()
    PARTIALLY_SERIALIZED = auto()
    FULLY_SERIALIZED = auto()


class DimcatPackage(Data):
    """Wrapper for a :obj:`frictionless.Package`. The purpose of a Package is to create, load, and store a collection
    of :obj:`DimcatResource` objects. The default way of storing a package is a ``[name.]datapackage.json``
    descriptor and a .zip file containing one .tsv file per DimcatResource contained in the package.

    Attributes
    ----------

    * ``package`` (:obj:`frictionless.Package`) - The frictionless Package object that is wrapped by this class.
    * ``package_name`` (:obj:`str`) - The name of the package that can be used to access it.
    * ``basepath`` (:obj:`str`) - The basepath where the package and its .json descriptor are stored.
    """

    @classmethod
    def from_package(
        cls,
        package: DimcatPackage,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> Self:
        """Create a new DimcatPackage from an existing DimcatPackage.

        Args:
            package:
            package_name:
            basepath:
            auto_validate:

        """
        if not isinstance(package, DimcatPackage):
            raise TypeError(f"Expected a DimcatPackage, got {type(package)!r}")
        fl_package = package._package
        if package_name is None:
            package_name = fl_package.name
        else:
            fl_package.name = package_name
        if basepath is not None:
            fl_package.basepath = basepath
        new_package = cls(package_name=package_name, auto_validate=auto_validate)
        new_package._package = fl_package
        if package._descriptor_filepath is not None:
            new_package._descriptor_filepath = package._descriptor_filepath
        for resource in package._resources:
            new_package._resources.append(resource.copy())
        new_package._status = package._status
        return new_package

    class Schema(Data.Schema):
        package = mm.fields.Method(
            serialize="get_package_descriptor", deserialize="raw"
        )
        basepath = mm.fields.String(allow_none=True)

        def get_package_descriptor(self, obj: DimcatResource):
            if obj.status == PackageStatus.FULLY_SERIALIZED:
                return obj.get_descriptor_filepath()
            descriptor = obj._package.to_descriptor()
            return descriptor

        def raw(self, data):
            return data

        @mm.post_load
        def init_object(self, data, **kwargs):
            if isinstance(data["package"], str) and "basepath" in data:
                descriptor_path = os.path.join(data["basepath"], data["package"])
                data["package"] = descriptor_path
            elif isinstance(data["package"], dict):
                data["package"] = fl.Package(data["package"])
            return super().init_object(data, **kwargs)

    def __init__(
        self,
        package: Optional[fl.Package | str] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        auto_validate: bool = False,
    ) -> None:
        """

        Args:
            package:
                Can be another DimcatPackage, a frictionless.Package (descriptor), or path to a package descriptor.
                If not specified, a new package is created and ``package_name`` needs to be specified.
            package_name:
                Name of the package that can be used to retrieve it. Can be None only if ``package`` is not None.
            basepath:
                The absolute path on the local file system where the package descriptor and all contained resources
                 are stored. If not specified, it will default to

                 * the directory where the package descriptor is located if ``package`` is a path to a package
                 * the current working directory otherwise

            auto_validate:
                By default, the package is validated everytime a resource is added. Set to False to disable this.
        """
        if sum(arg is None for arg in (package_name, package)) == 2:
            raise ValueError(
                "At least one of package_name and package needs to be specified."
            )
        self._package = fl.Package(resources=[])
        self._status = PackageStatus.EMPTY
        self._resources: List[DimcatResource] = []
        self._descriptor_filepath: Optional[str] = None
        self.auto_validate = auto_validate

        if package_name is not None:
            self.package_name = package_name
        if basepath is not None:
            basepath = resolve_path(basepath)
            self.basepath = basepath
        if descriptor_filepath is not None:
            self.descriptor_filepath = descriptor_filepath
        if package is not None:
            self.package = package

        if self.package_name is None:
            raise ValueError(
                "package_name needs to specified, otherwise the DimcatPackage cannot be retrieved"
            )
        if self.basepath is None:
            if self.n_resources > 0:
                raise ValueError(
                    "basepath needs to specified in order to serialize the package and its resources."
                )
            else:
                self.logger.info(
                    "Basepath of empty DimcatPackage will be set based on the first resource added."
                )
        if auto_validate:
            self.validate(raise_exception=True)

    def __getitem__(self, item: str) -> DimcatResource:
        try:
            return self.get_resource(item)
        except Exception as e:
            raise KeyError(str(e)) from e

    def __iter__(self) -> Iterator[DimcatResource]:
        yield from self._resources

    def __len__(self):
        return len(self._resources)

    def __repr__(self):
        values = self._package.to_descriptor()
        values["basepath"] = self.basepath
        return pformat(values, sort_dicts=False)

    def __str__(self):
        values = self._package.to_descriptor()
        values["basepath"] = self.basepath
        return pformat(values, sort_dicts=False)

    @property
    def basepath(self) -> str:
        return self._package.basepath

    @basepath.setter
    def basepath(self, basepath: str) -> None:
        basepath = resolve_path(basepath)
        if self.status > PackageStatus.NOT_SERIALIZED:
            if basepath == self.basepath:
                return
            state = "partially" if PackageStatus.PARTIALLY_SERIALIZED else "fully"
            raise NotImplementedError(
                f"Cannot change the basepath of a package that has already been {state} serialized. Attempted to "
                f"change from {self.basepath!r} to {basepath!r}."
            )
        assert os.path.isdir(
            basepath
        ), f"Basepath {basepath!r} is not an existing directory."
        self._package.basepath = basepath
        for resource in self._resources:
            resource.basepath = basepath  # this is meant to fail RN

    @property
    def descriptor_exists(self) -> bool:
        return os.path.isfile(self.get_descriptor_path())

    @property
    def descriptor_filepath(self) -> str:
        """The path to the descriptor file on disk, relative to the basepath."""
        return self._descriptor_filepath

    @descriptor_filepath.setter
    def descriptor_filepath(self, descriptor_filepath: str) -> None:
        """The path to the descriptor file on disk, relative to the basepath."""
        if self.descriptor_exists:
            if (
                descriptor_filepath == self._descriptor_filepath
                or descriptor_filepath == self.get_descriptor_path()
            ):
                self.logger.info(
                    f"Descriptor filepath for {self.name!r} was already set to {descriptor_filepath!r}."
                )
            else:
                raise RuntimeError(
                    f"Cannot set descriptor_filepath for {self.name!r} to {descriptor_filepath} because it already "
                    f"set to the existing one at {self.get_descriptor_path()!r}."
                )
        if os.path.isabs(descriptor_filepath):
            filepath = check_file_path(
                descriptor_filepath,
                extensions=("package.json", "package.yaml"),
                must_exist=False,
            )
            if self.basepath is None:
                basepath, rel_path = os.path.split(filepath)
                self.basepath = basepath
                self.logger.info(
                    f"The absolute descriptor_path {filepath!r} was used to set the basepath to "
                    f"{basepath!r} and descriptor_filepath to {rel_path}."
                )
            else:
                rel_path = make_rel_path(filepath, self.basepath)
                self.logger.info(
                    f"The absolute descriptor_path {filepath!r} was turned into the relative path "
                    f"{rel_path!r} using the basepath {self.basepath!r}."
                )
            self._descriptor_filepath = rel_path
        else:
            self.logger.info(f"Setting descriptor_filepath to {descriptor_filepath!r}.")
            self._descriptor_filepath = descriptor_filepath

    @property
    def n_resources(self) -> int:
        return len(self._resources)

    @property
    def package(self) -> fl.Package:
        return self._package

    @package.setter
    def package(self, package: fl.Package) -> None:
        if isinstance(package, str):
            descriptor_path = check_file_path(
                package,
                extensions=(
                    "package.json",
                    "package.yaml",
                ),  # both datapackage and package are fine
            )
            fl_package = fl.Package(descriptor_path)
            fl_basepath = fl_package.basepath
        elif isinstance(package, fl.Package):
            fl_package = package
            if not fl_package.resource_names:
                fl_package.clear_resources()  # otherwise the package might be invalid
            fl_basepath = fl_package.basepath
            descriptor_path = fl_package.metadata_descriptor_path
        elif isinstance(package, DimcatPackage):
            raise TypeError(
                f"To create a {self.name} from a {type(package)!r} use {self.name}.from_package(package)"
            )
        else:
            raise ValueError(
                f"Expected a path or a frictionless.Package, not {type(package)!r}"
            )
        if self.package_name is None:
            if not fl_package.name:
                raise ValueError(
                    f"{self.name}.package_name is not set and the given package has no name."
                )
            self.package_name = fl_package.name
        if fl_basepath is None:
            if self.basepath:
                fl_package.basepath = self.basepath
                self.logger.info(
                    f"The missing basepath of {fl_package.name!r} was set to {self.basepath!r}."
                )
            # else:
            #     default_basepath = self.get_basepath()
            #     fl_package.basepath = default_basepath
            #     self.logger.info(f"The missing basepath of {self.package_name!r} was set to the default "
            #                      f"{default_basepath!r}.")
        else:
            if self.basepath:
                if fl_basepath != self.basepath:
                    raise ValueError(
                        f"The basepath of the given package {fl_basepath!r} does not match the one of the "
                        f"DimcatPackage {self.basepath!r}."
                    )
            else:
                self.basepath = fl_basepath
                self.logger.info(
                    f"The missing basepath of {self.package_name!r} was set to the one from the "
                    f"frictionless.Package {fl_basepath!r}."
                )
        package_dp = self.get_descriptor_path()
        if descriptor_path and os.path.isfile(descriptor_path):
            if os.path.isfile(package_dp):
                if package_dp != descriptor_path:
                    raise ValueError(
                        f"The descriptor path of the given package {descriptor_path!r} does not match the one of the "
                        f"DimcatPackage {package_dp!r}."
                    )
                fl_package.metadata_descriptor_path = package_dp
            else:
                self.descriptor_filepath = descriptor_path
                self.logger.info(
                    f"The descriptor filepath of {self.package_name!r} is to be set to the one from "
                    f"the frictionless.Package {descriptor_path!r}."
                )
                fl_package.metadata_descriptor_path = descriptor_path
        elif os.path.isfile(package_dp):
            raise ValueError(
                f"The package has been setup pointing to the existing descriptor path {package_dp!r}. "
                f"But the given package seems to be independent and might clash with what is being described. "
                f"To create a {self.name} from the descriptor, use {self.name}({package_dp!r})."
            )
        # else:
        #     self.logger.info(f"The descriptor path of {self.package_name!r} was set to the default "
        #                      f"{package_dp!r} which does not exist yet.")
        #     fl_package.metadata_descriptor_path = package_dp
        self._package = fl_package
        if self.basepath is not None:
            self.descriptor_filepath = fl_package.metadata_descriptor_path
        for resource in self._package.resources:
            dc_resource = self._handle_resource_paths(resource)
            self.logger.info(
                f"New resource {dc_resource.resource_name!r} has status {dc_resource.status!r}"
            )
            self._resources.append(dc_resource)
        if self.n_resources > 0:
            self._status = PackageStatus.FULLY_SERIALIZED

    @property
    def package_name(self) -> str:
        return self._package.name

    @package_name.setter
    def package_name(self, package_name: str) -> None:
        check_name(package_name)
        self._package.name = package_name
        if self._descriptor_filepath is not None:
            self._descriptor_filepath = os.path.join(
                self.basepath, package_name + ".datapackage.json"
            )

    @property
    def resources(self) -> List[DimcatResource]:
        return self._resources

    @property
    def resource_names(self) -> List[str]:
        return self._package.resource_names

    @property
    def status(self) -> PackageStatus:
        return self._status

    @property
    def zip_file_exists(self) -> bool:
        return os.path.isfile(self.get_zip_path())

    def add_resource(
        self,
        resource: DimcatResource,
        how: AddingBehaviour = AddingBehaviour.RAISE,
    ) -> None:
        """Adds a resource to the package."""
        if not isinstance(resource, DimcatResource):
            raise TypeError(f"Expected a DimcatResource, not {type(resource)!r}")
        if resource.resource_name in self.resource_names:
            raise ValueError(
                f"Resource with name {resource.resource_name!r} already exists."
            )
        resource = self._handle_resource_paths(resource, how=how)
        if resource.is_loaded:
            resource._status = ResourceStatus.PACKAGED_LOADED
            self.logger.debug(
                f"Status of resource {resource.resource_name!r} was set to {resource.status!r}."
            )
        else:
            resource._status = ResourceStatus.PACKAGED_NOT_LOADED
            self.logger.debug(
                f"Status of resource {resource.resource_name!r} was set to {resource.status!r}."
            )
        self._resources.append(resource)
        self._package.add_resource(resource.resource)
        if self.status == PackageStatus.PARTIALLY_SERIALIZED:
            return
        if resource.is_serialized:
            if self.status in (PackageStatus.EMPTY, PackageStatus.FULLY_SERIALIZED):
                self._status = PackageStatus.FULLY_SERIALIZED
            elif self.status == PackageStatus.NOT_SERIALIZED:
                self._status = PackageStatus.PARTIALLY_SERIALIZED
            else:
                raise NotImplementedError(f"Unexpected status {self.status!r}")
        else:
            if self.status in (PackageStatus.EMPTY, PackageStatus.NOT_SERIALIZED):
                self._status = PackageStatus.NOT_SERIALIZED
            elif self.status == PackageStatus.FULLY_SERIALIZED:
                self._status = PackageStatus.PARTIALLY_SERIALIZED
            else:
                raise NotImplementedError(f"Unexpected status {self.status!r}")

    def copy(self) -> Self:
        """Returns a copy of the package."""
        return self.from_package(self)

    def extend(self, resources: Iterable[DimcatResource]) -> None:
        """Adds multiple resources to the package."""
        status_before = self.status
        resources = tuple(resources)
        if len(resources) == 0:
            self.logger.debug("Nothing to add.")
            return
        for n_added, resource in enumerate(resources, 1):
            self.add_resource(resource.copy())
        self.logger.info(
            f"Package {self.package_name!r} was extended with {n_added} resources to a total "
            f"of {self.n_resources}."
        )
        status_after = self.status
        if status_before != status_after:
            self.logger.debug(
                f"Status changed from {status_before!r} to {status_after!r}"
            )

    def extract_feature(self, feature: FeatureSpecs) -> Feature:
        feature_config = feature_specs2config(feature)
        feature_name = FeatureName(feature_config.options_dtype)
        name2resource = dict(
            Notes="notes",
            Annotations="expanded",
            KeyAnnotations="expanded",
            Metadata="metadata",
        )
        try:
            resource_name = name2resource[feature_name]
        except KeyError:
            raise NoMatchingResourceFoundError(feature_config)
        try:
            resource = self.get_resource(resource_name)
        except ResourceNotFoundError:
            raise NoMatchingResourceFoundError(feature_config)
        Constructor = feature_name.get_class()
        return Constructor.from_resource(resource)

    def get_basepath(self) -> str:
        """Get the basepath of the resource. If not specified, the default basepath is returned."""
        if not self.basepath:
            return get_default_basepath()
        return self.basepath

    def get_descriptor_path(self, create_if_necessary=False) -> Optional[str]:
        """Returns the path to the descriptor file."""
        descriptor_path = os.path.join(
            self.get_basepath(), self.get_descriptor_filepath()
        )
        if not os.path.isfile(descriptor_path) and create_if_necessary:
            self._store_descriptor(descriptor_path)
        return descriptor_path

    def get_descriptor_filepath(self) -> str:
        """Like :attr:`descriptor_filepath` but returning a default value if None."""
        if self.descriptor_filepath is not None:
            return self.descriptor_filepath
        if self.package_name:
            descriptor_filepath = f"{self.package_name}.datapackage.json"
        else:
            descriptor_filepath = "datapackage.json"
        return descriptor_filepath

    def get_feature(self, feature: FeatureSpecs) -> Feature:
        """Checks if the package includes a feature matching the specs, and extracts it otherwise, if possible.

        Raises:
            NoMatchingResourceFoundError:
                If none of the previously extracted features matches the specs and none of the input resources
                allows for extracting a matching feature.
        """
        feature_config = feature_specs2config(feature)
        try:
            return self.get_resource_by_config(feature_config)
        except NoMatchingResourceFoundError:
            pass
        return self.extract_feature(feature_config)

    def get_metadata(self) -> SomeDataframe:
        """Returns the metadata of the package."""
        return self.get_resource("metadata").df

    def get_resource_by_config(self, config: DimcatConfig) -> DimcatResource:
        """Returns the first resource that matches the given config."""
        for resource in self.resources:
            resource_config = resource.to_config()
            if resource_config.matches(config):
                self.logger.debug(
                    f"Requested config {config!r} matched with {resource_config!r}."
                )
                return resource
        raise NoMatchingResourceFoundError(config)

    def get_zip_filepath(self) -> str:
        """Returns the path of the ZIP file that the resources of this package are serialized to."""
        descriptor_filepath = self.get_descriptor_filepath()
        if descriptor_filepath == "datapackage.json":
            zip_filename = f"{self.package_name}.zip"
        elif descriptor_filepath.endswith(
            ".datapackage.json"
        ) or descriptor_filepath.endswith(".datapackage.yaml"):
            zip_filename = f"{descriptor_filepath[:-17]}.zip"
        return zip_filename

    def get_zip_path(self) -> str:
        """Returns the path of the ZIP file that the resources of this package are serialized to."""
        zip_filename = self.get_zip_filepath()
        return os.path.join(self.get_basepath(), zip_filename)

    def get_resource(self, name: Optional[str] = None) -> DimcatResource:
        """Returns the DimcatResource with the given name. If no name is given, returns the last resource.

        Raises:
            EmptyPackageError: If the package is empty.
            ResourceNotFoundError: If the resource with the given name is not found.
        """
        if self.n_resources == 0:
            raise EmptyPackageError(self.package_name)
        if name is None:
            return self._resources[-1]
        for resource in self._resources:
            if resource.resource_name == name:
                return resource
        raise ResourceNotFoundError(name, self.package_name)

    def _handle_resource_paths(
        self,
        resource: DimcatResource | fl.Resource,
        how: AddingBehaviour = AddingBehaviour.RAISE,
    ) -> DimcatResource:
        """If package paths haven't been set, use those from this resource that is to be added to the package.
        Otherwise, make sure the resource paths are compatible.
        """
        how = AddingBehaviour(how)
        if how != AddingBehaviour.RAISE:
            raise NotImplementedError
        is_dimcat_resource = isinstance(resource, DimcatResource)
        if is_dimcat_resource:
            fl_resource = resource.resource
        else:
            fl_resource = resource
        resource_is_serialized = fl_resource.basepath and os.path.isfile(
            fl_resource.normpath
        )
        if resource_is_serialized:
            if not fl_resource.path.endswith(".zip"):
                raise NotImplementedError(
                    f"Currently only zipped resources are supported. {fl_resource.name!r} is stored at "
                    f"{fl_resource.normpath}."
                )
            if not fl_resource.innerpath:
                raise ValueError(
                    f"The resource {fl_resource.name!r} is stored at {fl_resource.normpath} but has no innerpath."
                )
            if self.basepath is None:
                self.basepath = fl_resource.basepath
                self.logger.info(
                    f"The missing basepath of {self.package_name!r} was set to the one from the "
                    f"frictionless.Package {fl_resource.basepath!r}."
                )
            if fl_resource.basepath != self.basepath:
                new_filepath = make_rel_path(fl_resource.normpath, self.basepath)
                self.logger.info(
                    f"Adapting basepath and filepath of {fl_resource.name!r} from {fl_resource.basepath!r} / "
                    f"{fl_resource.path}  to {self.basepath} / {new_filepath}."
                )
                fl_resource.basepath = self.basepath
                fl_resource.path = new_filepath
        else:
            fl_resource.basepath = self.basepath
            fl_resource.path = self.get_zip_filepath()
        if is_dimcat_resource:
            return resource
        dc_resource = DimcatResource(
            resource=fl_resource,
            descriptor_filepath=self.descriptor_filepath,
        )
        return dc_resource

    def make_new_resource(
        self,
        resource: Optional[DimcatResource | fl.Resource | str] = None,
        resource_name: Optional[str] = None,
        df: Optional[D] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        auto_validate: bool = False,
    ) -> DimcatResource:
        """Adds a resource to the package. Parameters are passed to :class:`DimcatResource`."""
        if sum(x is not None for x in (resource, df)) == 2:
            raise ValueError("Pass either a DimcatResource or a dataframe, not both.")
        if df is not None:
            new_resource = DimcatResource.from_dataframe(
                df=df,
                resource_name=resource_name,
                basepath=basepath,
                filepath=filepath,
                column_schema=column_schema,
                auto_validate=auto_validate,
            )
        elif isinstance(resource, DimcatResource):
            new_resource = DimcatResource.from_resource(
                resource=resource,
                resource_name=resource_name,
                basepath=basepath,
                filepath=filepath,
                column_schema=column_schema,
                auto_validate=auto_validate,
            )
        elif resource is None or isinstance(resource, (fl.Resource, str)):
            new_resource = DimcatResource(
                resource=resource,
                resource_name=resource_name,
                basepath=basepath,
                filepath=filepath,
                column_schema=column_schema,
                auto_validate=auto_validate,
            )
        else:
            raise TypeError(
                f"resource is expected to be a resource or a path to a descriptor, not {type(resource)!r}"
            )
        if new_resource.resource_name in self.resource_names:
            raise ValueError(
                f"A DimcatResource {new_resource.resource_name!r} already exists in package {self.package_name!r}."
            )
        return self.add_resource(new_resource)

    def _store_descriptor(
        self, descriptor_path: Optional[str] = None, overwrite=True
    ) -> str:
        """Stores the descriptor to disk based on the package's configuration and returns its path."""
        if descriptor_path is None:
            descriptor_path = self.get_descriptor_path()
        if not overwrite and self.descriptor_exists:
            self.logger.info(
                f"Descriptor exists already and will not be overwritten: {descriptor_path}"
            )
            return descriptor_path
        if descriptor_path.endswith("package.yaml"):
            self._package.to_yaml(descriptor_path)
        elif descriptor_path.endswith("package.json"):
            self._package.to_json(descriptor_path)
        else:
            raise ValueError(
                f"Descriptor path must end with package.yaml or package.json: {descriptor_path}"
            )
        self.descriptor_filepath = descriptor_path
        return descriptor_path

    def validate(self, raise_exception: bool = False) -> fl.Report:
        if self.n_resources != len(self._package.resource_names):
            name = (
                "<unnamed DimcatPackage>"
                if self.package_name is None
                else f"package {self.package_name}"
            )
            raise ValueError(
                f"Number of DimcatResources in {name} ({self.n_resources}) does not match number of resources in "
                f"the wrapped frictionless.Package ({len(self._package.resource_names)})."
            )
        report = self._package.validate()
        if not report.valid and raise_exception:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise fl.FrictionlessException("\n".join(errors))
        return report


PackageSpecs: TypeAlias = Union[DimcatPackage, fl.Package, str]


# endregion DimcatPackage
# region DimcatCatalog


class DimcatCatalog(Data):
    """Has the purpose of collecting and managing a set of :obj:`DimcatPackage` objects.

    Analogous to a :obj:`frictionless.Catalog`, but without intermediate :obj:`frictionless.Dataset` objects.
    Nevertheless, a DimcatCatalog can be stored as and created from a Catalog descriptor (ToDo).
    """

    class Schema(Data.Schema):
        basepath = mm.fields.Str(
            required=False,
            allow_none=True,
            metadata=dict(description="The basepath for all packages in the catalog."),
        )
        packages = mm.fields.List(
            mm.fields.Nested(DimcatPackage.Schema),
            required=False,
            allow_none=True,
            metadata=dict(description="The packages in the catalog."),
        )

    def __init__(
        self,
        basepath: Optional[str] = None,
        packages: Optional[PackageSpecs | List[PackageSpecs]] = None,
    ) -> None:
        """Creates a DimcatCatalog which is essentially a list of :obj:`DimcatPackage` objects.

        Args:
            basepath: The basepath for all packages in the catalog.
        """
        self._packages: List[DimcatPackage] = []
        self._basepath: Optional[str] = None
        if basepath is not None:
            self.basepath = basepath
        if packages is not None:
            self.packages = packages

    def __getitem__(self, item: str) -> DimcatPackage:
        try:
            return self.get_package(item)
        except Exception as e:
            raise KeyError(str(e)) from e

    def __iter__(self) -> Iterator[DimcatPackage]:
        yield from self._packages

    def __len__(self) -> int:
        return len(self._packages)

    def __repr__(self):
        return pformat(self.summary_dict(), sort_dicts=False)

    def __str__(self):
        return pformat(self.summary_dict(), sort_dicts=False)

    @property
    def basepath(self) -> Optional[str]:
        """If specified, the basepath for all packages added to the catalog."""
        if self._basepath is None:
            self._basepath = get_default_basepath()
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str) -> None:
        self.set_basepath(basepath, set_packages=False)

    @property
    def package_names(self) -> List[str]:
        return [package.package_name for package in self._packages]

    @property
    def packages(self) -> List[DimcatPackage]:
        return self._packages

    @packages.setter
    def packages(self, packages: PackageSpecs | List[PackageSpecs]) -> None:
        if len(self._packages) > 0:
            raise ValueError("Cannot set packages if packages are already present.")
        if isinstance(packages, (DimcatPackage, fl.Package, str)):
            packages = [packages]
        for package in packages:
            self.add_package(package.copy())

    def add_package(
        self,
        package: DimcatPackage,
        basepath: Optional[str] = None,
    ):
        """Adds a package to the catalog."""
        if not isinstance(package, DimcatPackage):
            msg = f"{self.name}.add_package() takes a DimcatPackage, not {type(package)!r}."
            raise ValueError(msg)
        if package.package_name in self.package_names:
            msg = f"Package name {package.package_name!r} already exists in DimcatCatalog."
            raise ValueError(msg)
        if basepath is not None:
            package.basepath = basepath
        self._packages.append(package)

    def add_resource(
        self,
        resource: DimcatResource,
        package_name: Optional[str] = None,
    ):
        """Adds a resource to the catalog. If package_name is given, adds the resource to the package with that name."""
        package = self.get_package_by_name(package_name, create=True)
        package.add_resource(resource=resource)

    def check_feature_availability(self, feature: FeatureSpecs) -> bool:
        """Checks whether the given feature is potentially available."""
        return True

    def copy(self) -> Self:
        new_object = self.__class__()
        new_object.packages = self.packages
        return new_object

    def extend(self, catalog: Iterable[DimcatPackage]) -> None:
        """Adds all packages from another catalog to this one."""
        for package in catalog:
            if package.package_name not in self.package_names:
                self.add_package(package.copy())
                continue
            self_package = self.get_package_by_name(package.package_name)
            self_package.extend(package)

    def extend_package(self, package: DimcatPackage) -> None:
        """Adds all resources from the given package to the existing one with the same name."""
        catalog_package = self.get_package_by_name(package.package_name, create=True)
        catalog_package.extend(package)

    def get_package(self, name: Optional[str] = None) -> DimcatPackage:
        """If a name is given, calls :meth:`get_package_by_name`, otherwise returns the last loaded package.

        Raises:
            RuntimeError if no package has been loaded.
        """
        if name is not None:
            return self.get_package_by_name(name=name)
        if len(self._packages) == 0:
            raise EmptyCatalogError
        return self._packages[-1]

    def get_package_by_name(self, name: str, create: bool = False) -> DimcatPackage:
        """

        Raises:
            fl.FrictionlessException if none of the loaded packages has the given name.
        """
        for package in self._packages:
            if package.package_name == name:
                return package
        if create:
            self.make_new_package(
                package_name=name,
                basepath=self.basepath,
            )
            self.logger.info(f"Automatically added new empty package {name!r}")
            return self.get_package()
        raise PackageNotFoundError(name)

    def has_package(self, name: str) -> bool:
        """Returns True if a package with the given name is loaded, False otherwise."""
        for package in self._packages:
            if package.package_name == name:
                return True
        return False

    def make_new_package(
        self,
        package: Optional[PackageSpecs] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ):
        """Adds a package to the catalog. Parameters are the same as for :class:`DimcatPackage`."""
        if package is None or isinstance(package, (fl.Package, str)):
            package = DimcatPackage(
                package_name=package_name,
                basepath=basepath,
                auto_validate=auto_validate,
            )
        elif not isinstance(package, DimcatPackage):
            msg = f"{self.name} takes a Package, not {type(package)!r}."
            raise ValueError(msg)
        self.add_package(package, basepath=basepath)

    def replace_package(self, package: DimcatPackage) -> None:
        """Replaces the package with the same name as the given package with the given package."""
        if not isinstance(package, DimcatPackage):
            msg = f"{self.name}.replace_package() takes a DimcatPackage, not {type(package)!r}."
            raise TypeError(msg)
        for i, p in enumerate(self._packages):
            if p.package_name == package.package_name:
                self.logger.info(
                    f"Replacing package {p.package_name!r} ({p.n_resources} resources) with "
                    f"package {package.package_name!r} ({package.n_resources} resources)"
                )
                self._packages[i] = package
                return
        self.add_package(package)

    def set_basepath(
        self,
        basepath: str,
        set_packages: bool = True,
    ) -> None:
        """Sets the basepath for all packages in the catalog (if set_packages=True)."""
        basepath_arg = resolve_path(basepath)
        if not os.path.isdir(basepath_arg):
            raise ValueError(f"basepath {basepath_arg!r} is not an existing directory.")
        self._basepath = basepath_arg
        if not set_packages:
            return
        for package in self._packages:
            package.basepath = basepath_arg

    def summary_dict(self) -> dict:
        """Returns a summary of the dataset."""
        summary = {p.package_name: p.resource_names for p in self._packages}
        return dict(basepath=self.basepath, packages=summary)


class InputsCatalog(DimcatCatalog):
    def extract_feature(self, feature: FeatureSpecs) -> Feature:
        """Extracts the given features from all packages and combines them in a Feature resource."""
        package = self.get_package()
        feature_config = feature_specs2config(feature)
        return package.extract_feature(feature_config)

    def get_feature(self, feature: FeatureSpecs) -> Feature:
        """ToDo: Get features from all packages and merge them."""
        package = self.get_package()
        feature_config = feature_specs2config(feature)
        return package.get_feature(feature_config)

    def get_metadata(self) -> SomeDataframe:
        """Returns a dataframe with all metadata."""
        package = self.get_package()
        return package.get_metadata()


class OutputsCatalog(DimcatCatalog):
    def get_feature(self, feature: FeatureSpecs) -> DimcatResource:
        """Looks up the given feature in the "features" package and returns it.

        Raises:
            PackageNotFoundError: If no package with the name "features" is loaded.
            NoMatchingResourceFoundError: If no resource matching the specs is found in the "features" package.
        """
        package = self.get_package_by_name("features")
        feature_config = feature_specs2config(feature)
        return package.get_resource_by_config(feature_config)

    def iter_resources(self) -> Iterator[Tuple[str, DimcatResource]]:
        """Iterates over all resources in all packages.

        Yields:
            The package name and the resource.
        """
        for package in self._packages:
            for resource in package:
                yield package.package_name, resource


# endregion DimcatCatalog
# region Dataset


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

    @classmethod
    def from_catalogs(
        cls,
        inputs: DimcatCatalog | List[DimcatPackage],
        outputs: DimcatCatalog | List[DimcatPackage],
        pipeline: Optional[Pipeline] = None,
        **kwargs,
    ):
        """Instantiate by copying existing catalogs."""
        new_dataset = cls(**kwargs)
        if pipeline is not None:
            new_dataset._pipeline = pipeline
        new_dataset.inputs.basepath = inputs.basepath
        new_dataset.outputs.basepath = outputs.basepath
        new_dataset.inputs.extend(inputs)
        new_dataset.outputs.extend(outputs)
        return new_dataset

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        """Instantiate from this Dataset by copying its fields, empty fields otherwise."""
        return cls.from_catalogs(
            inputs=dataset.inputs,
            outputs=dataset.outputs,
            pipeline=dataset.pipeline,
            **kwargs,
        )

    class Schema(Data.Schema):
        """Dataset serialization schema."""

        inputs = mm.fields.Nested(DimcatCatalog.Schema, required=True)
        outputs = mm.fields.Nested(
            DimcatCatalog.Schema, required=False, load_default=[]
        )
        pipeline = (
            DimcatObjectField()
        )  # mm.fields.Nested(Pipeline.Schema) would cause circular import

        @mm.post_load
        def init_object(self, data, **kwargs) -> Dataset:
            return Dataset.from_catalogs(
                inputs=data["inputs"],
                outputs=data["outputs"],
            )

    def __init__(
        self,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of.

        Args:
            **kwargs: Dataset is cooperative and calls super().__init__(data=dataset, **kwargs)
        """
        if basepath is None:
            self._inputs = InputsCatalog()
            self._outputs = OutputsCatalog()
        else:
            basepath_arg = resolve_path(basepath)
            if not os.path.isdir(basepath_arg):
                raise NotADirectoryError(
                    f"basepath {basepath_arg!r} is not an existing directory."
                )
            self._inputs = InputsCatalog(basepath=basepath_arg)
            self._outputs = OutputsCatalog(basepath=basepath_arg)
        self._pipeline = None
        self.reset_pipeline()
        super().__init__(**kwargs)  # calls the Mixin's __init__

    def __repr__(self):
        return self.info(return_str=True)

    def __str__(self):
        return self.info(return_str=True)

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

    def apply(
        self,
        step: FeatureStep,
    ) -> Self:
        """Applies a pipeline step to the features it is configured for or, if None, to all active features."""
        return step.process_dataset(self)

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

    def _extract_feature(self, feature_config: DimcatConfig) -> Feature:
        """Extracts a feature from this Dataset.

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
        return self._pipeline.process_resource(extracted)

    def extract_feature(self, feature: FeatureSpecs) -> Feature:
        """Extracts a feature from this Dataset, adds it to the OutputsCatalog, and adds the corresponding
        FeatureExtractor to the dataset's pipeline as if it had been applied.

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

    def get_feature(self, feature: FeatureSpecs) -> Feature:
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

    @overload
    def info(self, return_str: Literal[False]) -> None:
        ...

    @overload
    def info(self, return_str: Literal[True]) -> str:
        ...

    def info(self, return_str: bool = False) -> Optional[str]:
        """Returns a summary of the dataset."""
        summary = self.summary_dict()
        title = self.name
        title += f"\n{'=' * len(title)}\n"
        summary_str = f"{title}{pformat(summary, sort_dicts=False)}"
        if return_str:
            return summary_str
        print(summary_str)

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

    def make_features_package(
        self,
        features: FeatureSpecs | Iterable[FeatureSpecs] = None,
    ) -> DimcatPackage:
        """Returns a DimcatPackage containing the requested or currently active features.

        Args:
            features:

        Returns:

        """
        if not features:
            if self.n_active_features == 0:
                raise NoFeaturesActiveError
            return self.outputs.get_package_by_name("features")
        new_package = DimcatPackage(package_name="features")
        for feature in self.iter_features(features):
            new_package.add_resource(feature)
        return new_package

    def get_metadata(self) -> SomeDataframe:
        metadata = self.inputs.get_metadata()
        return metadata

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
        package = DimcatPackage(package=package, **options)
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

    def load_feature(self, feature: FeatureSpecs) -> Feature:
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

    def summary_dict(self) -> str:
        """Returns a summary of the dataset."""
        summary = dict(
            inputs=self.inputs.summary_dict(),
            outputs=self.outputs.summary_dict(),
            pipeline=[step.name for step in self._pipeline],
        )
        return summary


# endregion Dataset
