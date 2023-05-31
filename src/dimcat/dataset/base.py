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
import re
from enum import IntEnum, auto
from pprint import pformat
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, TypeAlias, Union

import frictionless as fl
import marshmallow as mm
import ms3
from dimcat.base import Data, DimcatConfig
from dimcat.resources.base import D, DimcatResource, SomeDataframe
from dimcat.resources.features import Feature, FeatureName
from dimcat.utils import check_file_path, get_default_basepath
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from typing_extensions import Self

if TYPE_CHECKING:
    from dimcat.resources.results import Result

logger = logging.getLogger(__name__)


def feature_argument2config(
    feature: Union[FeatureName, str, DimcatConfig]
) -> DimcatConfig:
    if isinstance(feature, DimcatConfig):
        return feature
    feature_name = FeatureName(feature)
    return DimcatConfig(dtype=feature_name)


# region DimcatPackage


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

        if basepath is not None:
            basepath = ms3.resolve_dir(basepath)

        if package is not None:
            if isinstance(package, str):
                descriptor_path = check_file_path(
                    package,
                    extensions=(
                        "package.json",
                        "package.yaml",
                    ),  # both datapackage and package are fine
                )
                fl_package = fl.Package(descriptor_path)
                descriptor_dir, descriptor_filepath = os.path.split(descriptor_path)
                self._descriptor_filepath = descriptor_filepath
                if basepath is not None and basepath != descriptor_dir:
                    raise ValueError(
                        f"basepath {basepath!r} does not match the directory of the descriptor {descriptor_path!r}"
                    )
                if not fl_package.basepath:
                    fl_package.basepath = descriptor_dir
            elif isinstance(package, fl.Package):
                fl_package = package
                if not fl_package.resource_names:
                    fl_package.clear_resources()  # otherwise the package might be invalid
            else:
                raise ValueError(
                    f"Expected a path or a frictionless.Package, not {type(package)!r}"
                )
            self._package = fl_package
        if package_name is not None:
            self.package_name = package_name
        if self.package_name is None:
            raise ValueError(
                "package_name needs to specified, otherwise the DimcatPackage cannot be retrieved"
            )
        if basepath is not None:
            self.basepath = basepath
        elif self.basepath is None:
            self.basepath = get_default_basepath()
            self.logger.info(f"Using default basepath: {self.basepath}")
        for resource in self._package.resources:
            if not resource.path.endswith(".zip"):
                raise NotImplementedError(
                    "Currently, the resources in a DimcatPackage need to be zipped."
                )
            self._resources.append(
                DimcatResource(
                    resource=resource,
                    basepath=self.basepath,
                    descriptor_filepath=self._descriptor_filepath,
                )
            )
        if len(self._resources) > 0:
            self._status = PackageStatus.FULLY_SERIALIZED
        if auto_validate:
            self.validate(raise_exception=True)

    def __iter__(self) -> Iterator[DimcatResource]:
        yield from self._resources

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
        basepath = ms3.resolve_dir(basepath)
        if self.status > PackageStatus.NOT_SERIALIZED:
            if basepath == self.basepath:
                return
            state = "partially" if PackageStatus.PARTIALLY_SERIALIZED else "fully"
            raise NotImplementedError(
                f"Cannot change the basepath of a package that has already been  {state} serialized."
            )
        assert os.path.isdir(
            basepath
        ), f"Basepath {basepath!r} is not an existing directory."
        self._package.basepath = basepath

    @property
    def descriptor_filepath(self) -> str:
        """The path to the descriptor file on disk, relative to the basepath. If it hasn't been set, it will be
        generated by :meth:`generate_descriptor_path`."""
        if self._descriptor_filepath is not None:
            return self._descriptor_filepath
        if self.package_name:
            file_name = "datapackage.json"
        else:
            file_name = f"{self.package_name}.datapackage.json"
        self._descriptor_filepath = file_name
        return self._descriptor_filepath

    @property
    def package(self) -> fl.Package:
        return self._package

    @property
    def package_name(self) -> str:
        return self._package.name

    @package_name.setter
    def package_name(self, package_name: str) -> None:
        if not re.match(FRICTIONLESS_NAME_PATTERN, package_name):
            raise ValueError(
                f"Name can only contain [a-z], [0-9], [-._/], and no spaces: {package_name!r}"
            )
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

    def add_resource(self, resource: DimcatResource) -> None:
        """Adds a resource to the package."""
        if not isinstance(resource, DimcatResource):
            raise TypeError(f"Expected a DimcatResource, not {type(resource)!r}")
        if resource.resource_name in self.resource_names:
            raise ValueError(
                f"Resource with name {resource.resource_name!r} already exists."
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
        for resource in resources:
            self.add_resource(resource.copy())
        status_after = self.status
        if status_before == status_after:
            self.logger.debug(
                f"Status changed from {status_before!r} to {status_after!r}"
            )

    def get_descriptor_path(self, only_if_exists=True) -> Optional[str]:
        """Returns the path to the descriptor file."""
        descriptor_path = os.path.join(self.basepath, self.descriptor_filepath)
        if only_if_exists and not os.path.isfile(descriptor_path):
            return
        return descriptor_path

    def get_descriptor_filepath(self, create_if_necessary=True) -> str:
        """Like :attr:`descriptor_filepath` but making sure the descriptor exists at :attr:`basedir`."""
        descriptor_path = self.get_descriptor_path(only_if_exists=False)
        if not os.path.isfile(descriptor_path):
            if create_if_necessary:
                _ = self._store_descriptor()
        return self.descriptor_filepath

    def get_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        feature_config = feature_argument2config(feature)
        feature_name = FeatureName(feature_config.options_dtype)
        name2resource = dict(
            Notes="notes",
            Annotations="expanded",
            KeyAnnotations="expanded",
            Metadata="metadata",
        )
        resource_name = name2resource[feature_name]
        resource = self.get_resource(resource_name)
        Constructor = feature_name.get_class()
        return Constructor(resource=resource.resource)

    def _get_fl_resource(self, name: str) -> fl.Resource:
        """Returns the frictionless resource with the given name."""
        try:
            return self._package.get_resource(name)
        except fl.FrictionlessException:
            msg = (
                f"Resource {name!r} not found in package {self.package_name!r}. "
                f"Available resources: {self.resource_names!r}."
            )
            raise fl.FrictionlessException(msg)

    def get_resource(self, name: Optional[str] = None) -> DimcatResource:
        """Returns the DimcatResource with the given name. If no name is given, returns the last resource."""
        if len(self._resources) == 0:
            raise ValueError(f"Package {self.package_name} is empty.")
        if name is None:
            return self._resources[-1]
        for resource in self._resources:
            if resource.resource_name == name:
                return resource
        raise ValueError(f"Resource {name} not found in package {self.package_name}.")

    def make_new_resource(
        self,
        resource: Optional[DimcatResource | fl.Resource | str] = None,
        resource_name: Optional[str] = None,
        df: Optional[D] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        column_schema: Optional[fl.Schema | str] = None,
        auto_validate: bool = False,
    ) -> None:
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
        self.add_resource(new_resource)

    def _store_descriptor(self, overwrite=True) -> str:
        """Stores the descriptor to disk based on the package's configuration and returns its path."""
        descriptor_path = self.get_descriptor_path(only_if_exists=False)
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.debug(
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
        return descriptor_path

    def validate(self, raise_exception: bool = False) -> fl.Report:
        if len(self._resources) != len(self._package.resource_names):
            name = (
                "<unnamed DimcatPackage>"
                if self.package_name is None
                else f"package {self.package_name}"
            )
            raise ValueError(
                f"Number of DimcatResources in {name} ({len(self._resources)}) does not match number of resources in "
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
            description="The basepath for all packages in the catalog.",
        )
        packages = mm.fields.List(
            mm.fields.Nested(DimcatPackage.Schema),
            required=False,
            allow_none=True,
            description="The packages in the catalog.",
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

    def __iter__(self) -> Iterator[DimcatPackage]:
        yield from self._packages

    def __repr__(self):
        return repr(self._packages)

    def __str__(self):
        return str(self._packages)

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
        package.make_new_resource(resource=resource)

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

    def get_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        p = self.get_package()
        feature_config = feature_argument2config(feature)
        return p.get_feature(feature_config)

    def get_package(self, name: Optional[str] = None) -> DimcatPackage:
        """If a name is given, calls :meth:`get_package_by_name`, otherwise returns the first loaded package.

        Raises:
            RuntimeError if no package has been loaded.
        """
        if name is not None:
            return self.get_package_by_name(name=name)
        if len(self._packages) == 0:
            raise ValueError("Catalog is Empty.")
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
            self.add_package(package_name=name)
            self.logger.info(f"Automatically added new empty package {name!r}")
            return self.get_package()
        error = fl.errors.CatalogError(note=f'package "{name}" does not exist')
        raise fl.FrictionlessException(error)

    def has_package(self, name: str) -> bool:
        """Returns True if a package with the given name is loaded, False otherwise."""
        for package in self._packages:
            if package.package_name == name:
                return True
        return False

    def set_basepath(
        self,
        basepath: str,
        set_packages: bool = True,
    ) -> None:
        """Sets the basepath for all packages in the catalog (if set_packages=True)."""
        basepath_arg = ms3.resolve_dir(basepath)
        if not os.path.isdir(basepath_arg):
            raise ValueError(f"basepath {basepath_arg!r} is not an existing directory.")
        self._basepath = basepath_arg
        if not set_packages:
            return
        for package in self._packages:
            package.basepath = basepath_arg


# endregion DimcatCatalog
# region Dataset


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        """Instantiate from this Dataset by copying its fields, empty fields otherwise."""
        new_dataset = cls(**kwargs)
        new_dataset.inputs.extend(dataset.inputs)
        new_dataset.outputs.extend(dataset.outputs)
        return new_dataset

    class Schema(Data.Schema):
        """Dataset serialization schema."""

        inputs = mm.fields.Nested(DimcatCatalog.Schema, required=True)
        outputs = mm.fields.Nested(
            DimcatCatalog.Schema, required=False, load_default=[]
        )

    def __init__(
        self,
        data: Optional[Dataset] = None,
        **kwargs,
    ):
        """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of.

        Args:
            data: Instantiate from this Dataset by copying its fields, empty fields otherwise.
            **kwargs: Dataset is cooperative and calls super().__init__(data=dataset, **kwargs)
        """
        super().__init__(**kwargs)
        self._inputs = DimcatCatalog()
        self._outputs = DimcatCatalog()

    def __repr__(self):
        return repr(self.inputs)

    def __str__(self):
        return str(self.inputs)

    @property
    def inputs(self) -> DimcatCatalog:
        """The inputs catalog."""
        return self._inputs

    @property
    def outputs(self) -> DimcatCatalog:
        """The outputs catalog."""
        return self._outputs

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

    def copy(self) -> Dataset:
        """Returns a copy of this Dataset."""
        return Dataset.from_dataset(self)

    def get_metadata(self) -> SomeDataframe:
        metadata = self.get_feature("metadata")
        return metadata.df

    def load_package(
        self,
        package: Union[fl.Package, str],
        name: Optional[str] = None,
        **options,
    ):
        package = DimcatPackage(package=package, **options)
        if name is None:
            name = package.name
            assert (
                name is not None
            ), "Descriptor did not contain package name and no name was given."
        else:
            package.name = name
        self.inputs.add_package(package)

    def get_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        # if feature in self._feature_cache:
        #     return self._feature_cache[feature]
        feature_config = feature_argument2config(feature)
        feature = self.inputs.get_feature(feature_config)
        # self._feature_cache[feature_config] = feature
        return feature

    def load_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        feature = self.get_feature(feature)
        feature.load()
        return feature


# endregion Dataset
