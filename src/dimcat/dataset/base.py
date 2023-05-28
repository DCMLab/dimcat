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
from typing import TYPE_CHECKING, List, MutableMapping, Optional, TypeAlias, Union

import frictionless as fl
import marshmallow as mm
import ms3
from dimcat.base import Data, DimcatConfig
from dimcat.resources.base import (
    NEVER_STORE_UNVALIDATED_DATA,
    D,
    DimcatResource,
    get_default_basepath,
)
from dimcat.resources.features import Feature, FeatureName
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from ms3 import resolve_dir

if TYPE_CHECKING:
    from dimcat.analyzers.base import Result

logger = logging.getLogger(__name__)


def feature_argument2config(
    feature: Union[FeatureName, str, DimcatConfig]
) -> DimcatConfig:
    if isinstance(feature, DimcatConfig):
        return feature
    feature_name = FeatureName(feature)
    return DimcatConfig(dtype=feature_name)


class DimcatPackage(Data):
    """Wrapper for a :obj:`frictionless.Package`. The purpose of a Package is to create, load, and store a collection
    of :obj:`DimcatResource` objects. The preferred way of storing a package is a ``[name.]datapackage.json``
    descriptor and a .zip file containing one .tsv file per DimcatResource contained in the package.
    """

    class Schema(Data.Schema):
        package = mm.fields.Method(
            serialize="get_package_descriptor", deserialize="load_descriptor"
        )

        def get_package_descriptor(self, obj: DimcatResource):
            descriptor = obj._package.to_descriptor()
            return descriptor

        def load_descriptor(self, data):
            if isinstance(data, str):
                return fl.Package(data)
            if "resources" not in data:
                return fl.Package(**data)
            return fl.Package.from_descriptor(data)

    def __init__(
        self,
        package: Optional[DimcatPackage | fl.Package | MutableMapping | str] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        validate: bool = True,
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

            validate:
                By default, the package is validated everytime a resource is added. Set to False to disable this.
        """
        if sum(arg is None for arg in (package_name, package)) == 2:
            raise ValueError(
                "At least one of package_name and package needs to be specified."
            )
        self._package = fl.Package()
        self.resources: List[DimcatResource] = []
        self.descriptor_path: Optional[str] = None
        if package is not None:
            if isinstance(package, str):
                self.descriptor_path = resolve_dir(package)
                if not os.path.isfile(self.descriptor_path):
                    raise FileNotFoundError(f"{self.descriptor_path} is not a file.")
                if not self.descriptor_path.endswith("datapackage.json"):
                    logging.warning(
                        f"{self.descriptor_path} does not end with datapackage.json. Trying to load it anyway."
                    )
                self._package = fl.Package(self.descriptor_path)
                self.basepath = os.path.dirname(self.descriptor_path)
            elif isinstance(package, fl.Package):
                self._package = package
            elif isinstance(package, MutableMapping):
                self._package = fl.Package(**package)
            else:
                raise ValueError(
                    f"Expected a path or a frictionless.Package, not {type(package)}"
                )
        if package_name is not None:
            self.package_name = package_name
        if basepath is not None:
            self.basepath = basepath
        self._validate = validate
        if validate:
            self.validate(raise_exception=NEVER_STORE_UNVALIDATED_DATA)
        for resource in self._package.resources:
            self.resources.append(
                DimcatResource(resource=resource, basepath=self.basepath)
            )

    @property
    def basepath(self) -> str:
        if self._package.basepath is None:
            if self.descriptor_path is not None:
                self._package.basepath = os.path.dirname(self.descriptor_path)
            else:
                self._package.basepath = get_default_basepath()
        return self._package.basepath

    @basepath.setter
    def basepath(self, basepath: str) -> None:
        self._package.basepath = basepath

    @property
    def package(self) -> fl.Package:
        return self._package

    @property
    def package_name(self) -> str:
        return self._package.name

    @package_name.setter
    def package_name(self, package_name: str) -> None:
        name_lower = package_name.lower()
        if not re.match(FRICTIONLESS_NAME_PATTERN, name_lower):
            raise ValueError(
                f"Name must be lowercase and work as filename: {name_lower!r}"
            )
        self._package.name = name_lower
        if self.descriptor_path is not None:
            self.descriptor_path = os.path.join(
                self.basepath, name_lower + ".datapackage.json"
            )

    @property
    def resource_names(self) -> List[str]:
        return self._package.resource_names

    # def __getattr__(self, key):
    #     """Enables using DimcatPackage like the wrapped :obj:`frictionless.Package`."""
    #     return getattr(self.df, key)
    #
    # def __setattr__(self, key, value):
    #     return setattr(self.package, key, value)

    def __str__(self):
        return str(self._package)

    def __repr__(self):
        return repr(self._package)

    def add_resource(
        self,
        resource: Optional[DimcatResource | fl.Resource | str] = None,
        resource_name: Optional[str] = None,
        df: Optional[D] = None,
        column_schema: Optional[fl.Schema | str] = None,
        basepath: Optional[str] = None,
        filepath: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """Adds a resource to the package. Parameters are the same as for :class:`DimcatResource`."""
        if resource is None or isinstance(resource, (fl.Resource, str)):
            resource = DimcatResource(
                resource=resource,
                resource_name=resource_name,
                basepath=basepath,
                filepath=filepath,
                column_schema=column_schema,
                validate=validate,
            )
        elif not isinstance(resource, DimcatResource):
            msg = f"{self.name} takes a DimcatResource, not {type(resource)}"
            raise ValueError(msg)
        if resource.resource_name in self.resource_names:
            raise ValueError(
                f"Resource {resource.resource_name} already exists in package {self.package_name}."
            )
        self._package.add_resource(resource.resource)
        self.resources.append(resource)

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
        if len(self.resources) == 0:
            raise ValueError(f"Package {self.package_name} is empty.")
        if name is None:
            return self.resources[-1]
        for resource in self.resources:
            if resource.resource_name == name:
                return resource
        raise ValueError(f"Resource {name} not found in package {self.package_name}.")

    def validate(self, raise_exception: bool = False) -> fl.Report:
        report = self._package.validate()
        if not report.valid and raise_exception:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise fl.FrictionlessException("\n".join(errors))
        return report


PackageSpecs: TypeAlias = Union[DimcatPackage, fl.Package, str]


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
    def packages(self) -> List[DimcatPackage]:
        return self._packages

    @packages.setter
    def packages(self, packages: PackageSpecs | List[PackageSpecs]) -> None:
        if len(self._packages) > 0:
            raise ValueError("Cannot set packages if packages are already present.")
        if isinstance(packages, (DimcatPackage, fl.Package, str)):
            packages = [packages]

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

    def __str__(self):
        return str(self._packages)

    def __repr__(self):
        return repr(self._packages)

    @property
    def package_names(self):
        return [p.package_name for p in self._packages]

    def add_package(
        self,
        package: Optional[PackageSpecs] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        validate: bool = True,
    ):
        """Adds a package to the catalog. Parameters are the same as for :class:`DimcatPackage`."""
        if package is None or isinstance(package, (fl.Package, str)):
            package = DimcatPackage(
                package_name=package_name, basepath=basepath, validate=validate
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
        package.add_resource(resource=resource)

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
            logger.info(f"Automatically added new empty package {name!r}")
            return self.get_package()
        error = fl.errors.CatalogError(note=f'package "{name}" does not exist')
        raise fl.FrictionlessException(error)


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

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
        self.inputs = DimcatCatalog()
        self.outputs = DimcatCatalog()

    def add_result(self, result: Result):
        """Adds a result to the outputs catalog."""
        self.add_output(resource=result, package_name="results")

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

    def get_result(self, analyzer_name: Optional[str] = None) -> Result:
        """Returns the result of the previously applied analyzer with the given name."""
        results = self.outputs.get_package("results")
        if analyzer_name is None:
            return results.get_resource()
        raise NotImplementedError("get_result with analyzer_name not implemented yet.")

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

    def __str__(self):
        return str(self.inputs)

    def __repr__(self):
        return repr(self.inputs)
