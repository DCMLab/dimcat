from __future__ import annotations

import logging
import os
import re
from typing import List, MutableMapping, Optional, Union

import frictionless as fl
import marshmallow as mm
from dimcat.base import Data, DimcatConfig
from dimcat.resources.base import (
    NEVER_STORE_UNVALIDATED_DATA,
    DimcatResource,
    get_default_basepath,
)
from dimcat.resources.features import Feature, FeatureName
from frictionless.settings import NAME_PATTERN as FRICTIONLESS_NAME_PATTERN
from ms3 import resolve_dir

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
        package: Optional[Union[fl.Package, str]] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """

        Args:
            package:
                A frictionless package descriptor or path to a package descriptor. If not specified, a new package
                is created and ``package_name`` needs to be specified.
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
        try:
            return self._package.get_resource(name)
        except fl.FrictionlessException:
            msg = (
                f"Resource {name!r} not found in package {self.package_name!r}. "
                f"Available resources: {self.resource_names!r}."
            )
            raise fl.FrictionlessException(msg)

    def get_resource(self, name: str) -> DimcatResource:
        fl_resource = self._get_fl_resource(name)
        return DimcatResource(resource=fl_resource)

    def validate(self, raise_exception: bool = False) -> fl.Report:
        report = self._package.validate()
        if not report.valid and raise_exception:
            errors = [err.message for task in report.tasks for err in task.errors]
            raise fl.FrictionlessException("\n".join(errors))
        return report


class DimcatCatalog(Data):
    """Has the purpose of collecting and managing a set of :obj:`DimcatPackage` objects.

    Analogous to a :obj:`frictionless.Catalog`, but without intermediate :obj:`frictionless.Dataset` objects.
    Nevertheless, a DimcatCatalog can be stored as and created from a Catalog descriptor (ToDo).
    """

    def __init__(self) -> None:
        self.packages: List[DimcatPackage] = []

    def __str__(self):
        return str(self.packages)

    def __repr__(self):
        return repr(self.packages)

    @property
    def package_names(self):
        return [p.name for p in self.packages]

    def add_package(self, package: Union[DimcatPackage, fl.Package]):
        if isinstance(package, fl.Package):
            package = DimcatPackage(package_name=package)
        elif not isinstance(package, DimcatPackage):
            obj_name = type(self).__name__
            p_type = type(package)
            msg = f"{obj_name} takes a Package, not {p_type}."
            raise ValueError(msg)
        self.packages.append(package)

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
        if len(self.packages) == 0:
            raise RuntimeError("No data has been loaded.")
        return self.packages[0]

    def get_package_by_name(self, name: str) -> DimcatPackage:
        """

        Raises:
            fl.FrictionlessException if none of the loaded packages has the given name.
        """
        for package in self.packages:
            if package.name == name:
                return package
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
