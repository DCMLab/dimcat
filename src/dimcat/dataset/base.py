from __future__ import annotations

import logging
from typing import List, Optional, Union

import frictionless as fl
from dimcat.base import Data, DimcatConfig
from dimcat.resources.base import DimcatResource
from dimcat.resources.features import Feature, FeatureName
from typing_extensions import Self

logger = logging.getLogger(__name__)


def feature_argument2config(
    feature: Union[FeatureName, str, DimcatConfig]
) -> DimcatConfig:
    if isinstance(feature, DimcatConfig):
        return feature
    feature_name = FeatureName(feature)
    return DimcatConfig(dtype=feature_name)


class DimcatPackage(Data):
    """Wrapper for a :obj:`frictionless.Package`."""

    def __init__(self, package: Optional[fl.Package] = None) -> None:
        self.package: fl.Package
        if package is None:
            self.package = fl.Package()
        elif not isinstance(package, fl.Package):
            obj_name = type(self).__name__
            p_type = type(package)
            msg = f"{obj_name} takes a frictionless.Package, not {p_type}."
            if issubclass(p_type, str):
                msg += f" Try {obj_name}.from_descriptor()"
            raise ValueError(msg)
        else:
            self.package = package

    @classmethod
    def from_descriptor(
        cls, descriptor: Union[fl.interfaces.IDescriptor, str], **options
    ) -> Self:
        package = fl.Package.from_descriptor(descriptor, **options)
        return cls(package=package)

    # def __getattr__(self, key):
    #     """Enables using DimcatPackage like the wrapped :obj:`frictionless.Package`."""
    #     return getattr(self.df, key)
    #
    # def __setattr__(self, key, value):
    #     return setattr(self.package, key, value)

    def __str__(self):
        return str(self.package)

    def __repr__(self):
        return repr(self.package)

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

    def get_resource(self, name: str) -> DimcatResource:
        fl_resource = self.package.get_resource(name)
        return DimcatResource(resource=fl_resource)


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
            package = DimcatPackage(package)
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
        self.catalog = DimcatCatalog()
        self._feature_cache = DimcatPackage()

    def load_package(
        self,
        descriptor: Union[fl.interfaces.IDescriptor, str],
        name: Optional[str] = None,
        **options,
    ):
        package = DimcatPackage.from_descriptor(descriptor, **options)
        if name is None:
            name = package.name
            assert (
                name is not None
            ), "Descriptor did not contain package name and no name was given."
        else:
            package.name = name
        self.catalog.add_package(package)

    def get_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        # if feature in self._feature_cache:
        #     return self._feature_cache[feature]
        feature_config = feature_argument2config(feature)
        feature = self.catalog.get_feature(feature_config)
        # self._feature_cache[feature_config] = feature
        return feature

    def load_feature(self, feature: Union[FeatureName, str, DimcatConfig]) -> Feature:
        feature = self.get_feature(feature)
        feature.load()
        return feature

    def __str__(self):
        return str(self.catalog)

    def __repr__(self):
        return repr(self.catalog)
