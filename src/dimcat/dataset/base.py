from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union
from zipfile import ZipFile

import frictionless as fl
import ms3
import pandas as pd
from dimcat.base import Configuration, Data, WrappedDataframe
from dimcat.features.base import Feature, FeatureConfig, FeatureName
from typing_extensions import Self

logger = logging.getLogger(__name__)


def feature_argument2config(
    feature: Union[FeatureName, str, FeatureConfig]
) -> FeatureConfig:
    if isinstance(feature, FeatureConfig):
        return feature
    feature_name = FeatureName(feature)
    return feature_name.get_config()


class DimcatResource(Data):
    def __init__(self, resource: Optional[fl.Resource] = None) -> None:
        if resource is None:
            self.resource = fl.Resource()
        elif not isinstance(resource, fl.Resource):
            obj_name = type(self).__name__
            r_type = type(resource)
            msg = f"{obj_name} takes a frictionless.Resource, not {r_type}."
            if issubclass(r_type, str):
                msg += f" Try {obj_name}.from_descriptor()"
            raise ValueError(msg)
        else:
            self.resource = resource

    @classmethod
    def from_descriptor(
        cls, descriptor: Union[fl.interfaces.IDescriptor, str], **options
    ) -> Self:
        resource = fl.Resource.from_descriptor(descriptor, **options)
        return cls(resource=resource)

    def __str__(self):
        return str(self.resource)

    def __repr__(self):
        return repr(self.resource)

    def get_pandas(
        self, wrapped=True
    ) -> Union[WrappedDataframe[pd.DataFrame], pd.DataFrame]:
        r = self.resource
        s = r.schema
        if r.normpath.endswith(".zip") or r.compression == "zip":
            zip_file_handler = ZipFile(r.normpath)
            df = ms3.load_tsv(zip_file_handler.open(r.innerpath))
        else:
            raise NotImplementedError()
        if len(s.primary_key) > 0:
            df = df.set_index(s.primary_key)
        if wrapped:
            return WrappedDataframe.from_df(df)
        return df


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

    def get_feature(self, feature: Union[FeatureName, str, FeatureConfig]) -> Feature:
        feature_config = feature_argument2config(feature)
        feature_name = feature_config._configured_class
        name2resource = dict(
            Notes="notes",
            Annotations="expanded",
            KeyAnnotations="expanded",
            Metadata="metadata",
        )
        resource_name = name2resource[feature_name]
        r = self.get_resource(resource_name)
        return r.get_pandas()

    def get_resource(self, name: str) -> DimcatResource:
        resource = self.package.get_resource(name)
        return DimcatResource(resource)


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

    def get_feature(self, feature: Union[FeatureName, str, FeatureConfig]) -> Feature:
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


class DatasetConfig(Configuration):
    _configured_class = "Dataset"


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

    _config_type = DatasetConfig

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
        super().__init__(data=data, **kwargs)
        self.catalog = DimcatCatalog()
        self._feature_cache: Dict[FeatureConfig, Feature] = {}

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

    def get_feature(self, feature: Union[FeatureName, str, FeatureConfig]) -> Feature:
        if feature in self._feature_cache:
            return self._feature_cache[feature]
        feature_config = feature_argument2config(feature)
        feature = self.catalog.get_feature(feature_config)
        self._feature_cache[feature_config] = feature
        return feature

    def __str__(self):
        return str(self.catalog)

    def __repr__(self):
        return repr(self.catalog)
