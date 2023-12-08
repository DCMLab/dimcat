from __future__ import annotations

from typing import Iterable, Iterator, List, Literal, Optional, Type

import frictionless as fl
import marshmallow as mm
from dimcat import DimcatConfig, get_class
from dimcat.data.base import Data
from dimcat.data.packages.base import Package, PackageSpecs
from dimcat.data.resources.base import R, Resource
from dimcat.data.resources.dc import FeatureSpecs
from dimcat.dc_exceptions import (
    DuplicatePackageNameError,
    EmptyCatalogError,
    EmptyPackageError,
    NoMatchingResourceFoundError,
    PackageNotFoundError,
    ResourceNotFoundError,
)
from dimcat.utils import treat_basepath_argument
from frictionless import FrictionlessException
from typing_extensions import Self


class DimcatCatalog(Data):
    """Has the purpose of collecting and managing a set of :obj:`Package` objects.

    Analogous to a :obj:`frictionless.Catalog`, but without intermediate :obj:`frictionless.Dataset` objects.
    Nevertheless, a DimcatCatalog can be stored as and created from a Catalog descriptor (ToDo).
    """

    class PickleSchema(Data.PickleSchema):
        packages = mm.fields.List(
            mm.fields.Nested(Package.Schema),
            allow_none=True,
            metadata=dict(description="The packages in the catalog."),
        )

    class Schema(PickleSchema, Data.Schema):
        pass

    def __init__(
        self,
        basepath: Optional[str] = None,
        packages: Optional[PackageSpecs | List[PackageSpecs]] = None,
    ) -> None:
        """Creates a DimcatCatalog which is essentially a list of :obj:`Package` objects.

        Args:
            basepath: The basepath for all packages in the catalog.
        """
        self._packages: List[Package] = []
        super().__init__(basepath=basepath)
        if packages is not None:
            self.packages = packages

    def __getitem__(self, item: str) -> Package:
        try:
            return self.get_package(item)
        except Exception as e:
            raise KeyError(str(e)) from e

    def __iter__(self) -> Iterator[Package]:
        yield from self._packages

    def __len__(self) -> int:
        return len(self._packages)

    @property
    def basepath(self) -> Optional[str]:
        """If specified, the basepath for all packages added to the catalog."""
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str) -> None:
        new_catalog = self._basepath is None
        self._set_basepath(basepath, set_packages=new_catalog)

    @property
    def package_names(self) -> List[str]:
        return [package.package_name for package in self._packages]

    @property
    def packages(self) -> List[Package]:
        return self._packages

    @packages.setter
    def packages(self, packages: PackageSpecs | List[PackageSpecs]) -> None:
        if len(self._packages) > 0:
            raise ValueError("Cannot set packages if packages are already present.")
        if isinstance(packages, (Package, fl.Package, str)):
            packages = [packages]
        for package in packages:
            try:
                self.add_package(package)
            except FrictionlessException as e:
                self.logger.error(f"Adding the package {package!r} failed with\n{e!r}")

    def add_package(
        self,
        package: PackageSpecs,
        basepath: Optional[str] = None,
        copy: bool = False,
    ):
        """Adds a :obj:`Package` to the catalog."""
        if isinstance(package, fl.Package):
            dc_package = Package.from_descriptor(package)
        elif isinstance(package, str):
            dc_package = Package.from_descriptor_path(package)
        elif isinstance(package, Package):
            if copy:
                dc_package = package.copy()
            else:
                dc_package = package
        else:
            msg = f"{self.name}.add_package() takes a package, not {type(package)!r}."
            raise TypeError(msg)
        if dc_package.package_name in self.package_names:
            raise DuplicatePackageNameError(dc_package.package_name)
        if basepath is not None:
            dc_package.basepath = basepath
        self._packages.append(dc_package)

    def add_resource(
        self,
        resource: Resource,
        package_name: Optional[str] = None,
    ):
        """Adds a resource to the catalog. If package_name is given, adds the resource to the package with that name."""
        package = self.get_package_by_name(package_name, create=True)
        package.add_resource(resource=resource)

    def check_feature_availability(self, feature: FeatureSpecs) -> bool:
        """Checks whether the given feature is potentially available."""
        return True

    def copy(self) -> Self:
        new_object = self.__class__(basepath=self.basepath)
        new_object.packages = self.packages
        return new_object

    def extend(self, catalog: Iterable[Package]) -> None:
        """Adds all packages from another catalog to this one."""
        for package in catalog:
            if package.package_name not in self.package_names:
                self.add_package(package.copy())
                continue
            self_package = self.get_package_by_name(package.package_name)
            self_package.extend(package)

    def extend_package(self, package: Package) -> None:
        """Adds all resources from the given package to the existing one with the same name."""
        catalog_package = self.get_package_by_name(package.package_name, create=True)
        catalog_package.extend(package)

    def get_package(self, name: Optional[str] = None) -> Package:
        """If a name is given, calls :meth:`get_package_by_name`, otherwise returns the last loaded package.

        Raises:
            RuntimeError if no package has been loaded.
        """
        if name is not None:
            return self.get_package_by_name(name=name)
        if len(self._packages) == 0:
            raise EmptyCatalogError
        return self._packages[-1]

    def get_package_by_name(self, name: str, create: bool = False) -> Package:
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

    def get_resource_by_config(self, config: DimcatConfig) -> Resource:
        """Returns the first resource that matches the given config.

        Raises:
            EmptyCatalogError: If the package is empty.
            NoMatchingResourceFoundError: If no resource matching the specs is found in the "features" package.
        """
        if len(self._packages) == 0:
            raise EmptyCatalogError
        for package in self._packages:
            try:
                return package.get_resource_by_config(config)
            except (EmptyPackageError, ResourceNotFoundError):
                pass
        raise NoMatchingResourceFoundError(config)

    def get_resource_by_name(self, name: str) -> R:
        """Returns the Resource with the given name.

        Raises:
            EmptyCatalogError: If the package is empty.
            ResourceNotFoundError: If the resource with the given name is not found.
        """
        if len(self._packages) == 0:
            raise EmptyCatalogError
        for package in self._packages:
            try:
                return package.get_resource_by_name(name=name)
            except (EmptyPackageError, ResourceNotFoundError):
                pass
        raise ResourceNotFoundError(name, self.catalog_name)

    def get_resources_by_regex(self, regex: str) -> List[Resource]:
        """Returns the Resource objects whose names contain the given regex."""
        result = []
        for package in self._packages:
            result.extend(package.get_resources_by_regex(regex=regex))
        return result

    def get_resources_by_type(
        self,
        resource_type: Type[Resource] | str,
    ) -> List[Resource]:
        """Returns the Resource objects of the given type."""
        if isinstance(resource_type, str):
            resource_type = get_class(resource_type)
        results = []
        for package in self._packages:
            results.extend(package.get_resources_by_type(resource_type=resource_type))
        return results

    def has_package(self, name: str) -> bool:
        """Returns True if a package with the given name is loaded, False otherwise."""
        for package in self._packages:
            if package.package_name == name:
                return True
        return False

    def iter_resources(self):
        """Iterates over all resources in all packages."""
        for package in self:
            for resource in package:
                yield resource

    def make_new_package(
        self,
        package: Optional[PackageSpecs] = None,
        package_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
    ):
        """Adds a package to the catalog. Parameters are the same as for :class:`Package`."""
        if package is None or isinstance(package, (fl.Package, str)):
            package = Package(
                package_name=package_name,
                basepath=basepath,
                auto_validate=auto_validate,
            )
        elif not isinstance(package, Package):
            msg = f"{self.name} takes a Package, not {type(package)!r}."
            raise ValueError(msg)
        self.add_package(package, basepath=basepath)

    def replace_package(self, package: Package) -> None:
        """Replaces the package with the same name as the given package with the given package."""
        if not isinstance(package, Package):
            msg = (
                f"{self.name}.replace_package() takes a Package, not {type(package)!r}."
            )
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

    def _set_basepath(
        self,
        basepath: str | Literal[None],
        set_packages: bool = True,
    ) -> None:
        """Sets the basepath for all packages in the catalog (if set_packages=True)."""
        self._basepath = treat_basepath_argument(basepath, self.logger)
        if not set_packages:
            return
        for package in self._packages:
            package.basepath = self.basepath

    def summary_dict(self, include_type: bool = True) -> dict:
        """Returns a summary of the dataset."""
        if include_type:
            summary = {
                p.package_name: [f"{r.resource_name!r} ({r.dtype})" for r in p]
                for p in self._packages
            }
        else:
            summary = {p.package_name: p.resource_names for p in self._packages}
        return dict(basepath=self.basepath, packages=summary)
