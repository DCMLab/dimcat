from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from dimcat.base import get_setting
from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.packages.base import Package, PackageSpecs
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs
from dimcat.data.resources.utils import feature_specs2config


class OutputsCatalog(DimcatCatalog):
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
        super().__init__(basepath=basepath, packages=packages)
        if self.basepath is None:
            default_basepath = get_setting("default_basepath")
            self.logger.info(
                f"Dataset.outputs.basepath has been set to the default_basepath {default_basepath!r}, based on the "
                f"current setting. This is where all output packages will be serialized if not specified otherwise."
            )
            self.basepath = default_basepath

    def get_feature(self, feature: Optional[FeatureSpecs] = None) -> DimcatResource:
        """Looks up the given feature in the "features" package and returns it.

        Raises:
            PackageNotFoundError: If no package with the name "features" is loaded.
            NoMatchingResourceFoundError: If no resource matching the specs is found in the "features" package.
        """
        package = self.get_package_by_name("features")
        if feature is None:
            return package.get_resource_by_name()
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
