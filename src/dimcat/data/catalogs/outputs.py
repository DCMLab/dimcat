from __future__ import annotations

from typing import Iterator, Optional, Tuple

from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.resources.base import F
from dimcat.data.resources.dc import DimcatResource, FeatureSpecs
from dimcat.data.resources.utils import feature_specs2config


class OutputsCatalog(DimcatCatalog):
    def get_feature(self, feature: Optional[FeatureSpecs] = None) -> F:
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
