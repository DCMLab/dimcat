from __future__ import annotations

from dimcat.data.catalogs.base import DimcatCatalog
from dimcat.data.resources.base import SomeDataframe
from dimcat.data.resources.dc import Feature, FeatureSpecs
from dimcat.data.resources.utils import feature_specs2config


class InputsCatalog(DimcatCatalog):
    def extract_feature(self, feature: FeatureSpecs) -> Feature:
        """Extracts the given features from all packages and combines them in a Feature resource."""
        package = self.get_package()
        return package.extract_feature(feature)

    def get_feature(self, feature: FeatureSpecs) -> Feature:
        """ToDo: Get features from all packages and merge them."""
        package = self.get_package()
        feature_config = feature_specs2config(feature)
        return package.get_feature(feature_config)

    def get_metadata(self) -> SomeDataframe:
        """Returns a dataframe with all metadata."""
        package = self.get_package()
        return package.get_metadata()
