from __future__ import annotations

from typing import Iterable, Type

from dimcat.base import DimcatConfig
from dimcat.data.datasets.base import Dataset
from dimcat.data.resources import Feature, FeatureName
from dimcat.data.resources.dc import DimcatResource
from dimcat.dc_exceptions import ResourceNotProcessableError
from dimcat.steps.base import FeatureProcessingStep
from marshmallow import fields, validate


class FeatureExtractor(FeatureProcessingStep):
    output_package_name = "features"
    requires_at_least_one_feature = True

    class Schema(FeatureProcessingStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            validate=validate.Length(min=1),
        )

    def _get_new_resource_type(self, resource: DimcatResource) -> Type[Feature]:
        feature_specs = self.get_feature_specs()
        if len(feature_specs) > 1:
            raise NotImplementedError(
                "Extraction of multiple features from a single Resource is not implemented. "
                "Run the Extractor on a Dataset or specify only one feature."
            )
        feature_config = feature_specs[0]
        feature_name = FeatureName(feature_config.options_dtype)
        if feature_name not in resource.extractable_features:
            raise ResourceNotProcessableError(
                resource.resource_name, feature_name, resource.dtype
            )
        return feature_name.get_class()

    def _iter_features(self, dataset: Dataset) -> Iterable[DimcatResource]:
        features = self.get_feature_specs()
        return [dataset._extract_feature(feature) for feature in features]
