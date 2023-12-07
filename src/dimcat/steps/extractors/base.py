from __future__ import annotations

from typing import Iterable, Type

from dimcat.base import DimcatConfig
from dimcat.data.datasets.base import Dataset
from dimcat.data.resources import Feature, FeatureName
from dimcat.data.resources.base import DR
from dimcat.data.resources.dc import DimcatResource
from dimcat.dc_exceptions import ResourceNotProcessableError
from dimcat.steps.base import FeatureProcessingStep
from marshmallow import fields, validate


class FeatureExtractor(FeatureProcessingStep):
    _output_package_name = "features"
    _requires_at_least_one_feature = True

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
        if (
            resource.name != feature_name
            and feature_name not in resource.extractable_features
        ):
            raise ResourceNotProcessableError(
                resource.resource_name, feature_name, resource.dtype
            )
        return feature_name.get_class()

    def _make_new_resource(self, resource: DimcatResource) -> DR:
        """Dispatch the passed resource to the appropriate method."""
        resource_constructor = self._get_new_resource_type(resource)
        if resource.__class__ == resource_constructor:
            # this is the typical case when the FeatureExtractor simply iterates over features that have been
            # extracted from the Dataset using Dataset._extract_feature(). In this case there is no need to create a
            # copy.
            return resource
        # when we get here it's probably because process_resource() was called directly
        resource_name = self.resource_name_factory(resource)
        new_resource = resource_constructor.from_resource(
            resource, resource_name=resource_name
        )
        self.logger.debug(
            f"Created new resource {new_resource} of type {resource_constructor.name}."
        )
        return new_resource

    def _iter_features(self, dataset: Dataset) -> Iterable[DimcatResource]:
        features = self.get_feature_specs()
        return [dataset._extract_feature(feature) for feature in features]
