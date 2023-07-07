from __future__ import annotations

import logging
from typing import Iterable

from dimcat.base import DimcatConfig
from dimcat.data.dataset.base import Dataset
from dimcat.data.resources.base import DimcatResource
from marshmallow import fields, validate

from .base import FeatureProcessingStep

logger = logging.getLogger(__name__)


class FeatureExtractor(FeatureProcessingStep):
    output_package_name = "features"
    requires_at_least_one_feature = True

    class Schema(FeatureProcessingStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            validate=validate.Length(min=1),
        )

    def _make_new_resource(self, resource: DimcatResource) -> DimcatResource:
        """The extractor receives resources freshly created by :meth:`Dataset.extract_feature`
        (via :meth:`get_features`) and therefore does not need to create a new resource.
        """
        return resource

    def _iter_features(self, dataset: Dataset) -> Iterable[DimcatResource]:
        features = self.get_feature_specs()
        return [dataset._extract_feature(feature) for feature in features]
