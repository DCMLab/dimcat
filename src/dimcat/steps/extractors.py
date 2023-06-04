from __future__ import annotations

import logging
from typing import List

from dimcat.base import DimcatConfig
from dimcat.data.resources.features import features_argument2config_list
from marshmallow import fields, validate

from .base import PipelineStep

logger = logging.getLogger(__name__)


class FeatureExtractor(PipelineStep):
    output_package_name = "features"

    class Schema(PipelineStep.Schema):
        features = fields.List(
            fields.Nested(DimcatConfig.Schema),
            validate=validate.Length(min=1),
        )

    @property
    def features(self) -> List[DimcatConfig]:
        return self._features

    @features.setter
    def features(self, features):
        configs = features_argument2config_list(features)
        if len(self._features) > 0:
            self.logger.info(
                f"Previously selected features {self._features} overwritten by {configs}."
            )
        self._features = configs
