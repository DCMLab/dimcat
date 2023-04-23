from dataclasses import dataclass

from dimcat.features.base import Feature, FeatureConfig, FeatureName


@dataclass(frozen=True)
class AnnotationsConfig(FeatureConfig):
    _configured_class = FeatureName.Annotations


class Annotations(Feature):
    _config_type = AnnotationsConfig


@dataclass(frozen=True)
class KeyAnnotationsConfig(AnnotationsConfig):
    _configured_class = FeatureName.KeyAnnotations


class KeyAnnotations(Annotations):
    _config_type = KeyAnnotationsConfig
